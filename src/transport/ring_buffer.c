#include "vectlink/transport.h"
#include "vectlink/checksum.h"
#include "protocol/endian.h"
#include "transport/atomic_compat.h"

#include <string.h>

#define VTLK_CTRL_OFF_MAGIC       0u
#define VTLK_CTRL_OFF_VERSION     8u
#define VTLK_CTRL_OFF_CAPACITY    12u
#define VTLK_CTRL_OFF_SLOT_SIZE   16u
#define VTLK_CTRL_OFF_MAX_PAYLOAD 20u
#define VTLK_CTRL_OFF_WRITE_SEQ   64u
#define VTLK_CTRL_OFF_READ_SEQ    128u

static const uint8_t vtlk_shm_magic[8] = {
    'V', 'T', 'L', 'K', '_', 'S', 'H', 'M'
};

static int vtlk_is_power_of_two_u32(uint32_t v)
{
    return (v != 0u) && ((v & (v - 1u)) == 0u);
}

static uint64_t *vtlk_ctrl_write_seq_ptr(vtlk_ring_t *ring)
{
    return (uint64_t *)(void *)(ring->base + VTLK_CTRL_OFF_WRITE_SEQ);
}

static const uint64_t *vtlk_ctrl_write_seq_cptr(const vtlk_ring_t *ring)
{
    return (const uint64_t *)(const void *)(ring->base + VTLK_CTRL_OFF_WRITE_SEQ);
}

static uint64_t *vtlk_ctrl_read_seq_ptr(vtlk_ring_t *ring)
{
    return (uint64_t *)(void *)(ring->base + VTLK_CTRL_OFF_READ_SEQ);
}

static const uint64_t *vtlk_ctrl_read_seq_cptr(const vtlk_ring_t *ring)
{
    return (const uint64_t *)(const void *)(ring->base + VTLK_CTRL_OFF_READ_SEQ);
}

size_t vtlk_ring_required_bytes(uint32_t capacity,
                                uint32_t max_payload_bytes,
                                uint32_t *out_slot_size)
{
    uint64_t raw_slot;
    uint64_t aligned_slot;
    size_t data_bytes;

    if (!vtlk_is_power_of_two_u32(capacity)) {
        return 0u;
    }

    raw_slot = (uint64_t)VTLK_HEADER_SIZE + (uint64_t)max_payload_bytes;
    aligned_slot = (raw_slot + (uint64_t)(VTLK_RING_CACHE_LINE - 1u)) &
                   ~((uint64_t)(VTLK_RING_CACHE_LINE - 1u));

    if (aligned_slot > (uint64_t)UINT32_MAX) {
        return 0u;
    }

    if ((size_t)capacity > (SIZE_MAX - (size_t)VTLK_RING_CTRL_SIZE) / (size_t)aligned_slot) {
        return 0u;
    }

    data_bytes = (size_t)capacity * (size_t)aligned_slot;
    if (out_slot_size != NULL) {
        *out_slot_size = (uint32_t)aligned_slot;
    }

    return (size_t)VTLK_RING_CTRL_SIZE + data_bytes;
}

vtlk_status_t vtlk_ring_init(vtlk_ring_t *ring,
                             void *mem,
                             size_t mem_bytes,
                             uint32_t capacity,
                             uint32_t max_payload_bytes)
{
    uint32_t slot_size;
    size_t required;
    uint8_t *base;

    if (ring == NULL || mem == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (((uintptr_t)mem % (uintptr_t)VTLK_RING_CACHE_LINE) != 0u) {
        return VTLK_ERR_INVALID_ARG;
    }

    required = vtlk_ring_required_bytes(capacity, max_payload_bytes, &slot_size);
    if (required == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (mem_bytes < required) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    base = (uint8_t *)mem;
    memset(base, 0, VTLK_RING_CTRL_SIZE);

    memcpy(base + VTLK_CTRL_OFF_MAGIC, vtlk_shm_magic, sizeof(vtlk_shm_magic));
    vtlk_write_u32_le(base + VTLK_CTRL_OFF_VERSION, (uint32_t)VTLK_SHM_VERSION);
    vtlk_write_u32_le(base + VTLK_CTRL_OFF_CAPACITY, capacity);
    vtlk_write_u32_le(base + VTLK_CTRL_OFF_SLOT_SIZE, slot_size);
    vtlk_write_u32_le(base + VTLK_CTRL_OFF_MAX_PAYLOAD, max_payload_bytes);

    ring->base = base;
    ring->total_bytes = required;
    ring->capacity = capacity;
    ring->slot_size = slot_size;
    ring->max_payload_bytes = max_payload_bytes;
    ring->cached_read_seq = 0u;
    ring->cached_write_seq = 0u;
    ring->write_in_flight = 0;
    ring->write_slot_index = 0u;
    ring->write_seq_pending = 0u;
    ring->read_in_flight = 0;
    ring->read_slot_index = 0u;
    ring->read_seq_pending = 0u;

    vtlk_atomic_store_release_u64(vtlk_ctrl_write_seq_ptr(ring), 0u);
    vtlk_atomic_store_release_u64(vtlk_ctrl_read_seq_ptr(ring), 0u);

    return VTLK_OK;
}

vtlk_status_t vtlk_ring_attach(vtlk_ring_t *ring,
                               void *mem,
                               size_t mem_bytes)
{
    const uint8_t *base;
    uint32_t version;
    uint32_t capacity;
    uint32_t slot_size;
    uint32_t max_payload;
    uint32_t expected_slot_size;
    size_t required;

    if (ring == NULL || mem == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (((uintptr_t)mem % (uintptr_t)VTLK_RING_CACHE_LINE) != 0u) {
        return VTLK_ERR_INVALID_ARG;
    }
    if (mem_bytes < VTLK_RING_CTRL_SIZE) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    base = (const uint8_t *)mem;
    if (memcmp(base + VTLK_CTRL_OFF_MAGIC, vtlk_shm_magic, sizeof(vtlk_shm_magic)) != 0) {
        return VTLK_ERR_INVALID_MAGIC;
    }

    version = vtlk_read_u32_le(base + VTLK_CTRL_OFF_VERSION);
    if (version != (uint32_t)VTLK_SHM_VERSION) {
        return VTLK_ERR_UNSUPPORTED_VERSION;
    }

    capacity = vtlk_read_u32_le(base + VTLK_CTRL_OFF_CAPACITY);
    slot_size = vtlk_read_u32_le(base + VTLK_CTRL_OFF_SLOT_SIZE);
    max_payload = vtlk_read_u32_le(base + VTLK_CTRL_OFF_MAX_PAYLOAD);

    required = vtlk_ring_required_bytes(capacity, max_payload, &expected_slot_size);
    if (required == 0u || slot_size != expected_slot_size ||
        (slot_size % VTLK_RING_CACHE_LINE) != 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (mem_bytes < required) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    ring->base = (uint8_t *)mem;
    ring->total_bytes = required;
    ring->capacity = capacity;
    ring->slot_size = slot_size;
    ring->max_payload_bytes = max_payload;
    ring->cached_read_seq = vtlk_atomic_load_acquire_u64(vtlk_ctrl_read_seq_cptr(ring));
    ring->cached_write_seq = vtlk_atomic_load_acquire_u64(vtlk_ctrl_write_seq_cptr(ring));
    ring->write_in_flight = 0;
    ring->write_slot_index = 0u;
    ring->write_seq_pending = 0u;
    ring->read_in_flight = 0;
    ring->read_slot_index = 0u;
    ring->read_seq_pending = 0u;

    return VTLK_OK;
}

vtlk_status_t vtlk_ring_push(vtlk_ring_t *ring,
                             const vtlk_header_t *hdr,
                             const void *payload)
{
    uint64_t write_seq;
    uint64_t slot_idx;
    uint8_t *slot;
    vtlk_header_t local_hdr;
    vtlk_status_t st;

    if (ring == NULL || hdr == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (ring->base == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (ring->read_in_flight != 0) {
        return VTLK_ERR_BUSY;
    }
    if (ring->write_in_flight != 0) {
        return VTLK_ERR_BUSY;
    }
    if (hdr->payload_bytes > ring->max_payload_bytes) {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (hdr->payload_bytes > 0u && payload == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    write_seq = vtlk_atomic_load_acquire_u64(vtlk_ctrl_write_seq_cptr(ring));
    if ((write_seq - ring->cached_read_seq) >= (uint64_t)ring->capacity) {
        ring->cached_read_seq = vtlk_atomic_load_acquire_u64(vtlk_ctrl_read_seq_cptr(ring));
        if ((write_seq - ring->cached_read_seq) >= (uint64_t)ring->capacity) {
            return VTLK_ERR_FULL;
        }
    }

    slot_idx = write_seq & (uint64_t)(ring->capacity - 1u);
    slot = ring->base + VTLK_RING_CTRL_SIZE + (size_t)slot_idx * (size_t)ring->slot_size;

    local_hdr = *hdr;
    local_hdr.payload_crc32c = (local_hdr.payload_bytes > 0u)
                                   ? vtlk_crc32c(payload, local_hdr.payload_bytes)
                                   : 0u;

    st = vtlk_header_encode(&local_hdr, slot, VTLK_HEADER_SIZE);
    if (st != VTLK_OK) {
        return st;
    }

    if (local_hdr.payload_bytes > 0u) {
        memcpy(slot + VTLK_HEADER_SIZE, payload, local_hdr.payload_bytes);
    }

    vtlk_atomic_store_release_u64(vtlk_ctrl_write_seq_ptr(ring), write_seq + 1u);
    return VTLK_OK;
}

vtlk_status_t vtlk_ring_pop(vtlk_ring_t *ring,
                            vtlk_header_t *hdr_out,
                            void *payload_out,
                            size_t payload_out_cap)
{
    uint64_t read_seq;
    uint64_t slot_idx;
    const uint8_t *slot;
    vtlk_header_t tmp_hdr;
    vtlk_status_t st;
    uint32_t actual_crc;

    if (ring == NULL || hdr_out == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (ring->base == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (ring->read_in_flight != 0) {
        return VTLK_ERR_BUSY;
    }
    if (ring->write_in_flight != 0) {
        return VTLK_ERR_BUSY;
    }

    read_seq = vtlk_atomic_load_acquire_u64(vtlk_ctrl_read_seq_cptr(ring));
    if (ring->cached_write_seq == read_seq) {
        ring->cached_write_seq = vtlk_atomic_load_acquire_u64(vtlk_ctrl_write_seq_cptr(ring));
        if (ring->cached_write_seq == read_seq) {
            return VTLK_ERR_EMPTY;
        }
    }

    slot_idx = read_seq & (uint64_t)(ring->capacity - 1u);
    slot = ring->base + VTLK_RING_CTRL_SIZE + (size_t)slot_idx * (size_t)ring->slot_size;

    st = vtlk_header_decode(slot, VTLK_HEADER_SIZE, &tmp_hdr);
    if (st != VTLK_OK) {
        return st;
    }

    if (tmp_hdr.payload_bytes > ring->max_payload_bytes) {
        return VTLK_ERR_INVALID_PARAM;
    }
    if ((size_t)tmp_hdr.payload_bytes > payload_out_cap) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    if (tmp_hdr.payload_bytes > 0u) {
        if (payload_out == NULL) {
            return VTLK_ERR_NULL_PTR;
        }
        actual_crc = vtlk_crc32c(slot + VTLK_HEADER_SIZE, tmp_hdr.payload_bytes);
        if (actual_crc != tmp_hdr.payload_crc32c) {
            return VTLK_ERR_CRC_MISMATCH;
        }
        memcpy(payload_out, slot + VTLK_HEADER_SIZE, tmp_hdr.payload_bytes);
    }

    *hdr_out = tmp_hdr;
    vtlk_atomic_store_release_u64(vtlk_ctrl_read_seq_ptr(ring), read_seq + 1u);
    return VTLK_OK;
}

int vtlk_ring_acquire_write(vtlk_ring_t *ring, vtlk_write_slot_t *out_slot)
{
    uint64_t write_seq;
    uint32_t slot_idx;
    uint8_t *slot;

    if (ring == NULL || out_slot == NULL) {
        return (int)VTLK_ERR_NULL_PTR;
    }
    if (ring->base == NULL || ring->capacity == 0u) {
        return (int)VTLK_ERR_INVALID_PARAM;
    }
    if (ring->write_in_flight != 0) {
        return (int)VTLK_ERR_BUSY;
    }

    write_seq = vtlk_atomic_load_acquire_u64(vtlk_ctrl_write_seq_cptr(ring));
    if ((write_seq - ring->cached_read_seq) >= (uint64_t)ring->capacity) {
        ring->cached_read_seq = vtlk_atomic_load_acquire_u64(vtlk_ctrl_read_seq_cptr(ring));
        if ((write_seq - ring->cached_read_seq) >= (uint64_t)ring->capacity) {
            return (int)VTLK_ERR_FULL;
        }
    }

    slot_idx = (uint32_t)(write_seq & (uint64_t)(ring->capacity - 1u));
    slot = ring->base + VTLK_RING_CTRL_SIZE + (size_t)slot_idx * (size_t)ring->slot_size;

    ring->write_in_flight = 1;
    ring->write_slot_index = slot_idx;
    ring->write_seq_pending = write_seq;

    out_slot->payload = (void *)(slot + VTLK_HEADER_SIZE);
    out_slot->capacity = (size_t)ring->max_payload_bytes;
    out_slot->slot_index = slot_idx;
    return (int)VTLK_OK;
}

int vtlk_ring_commit_write(vtlk_ring_t *ring,
                           const vtlk_write_slot_t *slot,
                           const vtlk_header_t *hdr_template,
                           size_t payload_bytes)
{
    uint8_t *slot_ptr;
    vtlk_header_t local_hdr;
    vtlk_status_t st;

    if (ring == NULL || slot == NULL || hdr_template == NULL) {
        return (int)VTLK_ERR_NULL_PTR;
    }
    if (ring->base == NULL || ring->write_in_flight == 0 ||
        slot->slot_index != ring->write_slot_index) {
        return (int)VTLK_ERR_INVALID_PARAM;
    }
    if (payload_bytes > (size_t)ring->max_payload_bytes) {
        return (int)VTLK_ERR_INVALID_PARAM;
    }

    slot_ptr = ring->base + VTLK_RING_CTRL_SIZE + (size_t)slot->slot_index * (size_t)ring->slot_size;
    if (slot->payload != (const void *)(slot_ptr + VTLK_HEADER_SIZE)) {
        return (int)VTLK_ERR_INVALID_PARAM;
    }

    local_hdr = *hdr_template;
    local_hdr.payload_bytes = (uint32_t)payload_bytes;
    local_hdr.payload_crc32c = (payload_bytes > 0u)
                                   ? vtlk_crc32c(slot->payload, payload_bytes)
                                   : 0u;

    st = vtlk_header_encode(&local_hdr, slot_ptr, VTLK_HEADER_SIZE);
    if (st != VTLK_OK) {
        return (int)st;
    }

    vtlk_atomic_store_release_u64(vtlk_ctrl_write_seq_ptr(ring), ring->write_seq_pending + 1u);
    ring->write_in_flight = 0;
    return (int)VTLK_OK;
}

int vtlk_ring_abort_write(vtlk_ring_t *ring, const vtlk_write_slot_t *slot)
{
    if (ring == NULL || slot == NULL) {
        return (int)VTLK_ERR_NULL_PTR;
    }
    if (ring->base == NULL || ring->write_in_flight == 0 ||
        slot->slot_index != ring->write_slot_index) {
        return (int)VTLK_ERR_INVALID_PARAM;
    }

    ring->write_in_flight = 0;
    return (int)VTLK_OK;
}

int vtlk_ring_acquire_read(vtlk_ring_t *ring, vtlk_read_slot_t *out_slot)
{
    uint64_t read_seq;
    uint32_t slot_idx;
    const uint8_t *slot;
    vtlk_header_t tmp_hdr;
    vtlk_status_t st;
    uint32_t actual_crc;

    if (ring == NULL || out_slot == NULL) {
        return (int)VTLK_ERR_NULL_PTR;
    }
    if (ring->base == NULL || ring->capacity == 0u) {
        return (int)VTLK_ERR_INVALID_PARAM;
    }
    if (ring->read_in_flight != 0) {
        return (int)VTLK_ERR_BUSY;
    }

    read_seq = vtlk_atomic_load_acquire_u64(vtlk_ctrl_read_seq_cptr(ring));
    if (ring->cached_write_seq == read_seq) {
        ring->cached_write_seq = vtlk_atomic_load_acquire_u64(vtlk_ctrl_write_seq_cptr(ring));
        if (ring->cached_write_seq == read_seq) {
            return (int)VTLK_ERR_EMPTY;
        }
    }

    slot_idx = (uint32_t)(read_seq & (uint64_t)(ring->capacity - 1u));
    slot = ring->base + VTLK_RING_CTRL_SIZE + (size_t)slot_idx * (size_t)ring->slot_size;

    st = vtlk_header_decode(slot, VTLK_HEADER_SIZE, &tmp_hdr);
    if (st != VTLK_OK) {
        return (int)st;
    }

    if (tmp_hdr.payload_bytes > ring->max_payload_bytes) {
        return (int)VTLK_ERR_INVALID_PARAM;
    }

    if (tmp_hdr.payload_bytes > 0u) {
        actual_crc = vtlk_crc32c(slot + VTLK_HEADER_SIZE, tmp_hdr.payload_bytes);
        if (actual_crc != tmp_hdr.payload_crc32c) {
            return (int)VTLK_ERR_CRC_MISMATCH;
        }
    }

    out_slot->header = tmp_hdr;
    out_slot->payload = (const void *)(slot + VTLK_HEADER_SIZE);
    out_slot->bytes = (size_t)tmp_hdr.payload_bytes;
    out_slot->slot_index = slot_idx;

    ring->read_in_flight = 1;
    ring->read_slot_index = slot_idx;
    ring->read_seq_pending = read_seq;
    return (int)VTLK_OK;
}

int vtlk_ring_release_read(vtlk_ring_t *ring, const vtlk_read_slot_t *slot)
{
    if (ring == NULL || slot == NULL) {
        return (int)VTLK_ERR_NULL_PTR;
    }
    if (ring->base == NULL || ring->read_in_flight == 0 ||
        slot->slot_index != ring->read_slot_index) {
        return (int)VTLK_ERR_INVALID_PARAM;
    }

    vtlk_atomic_store_release_u64(vtlk_ctrl_read_seq_ptr(ring), ring->read_seq_pending + 1u);
    ring->read_in_flight = 0;
    return (int)VTLK_OK;
}
