#include "vectlink_sync.h"
#include "protocol/endian.h"

#include <string.h>

#define VTLK_DR_MAGIC_LEN 8u
static const uint8_t VTLK_DR_MAGIC[VTLK_DR_MAGIC_LEN] = {
    'V', 'T', 'L', 'K', '_', 'D', 'R', 'G'
};

#define VTLK_DR_OFF_MAGIC           0u
#define VTLK_DR_OFF_VERSION         8u
#define VTLK_DR_OFF_CAPACITY        12u
#define VTLK_DR_OFF_TOTAL_BYTES     16u
#define VTLK_DR_OFF_ENQUEUE_POS     64u
#define VTLK_DR_OFF_PUSH_NOTIFY_SEQ 72u
#define VTLK_DR_OFF_DEQUEUE_POS     128u
#define VTLK_DR_OFF_POP_NOTIFY_SEQ  136u

typedef struct vtlk_desc_ring_slot {
    uint64_t seq;
    vtlk_kv_descriptor_t desc;
} vtlk_desc_ring_slot_t;

#if defined(__GNUC__) || defined(__clang__)

static inline uint64_t vtlk_dr_load_acq_u64(const volatile uint64_t *p)
{
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

static inline uint64_t vtlk_dr_load_rlx_u64(const volatile uint64_t *p)
{
    return __atomic_load_n(p, __ATOMIC_RELAXED);
}

static inline void vtlk_dr_store_rel_u64(volatile uint64_t *p, uint64_t v)
{
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

static inline int vtlk_dr_cas_u64(volatile uint64_t *p,
                                  uint64_t *expected,
                                  uint64_t desired)
{
    return __atomic_compare_exchange_n(p,
                                       expected,
                                       desired,
                                       0,
                                       __ATOMIC_ACQ_REL,
                                       __ATOMIC_RELAXED);
}

static inline uint32_t vtlk_dr_load_acq_u32(const volatile uint32_t *p)
{
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

static inline void vtlk_dr_fetch_add_u32(volatile uint32_t *p, uint32_t v)
{
    (void)__atomic_fetch_add(p, v, __ATOMIC_RELEASE);
}

#elif defined(_MSC_VER)

#include <intrin.h>

static __inline uint64_t vtlk_dr_load_acq_u64(const volatile uint64_t *p)
{
    long long v = _InterlockedOr64((volatile long long *)(uintptr_t)p, 0);
    _ReadWriteBarrier();
    return (uint64_t)v;
}

static __inline uint64_t vtlk_dr_load_rlx_u64(const volatile uint64_t *p)
{
    return (uint64_t)_InterlockedOr64((volatile long long *)(uintptr_t)p, 0);
}

static __inline void vtlk_dr_store_rel_u64(volatile uint64_t *p, uint64_t v)
{
    _ReadWriteBarrier();
    (void)_InterlockedExchange64((volatile long long *)p, (long long)v);
}

static __inline int vtlk_dr_cas_u64(volatile uint64_t *p,
                                    uint64_t *expected,
                                    uint64_t desired)
{
    long long prev = _InterlockedCompareExchange64((volatile long long *)p,
                                                   (long long)desired,
                                                   (long long)*expected);
    if ((uint64_t)prev == *expected) {
        return 1;
    }
    *expected = (uint64_t)prev;
    return 0;
}

static __inline uint32_t vtlk_dr_load_acq_u32(const volatile uint32_t *p)
{
    long v = _InterlockedOr((volatile long *)(uintptr_t)p, 0);
    _ReadWriteBarrier();
    return (uint32_t)v;
}

static __inline void vtlk_dr_fetch_add_u32(volatile uint32_t *p, uint32_t v)
{
    _ReadWriteBarrier();
    (void)_InterlockedExchangeAdd((volatile long *)p, (long)v);
}

#else
#error "Unsupported compiler for desc_ring atomics"
#endif

static int vtlk_is_pow2_u32(uint32_t v)
{
    return (v != 0u) && ((v & (v - 1u)) == 0u);
}

size_t vtlk_desc_ring_required_bytes(uint32_t capacity)
{
    size_t slot_sz;
    size_t total;

    if (!vtlk_is_pow2_u32(capacity)) {
        return 0u;
    }

    slot_sz = sizeof(vtlk_desc_ring_slot_t);
    if ((size_t)capacity > ((SIZE_MAX - VTLK_DESC_RING_CTRL_SIZE) / slot_sz)) {
        return 0u;
    }

    total = VTLK_DESC_RING_CTRL_SIZE + ((size_t)capacity * slot_sz);
    return total;
}

vtlk_status_t vtlk_desc_ring_init(vtlk_desc_ring_t *ring,
                                  void *mem,
                                  size_t mem_bytes,
                                  uint32_t capacity)
{
    size_t required;
    uint8_t *base;
    vtlk_desc_ring_slot_t *slots;
    uint32_t i;

    if (ring == NULL || mem == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (((uintptr_t)mem & 7u) != 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    required = vtlk_desc_ring_required_bytes(capacity);
    if (required == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (mem_bytes < required) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    base = (uint8_t *)mem;
    memset(base, 0, VTLK_DESC_RING_CTRL_SIZE);

    memcpy(base + VTLK_DR_OFF_MAGIC, VTLK_DR_MAGIC, VTLK_DR_MAGIC_LEN);
    vtlk_write_u32_le(base + VTLK_DR_OFF_VERSION, VTLK_DESC_RING_VERSION);
    vtlk_write_u32_le(base + VTLK_DR_OFF_CAPACITY, capacity);
    vtlk_write_u64_le(base + VTLK_DR_OFF_TOTAL_BYTES, (uint64_t)required);

    slots = (vtlk_desc_ring_slot_t *)(void *)(base + VTLK_DESC_RING_CTRL_SIZE);
    for (i = 0u; i < capacity; ++i) {
        memset(&slots[i].desc, 0, sizeof(slots[i].desc));
        vtlk_dr_store_rel_u64(&slots[i].seq, (uint64_t)i);
    }

    ring->base = base;
    ring->total_bytes = required;
    ring->capacity = capacity;
    ring->mask = capacity - 1u;
    return VTLK_OK;
}

vtlk_status_t vtlk_desc_ring_attach(vtlk_desc_ring_t *ring,
                                    void *mem,
                                    size_t mem_bytes)
{
    const uint8_t *base;
    uint32_t version;
    uint32_t capacity;
    uint64_t stored_total;
    size_t required;

    if (ring == NULL || mem == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (((uintptr_t)mem & 7u) != 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (mem_bytes < VTLK_DESC_RING_CTRL_SIZE) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    base = (const uint8_t *)mem;
    if (memcmp(base + VTLK_DR_OFF_MAGIC, VTLK_DR_MAGIC, VTLK_DR_MAGIC_LEN) != 0) {
        return VTLK_ERR_INVALID_MAGIC;
    }

    version = vtlk_read_u32_le(base + VTLK_DR_OFF_VERSION);
    if (version != VTLK_DESC_RING_VERSION) {
        return VTLK_ERR_UNSUPPORTED_VERSION;
    }

    capacity = vtlk_read_u32_le(base + VTLK_DR_OFF_CAPACITY);
    required = vtlk_desc_ring_required_bytes(capacity);
    if (required == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    stored_total = vtlk_read_u64_le(base + VTLK_DR_OFF_TOTAL_BYTES);
    if (stored_total != (uint64_t)required || mem_bytes < required) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    ring->base = (uint8_t *)mem;
    ring->total_bytes = required;
    ring->capacity = capacity;
    ring->mask = capacity - 1u;
    return VTLK_OK;
}

vtlk_status_t vtlk_desc_ring_push(vtlk_desc_ring_t *ring,
                                  const vtlk_kv_descriptor_t *desc,
                                  uint32_t timeout_ms)
{
    volatile uint64_t *enqueue_pos;
    volatile uint32_t *push_notify;
    volatile uint32_t *pop_notify;
    vtlk_desc_ring_slot_t *slots;
    uint64_t pos;

    if (ring == NULL || desc == NULL || ring->base == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (desc->block_count > VTLK_KV_DESC_MAX_BLOCKS) {
        return VTLK_ERR_INVALID_PARAM;
    }

    enqueue_pos = (volatile uint64_t *)(void *)(ring->base + VTLK_DR_OFF_ENQUEUE_POS);
    push_notify = (volatile uint32_t *)(void *)(ring->base + VTLK_DR_OFF_PUSH_NOTIFY_SEQ);
    pop_notify = (volatile uint32_t *)(void *)(ring->base + VTLK_DR_OFF_POP_NOTIFY_SEQ);
    slots = (vtlk_desc_ring_slot_t *)(void *)(ring->base + VTLK_DESC_RING_CTRL_SIZE);

    pos = vtlk_dr_load_rlx_u64(enqueue_pos);
    for (;;) {
        vtlk_desc_ring_slot_t *slot = &slots[pos & ring->mask];
        uint64_t seq = vtlk_dr_load_acq_u64(&slot->seq);
        int64_t diff = (int64_t)seq - (int64_t)pos;

        if (diff == 0) {
            if (vtlk_dr_cas_u64(enqueue_pos, &pos, pos + 1u)) {
                slot->desc = *desc;
                vtlk_dr_store_rel_u64(&slot->seq, pos + 1u);
                vtlk_dr_fetch_add_u32(pop_notify, 1u);
                (void)vtlk_notify_wake_u32(pop_notify, 0);
                return VTLK_OK;
            }
        } else if (diff < 0) {
            uint32_t notify_val;
            vtlk_status_t wst;

            if (timeout_ms == 0u) {
                return VTLK_ERR_FULL;
            }
            notify_val = vtlk_dr_load_acq_u32(push_notify);
            seq = vtlk_dr_load_acq_u64(&slot->seq);
            if (((int64_t)seq - (int64_t)pos) < 0) {
                wst = vtlk_notify_wait_u32(push_notify, notify_val, timeout_ms);
                if (wst == VTLK_ERR_TIMEOUT) {
                    return VTLK_ERR_TIMEOUT;
                }
            }
            pos = vtlk_dr_load_rlx_u64(enqueue_pos);
        } else {
            pos = vtlk_dr_load_rlx_u64(enqueue_pos);
        }
    }
}

vtlk_status_t vtlk_desc_ring_pop(vtlk_desc_ring_t *ring,
                                 vtlk_kv_descriptor_t *desc_out,
                                 uint32_t timeout_ms)
{
    volatile uint64_t *dequeue_pos;
    volatile uint32_t *push_notify;
    volatile uint32_t *pop_notify;
    vtlk_desc_ring_slot_t *slots;
    uint64_t pos;

    if (ring == NULL || desc_out == NULL || ring->base == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    dequeue_pos = (volatile uint64_t *)(void *)(ring->base + VTLK_DR_OFF_DEQUEUE_POS);
    push_notify = (volatile uint32_t *)(void *)(ring->base + VTLK_DR_OFF_PUSH_NOTIFY_SEQ);
    pop_notify = (volatile uint32_t *)(void *)(ring->base + VTLK_DR_OFF_POP_NOTIFY_SEQ);
    slots = (vtlk_desc_ring_slot_t *)(void *)(ring->base + VTLK_DESC_RING_CTRL_SIZE);

    pos = vtlk_dr_load_rlx_u64(dequeue_pos);
    for (;;) {
        vtlk_desc_ring_slot_t *slot = &slots[pos & ring->mask];
        uint64_t seq = vtlk_dr_load_acq_u64(&slot->seq);
        int64_t diff = (int64_t)seq - (int64_t)(pos + 1u);

        if (diff == 0) {
            if (vtlk_dr_cas_u64(dequeue_pos, &pos, pos + 1u)) {
                *desc_out = slot->desc;
                vtlk_dr_store_rel_u64(&slot->seq, pos + (uint64_t)ring->capacity);
                vtlk_dr_fetch_add_u32(push_notify, 1u);
                (void)vtlk_notify_wake_u32(push_notify, 0);
                return VTLK_OK;
            }
        } else if (diff < 0) {
            uint32_t notify_val;
            vtlk_status_t wst;

            if (timeout_ms == 0u) {
                return VTLK_ERR_EMPTY;
            }
            notify_val = vtlk_dr_load_acq_u32(pop_notify);
            seq = vtlk_dr_load_acq_u64(&slot->seq);
            if (((int64_t)seq - (int64_t)(pos + 1u)) < 0) {
                wst = vtlk_notify_wait_u32(pop_notify, notify_val, timeout_ms);
                if (wst == VTLK_ERR_TIMEOUT) {
                    return VTLK_ERR_TIMEOUT;
                }
            }
            pos = vtlk_dr_load_rlx_u64(dequeue_pos);
        } else {
            pos = vtlk_dr_load_rlx_u64(dequeue_pos);
        }
    }
}
