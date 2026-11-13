#ifndef VECTLINK_TRANSPORT_H
#define VECTLINK_TRANSPORT_H

#include "vectlink/protocol.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VTLK_RING_CTRL_SIZE  192u
#define VTLK_RING_CACHE_LINE 64u
#define VTLK_SHM_VERSION     1u

/**
 * @brief Handle to an SPSC lock-free ring buffer mapped over caller or SHM memory.
 */
typedef struct vtlk_ring {
    uint8_t *base;
    size_t total_bytes;
    uint32_t capacity;
    uint32_t slot_size;
    uint32_t max_payload_bytes;
    uint64_t cached_read_seq;
    uint64_t cached_write_seq;
} vtlk_ring_t;

/**
 * @brief Handle to an OS-backed shared memory region.
 */
typedef struct vtlk_shm {
    void *addr;
    size_t size;
    char name[256];
    int is_owner;
#ifdef _WIN32
    void *handle;
#else
    int fd;
#endif
} vtlk_shm_t;

/**
 * @brief Computes the total byte size required for a ring buffer.
 *
 * @param capacity          Number of slots (must be a power of 2, >= 1).
 * @param max_payload_bytes Maximum payload bytes per slot.
 * @param out_slot_size     Optional output for the 64-byte-aligned slot size.
 * @return Required total bytes (192 + capacity * slot_size), or 0 on invalid
 *         parameters / overflow.
 */
size_t vtlk_ring_required_bytes(uint32_t capacity,
                                uint32_t max_payload_bytes,
                                uint32_t *out_slot_size);

/**
 * @brief Initializes a new SPSC ring buffer in the provided memory region.
 *
 * Writes the 192-byte cache-line-isolated control block and resets
 * write_seq and read_seq to 0.
 *
 * @param ring              Ring handle to initialize.
 * @param mem               Caller-provided or SHM memory (must be >= 8-byte aligned).
 * @param mem_bytes         Size of @p mem in bytes.
 * @param capacity          Slot count (must be a power of 2, >= 1).
 * @param max_payload_bytes Maximum payload size per packet in bytes.
 * @return VTLK_OK on success, or a negative vtlk_status_t error code.
 */
vtlk_status_t vtlk_ring_init(vtlk_ring_t *ring,
                             void *mem,
                             size_t mem_bytes,
                             uint32_t capacity,
                             uint32_t max_payload_bytes);

/**
 * @brief Attaches to an already-initialized SPSC ring buffer in memory.
 *
 * Validates the control block magic ("VTLK_SHM"), version, capacity,
 * slot_size alignment, and total buffer size.
 *
 * @param ring      Ring handle to populate.
 * @param mem       Existing ring buffer memory.
 * @param mem_bytes Size of @p mem in bytes.
 * @return VTLK_OK on success, or a negative vtlk_status_t error code.
 */
vtlk_status_t vtlk_ring_attach(vtlk_ring_t *ring,
                               void *mem,
                               size_t mem_bytes);

/**
 * @brief Pushes a packet (64-byte header + payload) into the SPSC ring buffer.
 *
 * Producer-only. Computes payload_crc32c and header_crc32c, writes the
 * encoded header and payload into the next slot, and publishes write_seq
 * with release semantics.
 *
 * @param ring    Initialized or attached ring buffer handle.
 * @param hdr     Packet header (payload_bytes must be <= max_payload_bytes).
 * @param payload Pointer to payload bytes (may be NULL only if payload_bytes == 0).
 * @return VTLK_OK on success, VTLK_ERR_FULL if the ring is full, or error code.
 */
vtlk_status_t vtlk_ring_push(vtlk_ring_t *ring,
                             const vtlk_header_t *hdr,
                             const void *payload);

/**
 * @brief Pops and validates the next packet from the SPSC ring buffer.
 *
 * Consumer-only. Reads write_seq with acquire semantics, decodes and validates
 * the 64-byte header, verifies payload_crc32c against the slot data, copies
 * the payload into @p payload_out, and advances read_seq with release semantics.
 *
 * @param ring            Initialized or attached ring buffer handle.
 * @param hdr_out         Destination for decoded header.
 * @param payload_out     Destination buffer for payload (may be NULL if cap == 0).
 * @param payload_out_cap Capacity of @p payload_out in bytes.
 * @return VTLK_OK on success, VTLK_ERR_EMPTY if empty, VTLK_ERR_CRC_MISMATCH
 *         on corruption, or another negative vtlk_status_t error code.
 */
vtlk_status_t vtlk_ring_pop(vtlk_ring_t *ring,
                            vtlk_header_t *hdr_out,
                            void *payload_out,
                            size_t payload_out_cap);

/**
 * @brief Creates and maps a new OS shared memory region.
 *
 * @param shm  Shared memory handle to populate.
 * @param name Cross-platform region name (e.g. "/vectlink_ipc").
 * @param size Size in bytes (> 0).
 * @return VTLK_OK on success, or VTLK_ERR_SHM / error code.
 */
vtlk_status_t vtlk_shm_create(vtlk_shm_t *shm, const char *name, size_t size);

/**
 * @brief Opens and maps an existing OS shared memory region.
 *
 * @param shm  Shared memory handle to populate.
 * @param name Cross-platform region name.
 * @param size Expected size in bytes (> 0).
 * @return VTLK_OK on success, or VTLK_ERR_SHM / error code.
 */
vtlk_status_t vtlk_shm_open(vtlk_shm_t *shm, const char *name, size_t size);

/**
 * @brief Unmaps and closes an OS shared memory handle.
 *
 * @param shm Shared memory handle (no-op if NULL).
 */
void vtlk_shm_close(vtlk_shm_t *shm);

/**
 * @brief Removes a named shared memory object (POSIX shm_unlink; no-op on Windows).
 *
 * @param name Cross-platform region name.
 * @return VTLK_OK on success, or VTLK_ERR_SHM / error code.
 */
vtlk_status_t vtlk_shm_unlink(const char *name);

#ifdef __cplusplus
}
#endif

#endif /* VECTLINK_TRANSPORT_H */
