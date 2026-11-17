#ifndef VECTLINK_SYNC_H
#define VECTLINK_SYNC_H

#include "vectlink/protocol.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VTLK_WAIT_INFINITE       0xFFFFFFFFu
#define VTLK_KV_DESC_MAX_BLOCKS  64u
#define VTLK_DESC_RING_CTRL_SIZE 192u
#define VTLK_DESC_RING_VERSION   1u

/**
 * @brief OS-native wait on a 32-bit address while *addr == expected.
 *
 * Uses WaitOnAddress on Windows, futex(FUTEX_WAIT) on Linux, and
 * sched_yield/nanosleep on other POSIX systems.
 *
 * @param addr       Pointer to the 32-bit word (typically in SHM).
 * @param expected   Value that causes the thread to sleep if matched.
 * @param timeout_ms Timeout in milliseconds (0 = immediate check,
 *                   VTLK_WAIT_INFINITE = no timeout).
 * @return VTLK_OK if woken or *addr != expected, VTLK_ERR_TIMEOUT on timeout,
 *         or VTLK_ERR_NULL_PTR if addr is NULL.
 */
VTLK_API vtlk_status_t vtlk_notify_wait_u32(volatile uint32_t *addr,
                                            uint32_t expected,
                                            uint32_t timeout_ms);

/**
 * @brief OS-native wake on a 32-bit address.
 *
 * @param addr     Pointer to the 32-bit word.
 * @param wake_all Non-zero to wake all waiters, 0 to wake a single waiter.
 * @return VTLK_OK on success, or VTLK_ERR_NULL_PTR if addr is NULL.
 */
VTLK_API vtlk_status_t vtlk_notify_wake_u32(volatile uint32_t *addr,
                                            int wake_all);

/**
 * @brief Lightweight KV-cache descriptor passed across SHM descriptor rings.
 */
typedef struct vtlk_kv_descriptor {
    uint64_t snapshot_id;
    uint64_t branch_id;
    uint64_t sequence;
    uint32_t pos_offset;
    uint32_t block_count;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t block_ids[VTLK_KV_DESC_MAX_BLOCKS];
} vtlk_kv_descriptor_t;

/**
 * @brief Handle to a lock-free SPSC/MPMC descriptor ring over SHM.
 */
typedef struct vtlk_desc_ring {
    uint8_t *base;
    size_t total_bytes;
    uint32_t capacity;
    uint32_t mask;
} vtlk_desc_ring_t;

/**
 * @brief Computes the total byte size required for a descriptor ring.
 *
 * @param capacity Number of descriptor slots (must be a power of 2, >= 1).
 * @return Required total bytes, or 0 on invalid capacity / overflow.
 */
VTLK_API size_t vtlk_desc_ring_required_bytes(uint32_t capacity);

/**
 * @brief Initializes a lock-free MPMC/SPSC descriptor ring in SHM/RAM.
 *
 * @param ring      Ring handle to initialize.
 * @param mem       Caller-provided or SHM memory (must be >= 8-byte aligned).
 * @param mem_bytes Size of @p mem in bytes.
 * @param capacity  Slot count (must be a power of 2, >= 1).
 * @return VTLK_OK on success, or a negative vtlk_status_t error code.
 */
VTLK_API vtlk_status_t vtlk_desc_ring_init(vtlk_desc_ring_t *ring,
                                           void *mem,
                                           size_t mem_bytes,
                                           uint32_t capacity);

/**
 * @brief Attaches to an existing descriptor ring in SHM/RAM.
 *
 * @param ring      Ring handle to populate.
 * @param mem       Existing ring memory.
 * @param mem_bytes Size of @p mem in bytes.
 * @return VTLK_OK on success, or a negative vtlk_status_t error code.
 */
VTLK_API vtlk_status_t vtlk_desc_ring_attach(vtlk_desc_ring_t *ring,
                                             void *mem,
                                             size_t mem_bytes);

/**
 * @brief Enqueues a descriptor into the ring (MPMC-safe, lock-free).
 *
 * Blocks using vtlk_notify_wait_u32 when full if timeout_ms > 0.
 *
 * @param ring       Initialized or attached descriptor ring.
 * @param desc       Descriptor to enqueue (block_count <= 64).
 * @param timeout_ms 0 for non-blocking, >0 for bounded wait, VTLK_WAIT_INFINITE.
 * @return VTLK_OK on success, VTLK_ERR_FULL / VTLK_ERR_TIMEOUT, or error code.
 */
VTLK_API vtlk_status_t vtlk_desc_ring_push(vtlk_desc_ring_t *ring,
                                           const vtlk_kv_descriptor_t *desc,
                                           uint32_t timeout_ms);

/**
 * @brief Dequeues a descriptor from the ring (MPMC-safe, lock-free).
 *
 * Blocks using vtlk_notify_wait_u32 when empty if timeout_ms > 0.
 *
 * @param ring       Initialized or attached descriptor ring.
 * @param desc_out   Destination descriptor struct.
 * @param timeout_ms 0 for non-blocking, >0 for bounded wait, VTLK_WAIT_INFINITE.
 * @return VTLK_OK on success, VTLK_ERR_EMPTY / VTLK_ERR_TIMEOUT, or error code.
 */
VTLK_API vtlk_status_t vtlk_desc_ring_pop(vtlk_desc_ring_t *ring,
                                          vtlk_kv_descriptor_t *desc_out,
                                          uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* VECTLINK_SYNC_H */
