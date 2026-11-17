#ifndef VECTLINK_INTERNAL_ATOMIC_COMPAT_H
#define VECTLINK_INTERNAL_ATOMIC_COMPAT_H

#include <stdint.h>

typedef char vtlk_static_assert_u32_size[(sizeof(uint32_t) == 4u) ? 1 : -1];
typedef char vtlk_static_assert_u64_size[(sizeof(uint64_t) == 8u) ? 1 : -1];

#if defined(__GNUC__) || defined(__clang__)

#if defined(__GCC_ATOMIC_INT_LOCK_FREE) && (__GCC_ATOMIC_INT_LOCK_FREE < 2)
#error "32-bit lock-free atomics are required on this architecture"
#endif

#if defined(__GCC_ATOMIC_LLONG_LOCK_FREE) && (__GCC_ATOMIC_LLONG_LOCK_FREE < 2)
#error "64-bit lock-free atomics are required on this architecture"
#endif

static inline uint32_t vtlk_atomic_load_acquire_u32(const uint32_t *ptr)
{
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

static inline void vtlk_atomic_store_release_u32(uint32_t *ptr, uint32_t val)
{
    __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
}

static inline uint32_t vtlk_atomic_fetch_add_u32(uint32_t *ptr, uint32_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_ACQ_REL);
}

static inline uint32_t vtlk_atomic_fetch_sub_u32(uint32_t *ptr, uint32_t val)
{
    return __atomic_fetch_sub(ptr, val, __ATOMIC_ACQ_REL);
}

static inline int vtlk_atomic_cas_u32(uint32_t *ptr,
                                      uint32_t *expected,
                                      uint32_t desired)
{
    return __atomic_compare_exchange_n(ptr, expected, desired, 0,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

static inline uint64_t vtlk_atomic_load_acquire_u64(const uint64_t *ptr)
{
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

static inline void vtlk_atomic_store_release_u64(uint64_t *ptr, uint64_t val)
{
    __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
}

static inline uint64_t vtlk_atomic_fetch_add_u64(uint64_t *ptr, uint64_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_ACQ_REL);
}

static inline int vtlk_atomic_cas_u64(uint64_t *ptr,
                                      uint64_t *expected,
                                      uint64_t desired)
{
    return __atomic_compare_exchange_n(ptr, expected, desired, 0,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

#elif defined(_MSC_VER)

#include <intrin.h>

static __inline uint32_t vtlk_atomic_load_acquire_u32(const uint32_t *ptr)
{
    long val = _InterlockedOr((volatile long *)(uintptr_t)ptr, 0);
    _ReadWriteBarrier();
    return (uint32_t)val;
}

static __inline void vtlk_atomic_store_release_u32(uint32_t *ptr, uint32_t val)
{
    _ReadWriteBarrier();
    (void)_InterlockedExchange((volatile long *)ptr, (long)val);
}

static __inline uint32_t vtlk_atomic_fetch_add_u32(uint32_t *ptr, uint32_t val)
{
    return (uint32_t)_InterlockedExchangeAdd((volatile long *)ptr, (long)val);
}

static __inline uint32_t vtlk_atomic_fetch_sub_u32(uint32_t *ptr, uint32_t val)
{
    return (uint32_t)_InterlockedExchangeAdd((volatile long *)ptr, -(long)val);
}

static __inline int vtlk_atomic_cas_u32(uint32_t *ptr,
                                        uint32_t *expected,
                                        uint32_t desired)
{
    long exp = (long)*expected;
    long prev = _InterlockedCompareExchange((volatile long *)ptr,
                                            (long)desired, exp);
    if (prev == exp) {
        return 1;
    }
    *expected = (uint32_t)prev;
    return 0;
}

static __inline uint64_t vtlk_atomic_load_acquire_u64(const uint64_t *ptr)
{
    long long val = _InterlockedOr64((volatile long long *)(uintptr_t)ptr, 0);
    _ReadWriteBarrier();
    return (uint64_t)val;
}

static __inline void vtlk_atomic_store_release_u64(uint64_t *ptr, uint64_t val)
{
    _ReadWriteBarrier();
    (void)_InterlockedExchange64((volatile long long *)ptr, (long long)val);
}

static __inline uint64_t vtlk_atomic_fetch_add_u64(uint64_t *ptr, uint64_t val)
{
    return (uint64_t)_InterlockedExchangeAdd64((volatile long long *)ptr,
                                               (long long)val);
}

static __inline int vtlk_atomic_cas_u64(uint64_t *ptr,
                                        uint64_t *expected,
                                        uint64_t desired)
{
    long long exp = (long long)*expected;
    long long prev = _InterlockedCompareExchange64((volatile long long *)ptr,
                                                 (long long)desired, exp);
    if (prev == exp) {
        return 1;
    }
    *expected = (uint64_t)prev;
    return 0;
}

#else
#error "Unsupported compiler for Vect-Link atomic primitives"
#endif

#endif /* VECTLINK_INTERNAL_ATOMIC_COMPAT_H */
