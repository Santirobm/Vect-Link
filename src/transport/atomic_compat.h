#ifndef VECTLINK_INTERNAL_ATOMIC_COMPAT_H
#define VECTLINK_INTERNAL_ATOMIC_COMPAT_H

#include <stdint.h>

typedef char vtlk_static_assert_u64_size[(sizeof(uint64_t) == 8u) ? 1 : -1];

#if defined(__GNUC__) || defined(__clang__)

#if defined(__GCC_ATOMIC_LLONG_LOCK_FREE) && (__GCC_ATOMIC_LLONG_LOCK_FREE < 2)
#error "64-bit lock-free atomics are required on this architecture"
#endif

static inline uint64_t vtlk_atomic_load_acquire_u64(const uint64_t *ptr)
{
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

static inline void vtlk_atomic_store_release_u64(uint64_t *ptr, uint64_t val)
{
    __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
}

#elif defined(_MSC_VER)

#include <intrin.h>

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

#else
#error "Unsupported compiler for Vect-Link atomic primitives"
#endif

#endif /* VECTLINK_INTERNAL_ATOMIC_COMPAT_H */
