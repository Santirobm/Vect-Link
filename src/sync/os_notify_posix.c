#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "vectlink_sync.h"

#ifndef _WIN32

#include <errno.h>
#include <limits.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/futex.h>
#include <sys/syscall.h>
#else
static uint64_t vtlk_posix_monotonic_ms(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0u;
    }
    return ((uint64_t)ts.tv_sec * 1000u) + ((uint64_t)ts.tv_nsec / 1000000u);
}
#endif

vtlk_status_t vtlk_notify_wait_u32(volatile uint32_t *addr,
                                   uint32_t expected,
                                   uint32_t timeout_ms)
{
    if (addr == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    if (__atomic_load_n(addr, __ATOMIC_ACQUIRE) != expected) {
        return VTLK_OK;
    }
    if (timeout_ms == 0u) {
        return VTLK_ERR_TIMEOUT;
    }

#ifdef __linux__
    {
        struct timespec ts;
        const struct timespec *pts = NULL;
        long rc;

        if (timeout_ms != VTLK_WAIT_INFINITE) {
            ts.tv_sec = (time_t)(timeout_ms / 1000u);
            ts.tv_nsec = (long)((timeout_ms % 1000u) * 1000000u);
            pts = &ts;
        }

        for (;;) {
            if (__atomic_load_n(addr, __ATOMIC_ACQUIRE) != expected) {
                return VTLK_OK;
            }
            rc = syscall(SYS_futex,
                         (void *)(uintptr_t)addr,
                         FUTEX_WAIT,
                         (int)expected,
                         pts,
                         NULL,
                         0);
            if (rc == 0) {
                return VTLK_OK;
            }
            if (errno == EAGAIN) {
                return VTLK_OK;
            }
            if (errno == ETIMEDOUT) {
                return VTLK_ERR_TIMEOUT;
            }
            if (errno != EINTR) {
                return VTLK_OK;
            }
        }
    }
#else
    {
        uint64_t start_ms = vtlk_posix_monotonic_ms();
        uint32_t spin = 0u;
        struct timespec req;

        req.tv_sec = 0;
        req.tv_nsec = 250000L; /* 250 us */

        for (;;) {
            if (__atomic_load_n(addr, __ATOMIC_ACQUIRE) != expected) {
                return VTLK_OK;
            }
            if (timeout_ms != VTLK_WAIT_INFINITE) {
                uint64_t elapsed = vtlk_posix_monotonic_ms() - start_ms;
                if (elapsed >= (uint64_t)timeout_ms) {
                    return VTLK_ERR_TIMEOUT;
                }
            }
            if (spin < 16u) {
                sched_yield();
                spin++;
            } else {
                nanosleep(&req, NULL);
            }
        }
    }
#endif
}

vtlk_status_t vtlk_notify_wake_u32(volatile uint32_t *addr, int wake_all)
{
    if (addr == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

#ifdef __linux__
    (void)syscall(SYS_futex,
                  (void *)(uintptr_t)addr,
                  FUTEX_WAKE,
                  (wake_all != 0) ? INT_MAX : 1,
                  NULL,
                  NULL,
                  0);
#else
    (void)wake_all;
#endif

    return VTLK_OK;
}

#else
typedef int vtlk_os_notify_posix_unused_t;
#endif
