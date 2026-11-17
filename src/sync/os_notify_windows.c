#include "vectlink_sync.h"

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <string.h>

typedef BOOL(WINAPI *vtlk_wait_on_address_fn)(volatile VOID *Address,
                                              PVOID CompareAddress,
                                              SIZE_T AddressSize,
                                              DWORD dwMilliseconds);
typedef VOID(WINAPI *vtlk_wake_by_address_single_fn)(PVOID Address);
typedef VOID(WINAPI *vtlk_wake_by_address_all_fn)(PVOID Address);

typedef struct vtlk_win_sync_api {
    LONG initialized;
    vtlk_wait_on_address_fn wait_on_address;
    vtlk_wake_by_address_single_fn wake_single;
    vtlk_wake_by_address_all_fn wake_all;
} vtlk_win_sync_api_t;

static vtlk_win_sync_api_t g_win_sync_api = {0, NULL, NULL, NULL};

static void vtlk_win_sync_init_once(void)
{
    HMODULE h_mod;
    FARPROC p_wait;
    FARPROC p_wake_one;
    FARPROC p_wake_all;

    if (InterlockedCompareExchange(&g_win_sync_api.initialized, 0, 0) == 2) {
        return;
    }

    if (InterlockedCompareExchange(&g_win_sync_api.initialized, 1, 0) == 0) {
        h_mod = GetModuleHandleA("api-ms-win-core-synch-l1-2-0.dll");
        if (h_mod == NULL) {
            h_mod = GetModuleHandleA("kernelbase.dll");
        }
        if (h_mod != NULL) {
            p_wait = GetProcAddress(h_mod, "WaitOnAddress");
            p_wake_one = GetProcAddress(h_mod, "WakeByAddressSingle");
            p_wake_all = GetProcAddress(h_mod, "WakeByAddressAll");
            if (p_wait != NULL && p_wake_one != NULL && p_wake_all != NULL) {
                memcpy(&g_win_sync_api.wait_on_address, &p_wait, sizeof(p_wait));
                memcpy(&g_win_sync_api.wake_single, &p_wake_one, sizeof(p_wake_one));
                memcpy(&g_win_sync_api.wake_all, &p_wake_all, sizeof(p_wake_all));
            }
        }
        InterlockedExchange(&g_win_sync_api.initialized, 2);
    } else {
        while (InterlockedCompareExchange(&g_win_sync_api.initialized, 0, 0) != 2) {
            SwitchToThread();
        }
    }
}

vtlk_status_t vtlk_notify_wait_u32(volatile uint32_t *addr,
                                   uint32_t expected,
                                   uint32_t timeout_ms)
{
    uint32_t exp_copy;
    DWORD dw_timeout;
    ULONGLONG start_ms;

    if (addr == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    if (*addr != expected) {
        return VTLK_OK;
    }
    if (timeout_ms == 0u) {
        return VTLK_ERR_TIMEOUT;
    }

    vtlk_win_sync_init_once();
    dw_timeout = (timeout_ms == VTLK_WAIT_INFINITE) ? INFINITE : (DWORD)timeout_ms;

    if (g_win_sync_api.wait_on_address != NULL) {
        exp_copy = expected;
        if (g_win_sync_api.wait_on_address((volatile VOID *)addr,
                                           &exp_copy,
                                           sizeof(uint32_t),
                                           dw_timeout)) {
            return VTLK_OK;
        }
        if (GetLastError() == ERROR_TIMEOUT) {
            return VTLK_ERR_TIMEOUT;
        }
        return VTLK_OK;
    }

    start_ms = GetTickCount64();
    for (;;) {
        if (*addr != expected) {
            return VTLK_OK;
        }
        if (timeout_ms != VTLK_WAIT_INFINITE) {
            ULONGLONG elapsed = GetTickCount64() - start_ms;
            if (elapsed >= (ULONGLONG)timeout_ms) {
                return VTLK_ERR_TIMEOUT;
            }
        }
        if (!SwitchToThread()) {
            Sleep(1u);
        }
    }
}

vtlk_status_t vtlk_notify_wake_u32(volatile uint32_t *addr, int wake_all)
{
    if (addr == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    vtlk_win_sync_init_once();

    if (wake_all != 0) {
        if (g_win_sync_api.wake_all != NULL) {
            g_win_sync_api.wake_all((PVOID)(uintptr_t)addr);
        }
    } else {
        if (g_win_sync_api.wake_single != NULL) {
            g_win_sync_api.wake_single((PVOID)(uintptr_t)addr);
        }
    }

    return VTLK_OK;
}

#else
typedef int vtlk_os_notify_windows_unused_t;
#endif
