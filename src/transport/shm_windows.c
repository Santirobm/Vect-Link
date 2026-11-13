#include "vectlink/transport.h"

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <stdio.h>
#include <string.h>

static vtlk_status_t vtlk_win_format_shm_name(const char *name,
                                              char *out,
                                              size_t out_cap)
{
    const char *clean;
    int written;

    if (name == NULL || name[0] == '\0') {
        return VTLK_ERR_INVALID_PARAM;
    }

    clean = (name[0] == '/') ? (name + 1) : name;
    if (clean[0] == '\0') {
        return VTLK_ERR_INVALID_PARAM;
    }

    if (strncmp(clean, "Local\\", 6u) == 0 || strncmp(clean, "Global\\", 7u) == 0) {
        written = snprintf(out, out_cap, "%s", clean);
    } else {
        written = snprintf(out, out_cap, "Local\\%s", clean);
    }

    if (written < 0 || (size_t)written >= out_cap) {
        return VTLK_ERR_INVALID_PARAM;
    }

    return VTLK_OK;
}

vtlk_status_t vtlk_shm_create(vtlk_shm_t *shm, const char *name, size_t size)
{
    HANDLE h_map;
    void *addr;
    DWORD size_hi;
    DWORD size_lo;
    vtlk_status_t st;

    if (shm == NULL || name == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (size == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    memset(shm, 0, sizeof(*shm));
    st = vtlk_win_format_shm_name(name, shm->name, sizeof(shm->name));
    if (st != VTLK_OK) {
        return st;
    }

    size_hi = (DWORD)(((uint64_t)size >> 32) & 0xFFFFFFFFu);
    size_lo = (DWORD)((uint64_t)size & 0xFFFFFFFFu);

    h_map = CreateFileMappingA(INVALID_HANDLE_VALUE,
                               NULL,
                               PAGE_READWRITE,
                               size_hi,
                               size_lo,
                               shm->name);
    if (h_map == NULL) {
        return VTLK_ERR_SHM;
    }

    addr = MapViewOfFile(h_map, FILE_MAP_ALL_ACCESS, 0u, 0u, size);
    if (addr == NULL) {
        CloseHandle(h_map);
        return VTLK_ERR_SHM;
    }

    shm->addr = addr;
    shm->size = size;
    shm->is_owner = 1;
    shm->handle = (void *)h_map;
    return VTLK_OK;
}

vtlk_status_t vtlk_shm_open(vtlk_shm_t *shm, const char *name, size_t size)
{
    HANDLE h_map;
    void *addr;
    vtlk_status_t st;

    if (shm == NULL || name == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (size == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    memset(shm, 0, sizeof(*shm));
    st = vtlk_win_format_shm_name(name, shm->name, sizeof(shm->name));
    if (st != VTLK_OK) {
        return st;
    }

    h_map = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, shm->name);
    if (h_map == NULL) {
        return VTLK_ERR_SHM;
    }

    addr = MapViewOfFile(h_map, FILE_MAP_ALL_ACCESS, 0u, 0u, size);
    if (addr == NULL) {
        CloseHandle(h_map);
        return VTLK_ERR_SHM;
    }

    shm->addr = addr;
    shm->size = size;
    shm->is_owner = 0;
    shm->handle = (void *)h_map;
    return VTLK_OK;
}

void vtlk_shm_close(vtlk_shm_t *shm)
{
    if (shm == NULL) {
        return;
    }
    if (shm->addr != NULL) {
        UnmapViewOfFile(shm->addr);
        shm->addr = NULL;
    }
    if (shm->handle != NULL) {
        CloseHandle((HANDLE)shm->handle);
        shm->handle = NULL;
    }
    shm->size = 0u;
}

vtlk_status_t vtlk_shm_unlink(const char *name)
{
    if (name == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    /* Windows kernel section objects are reference-counted and freed on close. */
    return VTLK_OK;
}

#else
typedef int vtlk_shm_windows_unused_t;
#endif
