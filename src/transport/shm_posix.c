#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#ifndef _WIN32

#include "vectlink/transport.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static vtlk_status_t vtlk_posix_format_shm_name(const char *name,
                                                char *out,
                                                size_t out_cap)
{
    int written;

    if (name == NULL || name[0] == '\0') {
        return VTLK_ERR_INVALID_PARAM;
    }

    if (name[0] == '/') {
        if (name[1] == '\0') {
            return VTLK_ERR_INVALID_PARAM;
        }
        written = snprintf(out, out_cap, "%s", name);
    } else {
        written = snprintf(out, out_cap, "/%s", name);
    }

    if (written < 0 || (size_t)written >= out_cap) {
        return VTLK_ERR_INVALID_PARAM;
    }

    return VTLK_OK;
}

vtlk_status_t vtlk_shm_create(vtlk_shm_t *shm, const char *name, size_t size)
{
    int fd;
    void *addr;
    vtlk_status_t st;

    if (shm == NULL || name == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (size == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    memset(shm, 0, sizeof(*shm));
    shm->fd = -1;

    st = vtlk_posix_format_shm_name(name, shm->name, sizeof(shm->name));
    if (st != VTLK_OK) {
        return st;
    }

    fd = shm_open(shm->name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        return VTLK_ERR_SHM;
    }

    if (ftruncate(fd, (off_t)size) != 0) {
        close(fd);
        shm_unlink(shm->name);
        return VTLK_ERR_SHM;
    }

    addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        shm_unlink(shm->name);
        return VTLK_ERR_SHM;
    }

    shm->addr = addr;
    shm->size = size;
    shm->is_owner = 1;
    shm->fd = fd;
    return VTLK_OK;
}

vtlk_status_t vtlk_shm_open(vtlk_shm_t *shm, const char *name, size_t size)
{
    int fd;
    void *addr;
    vtlk_status_t st;

    if (shm == NULL || name == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (size == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    memset(shm, 0, sizeof(*shm));
    shm->fd = -1;

    st = vtlk_posix_format_shm_name(name, shm->name, sizeof(shm->name));
    if (st != VTLK_OK) {
        return st;
    }

    fd = shm_open(shm->name, O_RDWR, 0600);
    if (fd < 0) {
        return VTLK_ERR_SHM;
    }

    {
        struct stat st_buf;
        if (fstat(fd, &st_buf) != 0 ||
            st_buf.st_uid != getuid() ||
            st_buf.st_size < (off_t)size) {
            close(fd);
            return VTLK_ERR_SHM;
        }
    }

    addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        return VTLK_ERR_SHM;
    }

    shm->addr = addr;
    shm->size = size;
    shm->is_owner = 0;
    shm->fd = fd;
    return VTLK_OK;
}

void vtlk_shm_close(vtlk_shm_t *shm)
{
    if (shm == NULL) {
        return;
    }
    if (shm->addr != NULL) {
        munmap(shm->addr, shm->size);
        shm->addr = NULL;
    }
    if (shm->fd >= 0) {
        close(shm->fd);
        shm->fd = -1;
    }
    shm->size = 0u;
}

vtlk_status_t vtlk_shm_unlink(const char *name)
{
    char shm_name[256];
    vtlk_status_t st;

    if (name == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    st = vtlk_posix_format_shm_name(name, shm_name, sizeof(shm_name));
    if (st != VTLK_OK) {
        return st;
    }

    if (shm_unlink(shm_name) != 0) {
        return VTLK_ERR_SHM;
    }

    return VTLK_OK;
}

#else
typedef int vtlk_shm_posix_unused_t;
#endif
