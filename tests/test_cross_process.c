#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "vectlink/transport.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sched.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define XPROC_NUM_PACKETS 10000u
#define XPROC_CAPACITY    256u
#define XPROC_PAYLOAD_SZ  128u

static void xproc_cpu_yield(void)
{
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}

static void fill_synthetic_payload(uint8_t *buf, size_t len, uint32_t seq)
{
    size_t i;
    for (i = 0u; i < len; ++i) {
        buf[i] = (uint8_t)((seq * 31u + (uint32_t)i * 7u) & 0xFFu);
    }
}

static int run_consumer(const char *shm_name)
{
    vtlk_shm_t shm;
    vtlk_ring_t ring;
    size_t req;
    uint32_t i;
    vtlk_header_t hdr;
    uint8_t actual[XPROC_PAYLOAD_SZ];
    uint8_t expected[XPROC_PAYLOAD_SZ];
    vtlk_status_t st;

    req = vtlk_ring_required_bytes(XPROC_CAPACITY, XPROC_PAYLOAD_SZ, NULL);
    assert(req > 0u);

    st = vtlk_shm_open(&shm, shm_name, req);
    if (st != VTLK_OK) {
        fprintf(stderr, "Consumer: vtlk_shm_open failed (%d)\n", (int)st);
        return 1;
    }

    st = vtlk_ring_attach(&ring, shm.addr, shm.size);
    if (st != VTLK_OK) {
        fprintf(stderr, "Consumer: vtlk_ring_attach failed (%d)\n", (int)st);
        vtlk_shm_close(&shm);
        return 1;
    }

    for (i = 0u; i < XPROC_NUM_PACKETS; ++i) {
        for (;;) {
            st = vtlk_ring_pop(&ring, &hdr, actual, sizeof(actual));
            if (st == VTLK_OK) {
                break;
            }
            if (st == VTLK_ERR_EMPTY) {
                xproc_cpu_yield();
                continue;
            }
            fprintf(stderr, "Consumer: vtlk_ring_pop error (%d) at %u\n", (int)st, i);
            vtlk_shm_close(&shm);
            return 1;
        }

        assert(hdr.sequence == (uint64_t)i);
        assert(hdr.d_model == 64u);
        assert(hdr.n_tokens == 2u);
        assert(hdr.payload_bytes == XPROC_PAYLOAD_SZ);

        fill_synthetic_payload(expected, sizeof(expected), i);
        assert(memcmp(actual, expected, sizeof(expected)) == 0);
    }

    vtlk_shm_close(&shm);
    return 0;
}

static int run_producer(const char *exe_path)
{
    char shm_name[128];
    vtlk_shm_t shm;
    vtlk_ring_t ring;
    size_t req;
    uint32_t i;
    vtlk_header_t hdr;
    uint8_t payload[XPROC_PAYLOAD_SZ];
    vtlk_status_t st;

#ifdef _WIN32
    DWORD pid = GetCurrentProcessId();
    char cmd_line[1024];
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    DWORD exit_code = 1u;
#else
    pid_t pid = getpid();
    pid_t child;
    int status = 0;
#endif

    snprintf(shm_name, sizeof(shm_name), "/vectlink_xproc_%lu", (unsigned long)pid);
    req = vtlk_ring_required_bytes(XPROC_CAPACITY, XPROC_PAYLOAD_SZ, NULL);
    assert(req > 0u);

    st = vtlk_shm_create(&shm, shm_name, req);
    assert(st == VTLK_OK);

    st = vtlk_ring_init(&ring, shm.addr, shm.size, XPROC_CAPACITY, XPROC_PAYLOAD_SZ);
    assert(st == VTLK_OK);

#ifdef _WIN32
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    memset(&pi, 0, sizeof(pi));
    snprintf(cmd_line, sizeof(cmd_line), "\"%s\" --child %s", exe_path, shm_name);
    if (!CreateProcessA(NULL, cmd_line, NULL, NULL, FALSE, 0u, NULL, NULL, &si, &pi)) {
        fprintf(stderr, "Producer: CreateProcessA failed (%lu)\n",
                (unsigned long)GetLastError());
        vtlk_shm_close(&shm);
        return 1;
    }
#else
    child = fork();
    if (child < 0) {
        fprintf(stderr, "Producer: fork failed\n");
        vtlk_shm_close(&shm);
        vtlk_shm_unlink(shm_name);
        return 1;
    }
    if (child == 0) {
        execl(exe_path, exe_path, "--child", shm_name, (char *)NULL);
        _exit(127);
    }
#endif

    vtlk_header_init_defaults(&hdr);
    hdr.d_model = 64u;
    hdr.n_tokens = 2u;
    hdr.payload_bytes = XPROC_PAYLOAD_SZ;
    hdr.raw_bytes = XPROC_PAYLOAD_SZ;

    for (i = 0u; i < XPROC_NUM_PACKETS; ++i) {
        hdr.sequence = (uint64_t)i;
        fill_synthetic_payload(payload, sizeof(payload), i);
        for (;;) {
            st = vtlk_ring_push(&ring, &hdr, payload);
            if (st == VTLK_OK) {
                break;
            }
            if (st == VTLK_ERR_FULL) {
                xproc_cpu_yield();
                continue;
            }
            fprintf(stderr, "Producer: vtlk_ring_push error (%d) at %u\n", (int)st, i);
            vtlk_shm_close(&shm);
            vtlk_shm_unlink(shm_name);
            return 1;
        }
    }

#ifdef _WIN32
    WaitForSingleObject(pi.hProcess, INFINITE);
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    vtlk_shm_close(&shm);
    vtlk_shm_unlink(shm_name);
    assert(exit_code == 0u);
#else
    if (waitpid(child, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "Producer: child process failed\n");
        vtlk_shm_close(&shm);
        vtlk_shm_unlink(shm_name);
        return 1;
    }
    vtlk_shm_close(&shm);
    vtlk_shm_unlink(shm_name);
#endif

    printf("Cross-process IPC test passed (%u packets verified).\n", XPROC_NUM_PACKETS);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "--child") == 0) {
        return run_consumer(argv[2]);
    }
    return run_producer(argv[0]);
}
