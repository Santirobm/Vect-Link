#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "vectlink/transport.h"

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
#include <time.h>
#endif

#define DEFAULT_SHM_NAME    "/vectlink_ipc_demo"
#define DEFAULT_NUM_PACKETS 10000u
#define DEMO_CAPACITY       256u
#define DEMO_PAYLOAD_BYTES  128u

static void demo_yield(void)
{
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}

static void demo_sleep_ms(unsigned int ms)
{
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts;
    ts.tv_sec = (time_t)(ms / 1000u);
    ts.tv_nsec = (long)((ms % 1000u) * 1000000u);
    nanosleep(&ts, NULL);
#endif
}

static void fill_payload(uint8_t *buf, size_t len, uint32_t seq)
{
    size_t i;
    for (i = 0u; i < len; ++i) {
        buf[i] = (uint8_t)((seq * 31u + (uint32_t)i * 7u) & 0xFFu);
    }
}

int main(int argc, char **argv)
{
    const char *shm_name = (argc > 1) ? argv[1] : DEFAULT_SHM_NAME;
    uint32_t num_packets = (argc > 2) ? (uint32_t)strtoul(argv[2], NULL, 10) : DEFAULT_NUM_PACKETS;
    size_t req;
    vtlk_shm_t shm;
    vtlk_ring_t ring;
    vtlk_header_t hdr;
    uint8_t actual[DEMO_PAYLOAD_BYTES];
    uint8_t expected[DEMO_PAYLOAD_BYTES];
    uint32_t i;
    int retries;
    vtlk_status_t st;

    req = vtlk_ring_required_bytes(DEMO_CAPACITY, DEMO_PAYLOAD_BYTES, NULL);
    if (req == 0u) {
        fprintf(stderr, "agent_receiver: invalid ring parameters\n");
        return 1;
    }

    st = VTLK_ERR_SHM;
    for (retries = 0; retries < 200; ++retries) {
        st = vtlk_shm_open(&shm, shm_name, req);
        if (st == VTLK_OK) {
            break;
        }
        demo_sleep_ms(10u);
    }
    if (st != VTLK_OK) {
        fprintf(stderr, "agent_receiver: vtlk_shm_open(%s) failed (%d)\n", shm_name, (int)st);
        return 1;
    }

    st = VTLK_ERR_INVALID_MAGIC;
    for (retries = 0; retries < 200; ++retries) {
        st = vtlk_ring_attach(&ring, shm.addr, shm.size);
        if (st == VTLK_OK) {
            break;
        }
        demo_sleep_ms(5u);
    }
    if (st != VTLK_OK) {
        fprintf(stderr, "agent_receiver: vtlk_ring_attach failed (%d)\n", (int)st);
        vtlk_shm_close(&shm);
        return 1;
    }

    printf("agent_receiver: attached to '%s'. Waiting for %u packets...\n",
           shm_name, num_packets);
    fflush(stdout);

    for (i = 0u; i < num_packets; ++i) {
        for (;;) {
            st = vtlk_ring_pop(&ring, &hdr, actual, sizeof(actual));
            if (st == VTLK_OK) {
                break;
            }
            if (st == VTLK_ERR_EMPTY) {
                demo_yield();
                continue;
            }
            fprintf(stderr, "agent_receiver: pop error (%d) at seq %u\n", (int)st, i);
            vtlk_shm_close(&shm);
            return 1;
        }

        if (hdr.sequence != (uint64_t)i || hdr.payload_bytes != DEMO_PAYLOAD_BYTES) {
            fprintf(stderr, "agent_receiver: header mismatch at seq %u (got %llu)\n",
                    i, (unsigned long long)hdr.sequence);
            vtlk_shm_close(&shm);
            return 1;
        }

        fill_payload(expected, sizeof(expected), i);
        if (memcmp(actual, expected, sizeof(expected)) != 0) {
            fprintf(stderr, "agent_receiver: payload mismatch at seq %u\n", i);
            vtlk_shm_close(&shm);
            return 1;
        }
    }

    printf("agent_receiver: received and verified %u packets.\n", num_packets);
    vtlk_shm_close(&shm);
    vtlk_shm_unlink(shm_name);
    return 0;
}
