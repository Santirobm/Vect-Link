#include "vectlink/transport.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sched.h>
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
    uint8_t payload[DEMO_PAYLOAD_BYTES];
    uint32_t i;
    vtlk_status_t st;

    req = vtlk_ring_required_bytes(DEMO_CAPACITY, DEMO_PAYLOAD_BYTES, NULL);
    if (req == 0u) {
        fprintf(stderr, "agent_sender: invalid ring parameters\n");
        return 1;
    }

    st = vtlk_shm_create(&shm, shm_name, req);
    if (st != VTLK_OK) {
        fprintf(stderr, "agent_sender: vtlk_shm_create(%s) failed (%d)\n", shm_name, (int)st);
        return 1;
    }

    st = vtlk_ring_init(&ring, shm.addr, shm.size, DEMO_CAPACITY, DEMO_PAYLOAD_BYTES);
    if (st != VTLK_OK) {
        fprintf(stderr, "agent_sender: vtlk_ring_init failed (%d)\n", (int)st);
        vtlk_shm_close(&shm);
        vtlk_shm_unlink(shm_name);
        return 1;
    }

    printf("agent_sender: SHM '%s' ready (%zu bytes). Sending %u packets...\n",
           shm_name, req, num_packets);
    fflush(stdout);

    vtlk_header_init_defaults(&hdr);
    hdr.d_model = 64u;
    hdr.n_tokens = 2u;
    hdr.payload_bytes = DEMO_PAYLOAD_BYTES;
    hdr.raw_bytes = DEMO_PAYLOAD_BYTES;

    for (i = 0u; i < num_packets; ++i) {
        hdr.sequence = (uint64_t)i;
        fill_payload(payload, sizeof(payload), i);

        for (;;) {
            st = vtlk_ring_push(&ring, &hdr, payload);
            if (st == VTLK_OK) {
                break;
            }
            if (st == VTLK_ERR_FULL) {
                demo_yield();
                continue;
            }
            fprintf(stderr, "agent_sender: push error (%d) at seq %u\n", (int)st, i);
            vtlk_shm_close(&shm);
            vtlk_shm_unlink(shm_name);
            return 1;
        }
    }

    printf("agent_sender: sent %u packets successfully.\n", num_packets);
    vtlk_shm_close(&shm);
    return 0;
}
