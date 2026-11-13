#include "vectlink/transport.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void make_header(vtlk_header_t *hdr, uint64_t seq, uint32_t payload_bytes)
{
    vtlk_header_init_defaults(hdr);
    hdr->d_model = 64u;
    hdr->n_tokens = 2u;
    hdr->payload_bytes = payload_bytes;
    hdr->raw_bytes = payload_bytes;
    hdr->sequence = seq;
}

static void test_required_bytes_and_init(void)
{
    uint32_t slot_size = 0u;
    size_t req;
    uint64_t buf[512];
    vtlk_ring_t ring;

    /* Non-power-of-two capacity must fail */
    assert(vtlk_ring_required_bytes(0u, 64u, &slot_size) == 0u);
    assert(vtlk_ring_required_bytes(3u, 64u, &slot_size) == 0u);
    assert(vtlk_ring_required_bytes(6u, 64u, &slot_size) == 0u);

    /* 64B header + 100B payload -> 164B -> aligned up to 192B (3 * 64B) */
    req = vtlk_ring_required_bytes(4u, 100u, &slot_size);
    assert(slot_size == 192u);
    assert(req == 192u + 4u * 192u);

    /* Buffer too small */
    assert(vtlk_ring_init(&ring, buf, req - 1u, 4u, 100u) == VTLK_ERR_BUFFER_TOO_SMALL);
    assert(vtlk_ring_init(&ring, buf, req, 4u, 100u) == VTLK_OK);
}

static void test_fifo_full_empty_wraparound(void)
{
    uint64_t arena[1024];
    vtlk_ring_t prod;
    vtlk_ring_t cons;
    uint32_t capacity = 4u;
    uint32_t max_payload = 128u;
    size_t req;
    vtlk_header_t h_in;
    vtlk_header_t h_out;
    uint8_t p_in[128];
    uint8_t p_out[128];
    uint64_t i;

    req = vtlk_ring_required_bytes(capacity, max_payload, NULL);
    assert(req > 0u && req <= sizeof(arena));

    assert(vtlk_ring_init(&prod, arena, req, capacity, max_payload) == VTLK_OK);
    assert(vtlk_ring_attach(&cons, arena, req) == VTLK_OK);

    /* Empty on start */
    assert(vtlk_ring_pop(&cons, &h_out, p_out, sizeof(p_out)) == VTLK_ERR_EMPTY);

    /* Fill to capacity */
    for (i = 0u; i < capacity; ++i) {
        memset(p_in, (int)(0x10u + i), 64u);
        make_header(&h_in, i, 64u);
        assert(vtlk_ring_push(&prod, &h_in, p_in) == VTLK_OK);
    }

    /* Next push must return VTLK_ERR_FULL */
    make_header(&h_in, 999u, 64u);
    assert(vtlk_ring_push(&prod, &h_in, p_in) == VTLK_ERR_FULL);

    /* Drain all in FIFO order */
    for (i = 0u; i < capacity; ++i) {
        memset(p_out, 0, sizeof(p_out));
        assert(vtlk_ring_pop(&cons, &h_out, p_out, sizeof(p_out)) == VTLK_OK);
        assert(h_out.sequence == i);
        assert(h_out.payload_bytes == 64u);
        assert(p_out[0] == (uint8_t)(0x10u + i));
        assert(p_out[63] == (uint8_t)(0x10u + i));
    }
    assert(vtlk_ring_pop(&cons, &h_out, p_out, sizeof(p_out)) == VTLK_ERR_EMPTY);

    /* Wrap-around stress: 100 packets (> 25x capacity) */
    for (i = 0u; i < 100u; ++i) {
        uint32_t sz = (uint32_t)((i % 64u) + 1u);
        memset(p_in, (int)(i & 0xFFu), sz);
        make_header(&h_in, 1000u + i, sz);
        assert(vtlk_ring_push(&prod, &h_in, p_in) == VTLK_OK);

        memset(p_out, 0, sizeof(p_out));
        assert(vtlk_ring_pop(&cons, &h_out, p_out, sizeof(p_out)) == VTLK_OK);
        assert(h_out.sequence == 1000u + i);
        assert(h_out.payload_bytes == sz);
        assert(memcmp(p_in, p_out, sz) == 0);
    }
}

static void test_payload_corruption_detection(void)
{
    uint64_t arena[512];
    uint8_t *raw = (uint8_t *)arena;
    vtlk_ring_t prod;
    vtlk_ring_t cons;
    size_t req;
    vtlk_header_t h_in;
    vtlk_header_t h_out;
    uint8_t p_in[64];
    uint8_t p_out[64];

    req = vtlk_ring_required_bytes(4u, 64u, NULL);
    assert(req > 0u && req <= sizeof(arena));
    assert(vtlk_ring_init(&prod, arena, req, 4u, 64u) == VTLK_OK);
    assert(vtlk_ring_attach(&cons, arena, req) == VTLK_OK);

    /* Push packet into slot 0 */
    memset(p_in, 0xAB, sizeof(p_in));
    make_header(&h_in, 7u, sizeof(p_in));
    assert(vtlk_ring_push(&prod, &h_in, p_in) == VTLK_OK);

    /* Corrupt 1 byte inside slot 0 payload (offset 192 + 64 + 10) */
    raw[VTLK_RING_CTRL_SIZE + VTLK_HEADER_SIZE + 10u] ^= 0x01u;

    assert(vtlk_ring_pop(&cons, &h_out, p_out, sizeof(p_out)) == VTLK_ERR_CRC_MISMATCH);
}

int main(void)
{
    test_required_bytes_and_init();
    test_fifo_full_empty_wraparound();
    test_payload_corruption_detection();

    printf("All Vect-Link ring buffer tests passed.\n");
    return 0;
}
