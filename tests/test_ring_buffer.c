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
#include <malloc.h>
#endif

static void *test_alloc_aligned64(size_t bytes)
{
#ifdef _WIN32
    return _aligned_malloc(bytes, 64u);
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64u, bytes) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static void test_free_aligned64(void *ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

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
    uint8_t *buf;
    vtlk_ring_t ring;

    /* Non-power-of-two capacity must fail */
    assert(vtlk_ring_required_bytes(0u, 64u, &slot_size) == 0u);
    assert(vtlk_ring_required_bytes(3u, 64u, &slot_size) == 0u);
    assert(vtlk_ring_required_bytes(6u, 64u, &slot_size) == 0u);

    /* 64B header + 100B payload -> 164B -> aligned up to 192B (3 * 64B) */
    req = vtlk_ring_required_bytes(4u, 100u, &slot_size);
    assert(slot_size == 192u);
    assert(req == 192u + 4u * 192u);

    buf = (uint8_t *)test_alloc_aligned64(req + 64u);
    assert(buf != NULL);

    /* Unaligned pointer must be rejected with VTLK_ERR_INVALID_ARG */
    assert(vtlk_ring_init(&ring, buf + 8u, req, 4u, 100u) == VTLK_ERR_INVALID_ARG);
    assert(vtlk_ring_attach(&ring, buf + 8u, req) == VTLK_ERR_INVALID_ARG);

    /* Buffer too small */
    assert(vtlk_ring_init(&ring, buf, req - 1u, 4u, 100u) == VTLK_ERR_BUFFER_TOO_SMALL);
    assert(vtlk_ring_init(&ring, buf, req, 4u, 100u) == VTLK_OK);

    test_free_aligned64(buf);
}

static void test_fifo_full_empty_wraparound(void)
{
    uint8_t *arena;
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
    assert(req > 0u);
    arena = (uint8_t *)test_alloc_aligned64(req);
    assert(arena != NULL);

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

    test_free_aligned64(arena);
}

static void test_payload_corruption_detection(void)
{
    uint8_t *raw;
    vtlk_ring_t prod;
    vtlk_ring_t cons;
    size_t req;
    vtlk_header_t h_in;
    vtlk_header_t h_out;
    uint8_t p_in[64];
    uint8_t p_out[64];

    req = vtlk_ring_required_bytes(4u, 64u, NULL);
    assert(req > 0u);
    raw = (uint8_t *)test_alloc_aligned64(req);
    assert(raw != NULL);

    assert(vtlk_ring_init(&prod, raw, req, 4u, 64u) == VTLK_OK);
    assert(vtlk_ring_attach(&cons, raw, req) == VTLK_OK);

    /* Push packet into slot 0 */
    memset(p_in, 0xAB, sizeof(p_in));
    make_header(&h_in, 7u, sizeof(p_in));
    assert(vtlk_ring_push(&prod, &h_in, p_in) == VTLK_OK);

    /* Corrupt 1 byte inside slot 0 payload (offset 192 + 64 + 10) */
    raw[VTLK_RING_CTRL_SIZE + VTLK_HEADER_SIZE + 10u] ^= 0x01u;

    assert(vtlk_ring_pop(&cons, &h_out, p_out, sizeof(p_out)) == VTLK_ERR_CRC_MISMATCH);

    test_free_aligned64(raw);
}

static void test_zero_copy_slot_loan(void)
{
    uint8_t *arena;
    vtlk_ring_t prod;
    vtlk_ring_t cons;
    size_t req;
    vtlk_write_slot_t wslot;
    vtlk_write_slot_t wslot2;
    vtlk_read_slot_t rslot;
    vtlk_read_slot_t rslot2;
    vtlk_header_t h_tmpl;
    uint32_t i;

    req = vtlk_ring_required_bytes(4u, 256u, NULL);
    assert(req > 0u);
    arena = (uint8_t *)test_alloc_aligned64(req);
    assert(arena != NULL);

    assert(vtlk_ring_init(&prod, arena, req, 4u, 256u) == VTLK_OK);
    assert(vtlk_ring_attach(&cons, arena, req) == VTLK_OK);

    /* Acquire read on empty ring returns VTLK_ERR_EMPTY */
    assert(vtlk_ring_acquire_read(&cons, &rslot) == (int)VTLK_ERR_EMPTY);

    /* Acquire write, verify 64B payload alignment, verify double acquire returns VTLK_ERR_BUSY */
    assert(vtlk_ring_acquire_write(&prod, &wslot) == (int)VTLK_OK);
    assert(((uintptr_t)wslot.payload % 64u) == 0u);
    assert(wslot.capacity == 256u);
    assert(vtlk_ring_acquire_write(&prod, &wslot2) == (int)VTLK_ERR_BUSY);

    /* Abort leaves ring empty and frees producer slot loan */
    assert(vtlk_ring_abort_write(&prod, &wslot) == (int)VTLK_OK);
    assert(vtlk_ring_acquire_read(&cons, &rslot) == (int)VTLK_ERR_EMPTY);

    /* Re-acquire, write directly into SHM payload, commit */
    assert(vtlk_ring_acquire_write(&prod, &wslot) == (int)VTLK_OK);
    memset(wslot.payload, 0x5Au, 128u);
    make_header(&h_tmpl, 42u, 128u);
    assert(vtlk_ring_commit_write(&prod, &wslot, &h_tmpl, 128u) == (int)VTLK_OK);

    /* Acquire read, verify BUSY on second acquire_read, verify header & direct payload, release */
    assert(vtlk_ring_acquire_read(&cons, &rslot) == (int)VTLK_OK);
    assert(vtlk_ring_acquire_read(&cons, &rslot2) == (int)VTLK_ERR_BUSY);
    assert(rslot.header.sequence == 42u);
    assert(rslot.bytes == 128u);
    assert(((const uint8_t *)rslot.payload)[0] == 0x5Au);
    assert(((const uint8_t *)rslot.payload)[127] == 0x5Au);
    assert(vtlk_ring_release_read(&cons, &rslot) == (int)VTLK_OK);
    assert(vtlk_ring_acquire_read(&cons, &rslot) == (int)VTLK_ERR_EMPTY);

    /* Wrap-around stress with zero-copy slot loan */
    for (i = 0u; i < 32u; ++i) {
        size_t sz = (size_t)((i % 200u) + 1u);
        assert(vtlk_ring_acquire_write(&prod, &wslot) == (int)VTLK_OK);
        memset(wslot.payload, (int)(i & 0xFFu), sz);
        make_header(&h_tmpl, 500u + (uint64_t)i, (uint32_t)sz);
        assert(vtlk_ring_commit_write(&prod, &wslot, &h_tmpl, sz) == (int)VTLK_OK);

        assert(vtlk_ring_acquire_read(&cons, &rslot) == (int)VTLK_OK);
        assert(rslot.header.sequence == 500u + (uint64_t)i);
        assert(rslot.bytes == sz);
        assert(((const uint8_t *)rslot.payload)[0] == (uint8_t)(i & 0xFFu));
        assert(((const uint8_t *)rslot.payload)[sz - 1u] == (uint8_t)(i & 0xFFu));
        assert(vtlk_ring_release_read(&cons, &rslot) == (int)VTLK_OK);
    }

    /* vtlk_ring_close must return VTLK_ERR_BUSY while write or read loan is active */
    assert(vtlk_ring_acquire_write(&prod, &wslot) == (int)VTLK_OK);
    assert(vtlk_ring_close(&prod) == (int)VTLK_ERR_BUSY);
    memset(wslot.payload, 0xCC, 32u);
    make_header(&h_tmpl, 999u, 32u);
    assert(vtlk_ring_commit_write(&prod, &wslot, &h_tmpl, 32u) == (int)VTLK_OK);

    assert(vtlk_ring_acquire_read(&cons, &rslot) == (int)VTLK_OK);
    assert(vtlk_ring_close(&cons) == (int)VTLK_ERR_BUSY);
    assert(vtlk_ring_abort_read(&cons, &rslot) == (int)VTLK_OK);

    /* Packet remains readable after abort_read */
    assert(vtlk_ring_acquire_read(&cons, &rslot) == (int)VTLK_OK);
    assert(rslot.header.sequence == 999u);
    assert(vtlk_ring_release_read(&cons, &rslot) == (int)VTLK_OK);

    assert(vtlk_ring_close(&prod) == (int)VTLK_OK);
    assert(vtlk_ring_close(&cons) == (int)VTLK_OK);

    test_free_aligned64(arena);
}

int main(void)
{
    test_required_bytes_and_init();
    test_fifo_full_empty_wraparound();
    test_payload_corruption_detection();
    test_zero_copy_slot_loan();

    printf("All Vect-Link ring buffer tests passed.\n");
    return 0;
}
