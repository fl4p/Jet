// JET_WIRE_SWAP: the framebuffer holds byte-swapped (wire-order) RGB565 so the
// panel DMA push needs zero conversion passes (saves a full read+write of the
// frame per frame; PSRAM bandwidth is the bottleneck on ESP32-S3). All
// framebuffer writes pre-swap, all reads unswap. Covers ONLY the flat
// painter's-config paths (clear / span fill / fog blend) — guarded in
// Renderer.cpp. Host frame dumpers must unswap after rendering.
#ifndef JET_WIRE_SWAP
#define JET_WIRE_SWAP 0
#endif

static inline uint16_t jetWs565(uint16_t c) {
#if JET_WIRE_SWAP
    return (uint16_t)((c << 8) | (c >> 8));
#else
    return c;
#endif
}
