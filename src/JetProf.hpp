// Optional cycle-count profiling hooks (compile with -DJET_PROFILE=1).
#pragma once
#include <cstdint>
#if defined(ESP_PLATFORM)
#include "esp_cpu.h"
static inline uint32_t jet_prof_now() { return esp_cpu_get_cycle_count(); }
#else
#include <chrono>
static inline uint32_t jet_prof_now() {
    return (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count();
}
#endif
// JP_TRI_PIX counts SPAN PIXELS WRITTEN (cnt only, no cycles): with JP_TRI_ROWS and JP_TRI_SETUP's count it
// separates the three ways a flat raster can be slow -- too many triangles (setup), too many short rows
// (per-row edge stepping), or too much overdraw (pixels written / pixels on screen).
enum { JP_TRI_SETUP = 0, JP_TRI_ROWS, JP_TRI_PIX, JP_K_TRI, JP_K_ROWS, JP_BAND_WALK, JP_XFORM, JP_SORT, JP_VERTS, JP_OBJ, JP_EMIT, JP_EMIT_CULL, JP_EMIT_BUILD, JP_EMIT_TAIL, JP_N };
extern uint32_t jet_prof_cyc[JP_N];
extern uint32_t jet_prof_cnt[JP_N];
