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
enum { JP_TRI_SETUP = 0, JP_TRI_ROWS, JP_BAND_WALK, JP_XFORM, JP_SORT, JP_VERTS, JP_OBJ, JP_EMIT, JP_EMIT_CULL, JP_EMIT_BUILD, JP_EMIT_TAIL, JP_N };
extern uint32_t jet_prof_cyc[JP_N];
extern uint32_t jet_prof_cnt[JP_N];
