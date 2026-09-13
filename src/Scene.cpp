#include "Scene.hpp"
#include "TrigLUT.hpp"
#include "Renderer.hpp"
#include "BlendSpans.hpp"
#include "WireSwap565.hpp"
#include "JetConfig.hpp"
#if JET_PROFILE
#include "JetProf.hpp"
#endif
#include <cstring> // For memset
#include <algorithm> // For std::min, std::max
#include <cmath> // For sqrtf (per-object distance fade / LOD pick)
#ifndef JET_CULL_SLIVERS
#define JET_CULL_SLIVERS 1   // drop triangles thinner than half a pixel at emit time (see emitTri)
#endif
#if JET_CULL_SLIVERS
namespace Renderer {
float   jetSliverMinThickness = 0.05f;   // pixels: thinner than this is dropped (degenerate / sign-flipped back faces)
float   jetSliverPushMaxThickness = 0.5f; // pixels: thinner than this is sorted behind its neighbours (see emitTri)
int32_t jetSliverPushZ = 80;             // world units a sub-pixel triangle is sorted farther back (~2 buckets of 40)
}
#endif

namespace Renderer {

// Static emissive material used to render triangles whose lighting has been
// pre-baked into Triangle::bakedColor.  The colour field is overwritten
// per-triangle before emitTri() is called; all other fields are constant.
static Material s_bakedMat;
static bool     s_bakedMatInit = false;
static void initBakedMat() {
    if (s_bakedMatInit) return;
    s_bakedMat.diffuseMap  = nullptr;
    s_bakedMat.shader      = nullptr;
    s_bakedMat.emissive    = true;
    s_bakedMat.alpha       = 255;
    s_bakedMat.diffuse     = 0;
    s_bakedMat.specular    = 0;
    s_bakedMat.shadingMode = ShadingMode::FLAT;
    s_bakedMat.name        = nullptr;
    s_bakedMatInit = true;
}

#if LIGHTING
// Diffuse-only Lambert shading, evaluated in whatever frame N and L
// share (object-local for the precompute path below). Matches the
// diffuse branch of Renderer.cpp's jetShadeBrightness when
// specularCoef == 0: 12-bit reduce, clamp, square-falloff, scale by
// lightIntensity, scale by diffuseCoef, clamp.
//
// The specular branch is intentionally absent — this helper is only
// used for objects whose materials are ALL non-specular, which is the
// exact precondition for skipping the view-space normal transform.
// Inlining the specular branch here would mean its `N.z < 0` test
// would be reading an object-local component instead of view-space Z
// and give wrong results.
static inline uint16_t sceneLambertDiffuse(const Vector3& N, const Vector3& L,
                                           uint16_t lightIntensity,
                                           uint8_t diffuseCoef)
{
    if (lightIntensity > 255) lightIntensity = 255;
    int64_t lit = Vector3::dotProduct(N, L);
    if (lit <= 0) return 0;
    uint32_t lambert = (uint32_t)(lit >> 12);
    if (lambert > 255) lambert = 255;
    lambert = (lambert * lambert + 128) >> 8;       // squared falloff
    lambert = (lambert * lightIntensity) >> 8;
    uint32_t diffuseTerm = (lambert * diffuseCoef) >> 8;
    if (diffuseTerm > 255) diffuseTerm = 255;
    return (uint16_t)diffuseTerm;
}
#endif

// Returns true if the object's AABB is entirely outside the view frustum.
bool Scene::cullObject(Object* obj,
                       int32_t camCosX, int32_t camSinX,
                       int32_t camCosY, int32_t camSinY,
                       int32_t camCosZ, int32_t camSinZ) const {
    if (obj->isBillboard) return false; // keep billboards simple for now

    Vector3 camPos(camera->position);
    Vector3 objPos(obj->position);
    const Vector3& bMin = obj->boundingBoxMin;
    const Vector3& bMax = obj->boundingBoxMax;

    int outLeft = 0, outRight = 0, outTop = 0, outBottom = 0, outNear = 0, outFar = 0;
    float   fovFactor = camera->fovFactor;
    int32_t nearPlane = camera->nearPlane;
    int32_t farPlane  = camera->farPlane;

    // ---- Quick sphere-vs-frustum classification ----------------------------
    // One point transform (~20 ops) instead of the 8-corner AABB test
    // (~240 ops) for the vast majority of objects. Conservative bounding
    // sphere: centre at position+centreVolume (same rotation-ignoring
    // convention as prepareFrame's far pre-cull) with radius = longest
    // AABB dimension, which always covers the true half-diagonal
    // (halfDiag <= 0.866*maxExtent) plus slack for rotated meshes.
    //
    //   * sphere entirely outside ANY single frustum half-space → culled.
    //   * sphere entirely inside ALL half-spaces → visible.
    //   * straddling a boundary → fall through to the precise 8-corner test.
    //
    // The side-plane half-space checks use the plane equation directly
    // (signed distance × normal length), so they're valid regardless of
    // the sphere's z sign — no divides, no per-object sqrt (plane normal
    // lengths cullPlaneLh/Lv are cached per frame in prepareFrame()).
    {
        const float r = (float)std::max({bMax.x - bMin.x,
                                         bMax.y - bMin.y,
                                         bMax.z - bMin.z});
        constexpr float invFps = 1.0f / (float)FIXED_POINT_SCALE;
        const float cYc = (float)camCosY * invFps, cYs = (float)camSinY * invFps;
        const float cXc = (float)camCosX * invFps, cXs = (float)camSinX * invFps;
        const float cZc = (float)camCosZ * invFps, cZs = (float)camSinZ * invFps;
        const float px = (float)(objPos.x + obj->centreVolume.x - camPos.x);
        const float py = (float)(objPos.y + obj->centreVolume.y - camPos.y);
        const float pz = (float)(objPos.z + obj->centreVolume.z - camPos.z);
        // Camera rotation Y, X, Z — same order as the corner loop below.
        const float t1x =  px * cYc + pz * cYs;
        const float t1z = -px * cYs + pz * cYc;
        const float t2y =  py * cXc - t1z * cXs;
        const float t2z =  py * cXs + t1z * cXc;
        const float cxv =  t1x * cZc - t2y * cZs;
        const float cyv =  t1x * cZs + t2y * cZc;
        const float czv =  t2z;

        if (czv + r < (float)nearPlane) return true;   // fully behind
        if (czv - r > (float)farPlane)  return true;   // fully beyond

        const float hw  = (float)screenWidth  * 0.5f;
        const float hh  = (float)screenHeight * 0.5f;
        const float rLh = r * cullPlaneLh;
        const float rLv = r * cullPlaneLv;
        // Signed distance × plane length to each side plane: positive =
        // outside that plane's visible half-space.
        const float dR =  cxv * fovFactor - czv * hw;   // right of right plane
        const float dL = -cxv * fovFactor - czv * hw;   // left of left plane
        const float dT =  cyv * fovFactor - czv * hh;   // above top plane
        const float dB = -cyv * fovFactor - czv * hh;   // below bottom plane
        if (dR > rLh || dL > rLh || dT > rLv || dB > rLv) return true;

        if (czv - r > (float)nearPlane && czv + r < (float)farPlane &&
            dR < -rLh && dL < -rLh && dT < -rLv && dB < -rLv)
            return false;   // fully inside — skip the 8-corner test
    }

    const bool rotated = obj->rotation.x != 0 || obj->rotation.y != 0 || obj->rotation.z != 0
                         || obj->rotationFloat;
    int32_t objCosX=FIXED_POINT_SCALE, objSinX=0, objCosY=FIXED_POINT_SCALE, objSinY=0;
    int32_t objCosZ=FIXED_POINT_SCALE, objSinZ=0;
    if (rotated) {
#if FLOAT_CAMERA_ANGLES
        if (obj->rotationFloat) {
            objCosX=(int32_t)(lookupCos(obj->rotFx)*FIXED_POINT_SCALE); objSinX=(int32_t)(lookupSin(obj->rotFx)*FIXED_POINT_SCALE);
            objCosY=(int32_t)(lookupCos(obj->rotFy)*FIXED_POINT_SCALE); objSinY=(int32_t)(lookupSin(obj->rotFy)*FIXED_POINT_SCALE);
            objCosZ=(int32_t)(lookupCos(obj->rotFz)*FIXED_POINT_SCALE); objSinZ=(int32_t)(lookupSin(obj->rotFz)*FIXED_POINT_SCALE);
        } else
#endif
        {
        objCosX=lookupCosI(obj->rotation.x); objSinX=lookupSinI(obj->rotation.x);
        objCosY=lookupCosI(obj->rotation.y); objSinY=lookupSinI(obj->rotation.y);
        objCosZ=lookupCosI(obj->rotation.z); objSinZ=lookupSinI(obj->rotation.z);
        }
    }

    for (int i = 0; i < 8; ++i) {
        Vector3 p(
            (i & 1) ? bMax.x : bMin.x,
            (i & 2) ? bMax.y : bMin.y,
            (i & 4) ? bMax.z : bMin.z);

        // Static scenery needs no identity rotations at each corner.
        if (rotated) {
            // Object rotation (X, Y, Z) — same order as renderObject
            p.assign(p.x,
                     (p.y * objCosX - p.z * objSinX) / FIXED_POINT_SCALE,
                     (p.y * objSinX + p.z * objCosX) / FIXED_POINT_SCALE);
            p.assign((p.x * objCosY + p.z * objSinY) / FIXED_POINT_SCALE,
                      p.y,
                     (-p.x * objSinY + p.z * objCosY) / FIXED_POINT_SCALE);
            p.assign((p.x * objCosZ - p.y * objSinZ) / FIXED_POINT_SCALE,
                     (p.x * objSinZ + p.y * objCosZ) / FIXED_POINT_SCALE,
                      p.z);
        }

        // Translation
        p.add(objPos);
        p.add(camPos.inverse());

        // Camera rotation (Y, X, Z)
        Vector3 r;
        r.assign((p.x * camCosY + p.z * camSinY) / FIXED_POINT_SCALE,
                  p.y,
                 (-p.x * camSinY + p.z * camCosY) / FIXED_POINT_SCALE); p = r;
        r.assign(p.x,
                 (p.y * camCosX - p.z * camSinX) / FIXED_POINT_SCALE,
                 (p.y * camSinX + p.z * camCosX) / FIXED_POINT_SCALE); p = r;
        r.assign((p.x * camCosZ - p.y * camSinZ) / FIXED_POINT_SCALE,
                 (p.x * camSinZ + p.y * camCosZ) / FIXED_POINT_SCALE,
                  p.z); p = r;

        if (p.z < nearPlane) { outNear++; continue; }
        if (p.z > farPlane)  { outFar++;  continue; }
        if (p.z <= 0)        { outNear++; continue; }

        const float invZ = fovFactor / (float)p.z;
        int32_t sx = (int32_t)(p.x * invZ) + screenWidth / 2;
        int32_t sy = screenHeight / 2 - (int32_t)(p.y * invZ);
        if (sx < 0)            outLeft++;
        if (sx > screenWidth)  outRight++;
        if (sy < 0)            outTop++;
        if (sy > screenHeight) outBottom++;
    }

    return (outNear == 8 || outFar == 8 ||
            outLeft == 8 || outRight == 8 ||
            outTop  == 8 || outBottom == 8);
}

Scene::Scene(uint16_t* framebuffer, uint16_t* zBuffer, int screenWidth, int screenHeight)
    : camera(nullptr), directionalLight(nullptr), ambientLight(nullptr),
      framebuffer(framebuffer), zBuffer(zBuffer), screenWidth(screenWidth), screenHeight(screenHeight), scanlinesUpdated(nullptr) {
    initializeTrigTables();
    initBakedMat();
    renderer = new Rasterizer(framebuffer, screenWidth, screenHeight, zBuffer);
    postFX = new PostFX(screenWidth, screenHeight);
}
Scene::~Scene() {
    if (renderer) {
        delete renderer;
        renderer = nullptr;
    }
    if (postFX) {
        delete postFX;
        postFX = nullptr;
    }
}

void Scene::reserveQueues(size_t n) {
    queueCap = (uint32_t)n;
    renderQueue.resize(n);        // materialised once: the lanes write by index, nothing ever reallocates
    for (auto& v : bandOrder) v.reserve(n / 2);
    renderBuckets.resize(n);
    renderYSpan.resize(2 * n);
    renderOrder.reserve(n);
#if TEXTURE_MAPPING
    textureQueue.reserve(n);
#endif
    // lane 1 of a split prepareFrame emits about half the frame
    lanes[1].transformedVertices.reserve(256);
    lanes[0].transformedVertices.reserve(256);
}

void Scene::buildBandLists(int bandRows) {
    bandListRows = 0;
    if (bandRows <= 0) return;
    const int nb = (screenHeight + bandRows - 1) / bandRows;
    if (nb > MaxBandLists) return;
    for (int b = 0; b < nb; ++b) bandOrder[b].clear();
    for (const int32_t idx : renderOrder) {
        int b0 = renderYSpan[2 * idx] / bandRows, b1 = renderYSpan[2 * idx + 1] / bandRows;
        if (b0 < 0) b0 = 0;
        if (b1 >= nb) b1 = nb - 1;
        for (int b = b0; b <= b1; ++b) bandOrder[b].push_back(idx);
    }
    bandListRows = bandRows;
}

void Scene::setFramebuffer(uint16_t *newBuffer) {
    framebuffer = newBuffer;
    renderer->setFramebuffer(newBuffer);
}

void Scene::resize(uint16_t* newFramebuffer, uint16_t* newZBuffer,
                   int newWidth, int newHeight) {
    framebuffer  = newFramebuffer;
    zBuffer      = newZBuffer;
    screenWidth  = newWidth;
    screenHeight = newHeight;
    if (renderer) {
        renderer->resize(newFramebuffer, newZBuffer, newWidth, newHeight);
    }
    // PostFX caches its own dimensions and (when enabled) owns scratch
    // buffers sized to them. Easiest correct thing is to rebuild it.
    if (postFX) {
        delete postFX;
        postFX = new PostFX(newWidth, newHeight);
    }
}

void Scene::addObject(Object* obj) {
    objects.push_back(obj);
}

void Scene::addPointLight(PointLight* light) {
    pointLights.push_back(light);
}

void Scene::setCamera(Camera* cam) {
    camera = cam;
    renderer->camera = cam;
}

void Scene::setDirectionalLight(DirectionalLight* light) {
    directionalLight = light;
}

void Scene::setAmbientLight(AmbientLight* light) {
    ambientLight = light;
}

#if MAX_PICK_QUERIES > 0
void Scene::setPickQueries(const PickQuery* queries, int count) {
    if (count < 0) count = 0;
    if (count > MAX_PICK_QUERIES) count = MAX_PICK_QUERIES;
    pickQueryCount = count;
    for (int i = 0; i < count; ++i) {
        pickQueries[i] = queries[i];
    }
    // Slots beyond `count` keep their previous contents but are ignored
    // by the rasterizer (it loops 0..pickQueryCount). We don't bother
    // zeroing them.
}
#endif

void Scene::reconstructCheckerboard() {
    // For each pixel that was NOT rendered this frame (opposite parity),
    // approximate it as a 3-tap average of its own previous-frame value and
    // its two freshly-rendered horizontal neighbours. The old value provides
    // temporal stability; the fresh neighbours provide spatial accuracy.
    // At the left/right edges the missing neighbour is omitted (2-tap).
    // This is a separate pass over the completed framebuffer so that painter's
    // algorithm overdraw doesn't cause repeated averaging of the gap pixel.
    const int cbParity = renderEvenLines ? 0 : 1;
    for (int y = 0; y < screenHeight; ++y) {
        uint16_t* row = framebuffer + y * screenWidth;
        for (int x = 0; x < screenWidth; ++x) {
            if (((x ^ y) & 1) == cbParity) continue; // freshly rendered — skip
            const uint16_t old = row[x];
            if (x == 0) {
                // 2-tap: old + right
                const uint16_t r = row[x + 1];
                row[x] = (uint16_t)(
                    ((((old >> 11) & 0x1F) + ((r >> 11) & 0x1F)) >> 1 << 11) |
                    ((((old >>  5) & 0x3F) + ((r >>  5) & 0x3F)) >> 1 <<  5) |
                     (((old        & 0x1F) + ( r        & 0x1F)) >> 1));
            } else if (x == screenWidth - 1) {
                // 2-tap: left + old
                const uint16_t l = row[x - 1];
                row[x] = (uint16_t)(
                    ((((l >> 11) & 0x1F) + ((old >> 11) & 0x1F)) >> 1 << 11) |
                    ((((l >>  5) & 0x3F) + ((old >>  5) & 0x3F)) >> 1 <<  5) |
                     ((( l       & 0x1F) + ( old        & 0x1F)) >> 1));
            } else {
                // 3-tap: left + old + right
                const uint16_t l = row[x - 1];
                const uint16_t r = row[x + 1];
                row[x] = (uint16_t)(
                    ((((l >> 11) & 0x1F) + ((old >> 11) & 0x1F) + ((r >> 11) & 0x1F)) / 3 << 11) |
                    ((((l >>  5) & 0x3F) + ((old >>  5) & 0x3F) + ((r >>  5) & 0x3F)) / 3 <<  5) |
                     ((( l       & 0x1F) + ( old        & 0x1F) + ( r        & 0x1F)) / 3));
            }
        }
    }
}

#if defined(CONFIG_IDF_TARGET_ESP32S3)
// clearBuffers() row fill using 128-bit EE.VST.128.XP stores (4 × uint32
// per instruction). dest must be 16-byte aligned; n32 need not be a
// multiple of 4 — trailing elements are handled with scalar stores.
static void jet_fill_u32x16(uint32_t* dest, uint32_t val, int n32) {
    const int n16 = n32 >> 2;
    if (n16 > 0) {
        __asm__ volatile (
            "ee.movi.32.q q0, %[v], 0\n\t"
            "ee.movi.32.q q0, %[v], 1\n\t"
            "ee.movi.32.q q0, %[v], 2\n\t"
            "ee.movi.32.q q0, %[v], 3\n\t"
            : : [v] "r"(val)
        );
        // EE.VST.128.XP takes the post-increment stride as a register,
        // not a bare immediate: as += stride_reg after each 128-bit store.
        // 4x-unrolled: the volatile asm + memory clobber stops GCC from
        // unrolling this itself, so retire 64 bytes per loop iteration by
        // hand and mop up the remainder one store at a time.
        const int stride = 16;
        for (int i = n16 >> 2; i > 0; --i) {
            __asm__ volatile (
                "ee.vst.128.xp q0, %[p], %[s]\n\t"
                "ee.vst.128.xp q0, %[p], %[s]\n\t"
                "ee.vst.128.xp q0, %[p], %[s]\n\t"
                "ee.vst.128.xp q0, %[p], %[s]\n\t"
                : [p] "+r"(dest) : [s] "r"(stride) : "memory"
            );
        }
        for (int i = n16 & 3; i > 0; --i) {
            __asm__ volatile (
                "ee.vst.128.xp q0, %[p], %[s]\n\t"
                : [p] "+r"(dest) : [s] "r"(stride) : "memory"
            );
        }
    }
    for (int r = n32 & 3; r-- > 0; ) *dest++ = val;
}
#endif

void PERF_CRITICAL Scene::clearBuffers() {
    //cast the framebuffer to a 32-bit pointer
    uint32_t* framebuffer32 = (uint32_t*)framebuffer;

    // Wireframe mode forces a solid black background regardless of the
    // configured backcolor / gradient so the outlines pop. We detour the
    // gradient and backcolor for the duration of this clear and restore
    // them on the way out so the host-visible state is unchanged.
    uint16_t* savedGradient = backgroundGradientColors;
    uint16_t  savedBack     = backcolor;
    const bool wireClear = (renderer && renderer->wireframeMode);
    if (wireClear) {
        backgroundGradientColors = nullptr;
        backcolor = 0;
    }

    if (clearRenderBuffer) {
        if (renderer->checkerboardMode) {
            // Checkerboard clear: only wipe current-parity pixels. Opposite-parity
            // pixels keep last frame's values; when CHECKERBOARD_RECONSTRUCTION
            // is enabled the rasterizer fills them in-situ as it renders each
            // current-parity pixel.
            const int cbParity = renderEvenLines ? 0 : 1;
            for (int y = 0; y < screenHeight; ++y) {
                #if DEBUG_OVERDRAW
                const uint16_t lineColor = 0;
                #else
                const uint16_t lineColor = jetWs565(backgroundGradientColors
                                           ? backgroundGradientColors[y] : backcolor);
                #endif
                uint16_t* row = framebuffer + y * screenWidth;
                for (int x = 0; x < screenWidth; ++x) {
                    if (((x ^ y) & 1) == cbParity) {
                        row[x] = lineColor;
                    }
                }
            }
            #if Z_BUFFERING
            memset(zBuffer, 0xFF, (size_t)screenWidth * screenHeight * sizeof(uint16_t));
            #endif
        } else if (renderer->interlacedMode) {
            for (int y = (int)renderEvenLines; y < screenHeight; y += 2) {
                #if DEBUG_OVERDRAW
                uint16_t lineColor = 0;
                uint32_t lineColor32 = 0;
                #else
                uint16_t lineColor = jetWs565(backgroundGradientColors ? backgroundGradientColors[y] : backcolor);
                uint32_t lineColor32 = (lineColor << 16) | lineColor;
                #endif
                #if HALF_WIDTH_BUFFERS
                const int divisor = 4;
                #else
                const int divisor = 2;
                #endif
                #if FIELD_BUFFERS
                // Field-buffer layout: packed half-height. Row index = y>>1.
                uint32_t* lineStart = framebuffer32 + (y >> 1) * (screenWidth / divisor);
                #else
                uint32_t* lineStart = framebuffer32 + y * (screenWidth / divisor);
                #endif
#if defined(CONFIG_IDF_TARGET_ESP32S3)
                jet_fill_u32x16(lineStart, lineColor32, screenWidth / divisor);
#else
                for (int x = 0; x < screenWidth / divisor; x++) {
                    lineStart[x] = lineColor32;
                }
#endif
                #if Z_BUFFERING
                #if HALF_WIDTH_BUFFERS
                memset(zBuffer + (y / 2) * (screenWidth / 2), 0xFF, (screenWidth / 2) * sizeof(uint16_t));
                #else
                memset(zBuffer + y * screenWidth, 0xFF, screenWidth * sizeof(uint16_t));
                #endif
                #endif
            }
        } else {
            // Non-interlaced clear: fill every row. Honour the per-row
            // background gradient if one is set; otherwise fall back to
            // a solid backcolor. Note that `memset(fb, backcolor, ...)` is
            // wrong for a 16-bit backcolor because memset writes bytes —
            // we'd get backcolor's low byte duplicated. Pack two pixels per
            // 32-bit store and walk rows so the gradient (if any) actually
            // shows up.
            // HALF_WIDTH_BUFFERS makes the buffer half-width. FIELD_BUFFERS
            // additionally halves the height (each field buffer covers every other
            // row). When only HALF_WIDTH_BUFFERS is set the buffer is half-width
            // but FULL-height, so rowCount must be screenHeight in that case.
            #if HALF_WIDTH_BUFFERS
            const int rowPixels = screenWidth / 2;
            #if FIELD_BUFFERS
            const int rowCount  = screenHeight / 2;
            #else
            const int rowCount  = screenHeight;
            #endif
            #else
            const int rowPixels = screenWidth;
            const int rowCount  = screenHeight;
            #endif
            const int row32     = rowPixels / 2; // pairs of pixels per row
            // Respect yBandMin/yBandMax so the clear only touches the rows
            // assigned to this band (critical for virtual-base-pointer band
            // rendering where the buffer only covers [yBandMin, yBandMax)).
            const int yClearStart = renderer ? renderer->yBandMin             : 0;
            int yClearEnd   = renderer ? std::min(renderer->yBandMax, rowCount) : rowCount;
            if (backgroundClearMaxRows > 0) yClearEnd = std::min(yClearEnd, backgroundClearMaxRows);
            for (int y = yClearStart; y < yClearEnd; ++y) {
                #if DEBUG_OVERDRAW
                const uint16_t lineColor = 0;
                #else
                // Source row in the gradient table maps 1:1 with the
                // logical screen row (`screenHeight`). When the back buffer
                // is half-height (HALF_WIDTH_BUFFERS implies a y/2-style
                // layout in some configs), index by `y` directly because
                // rowCount already accounts for the halving.
                const uint16_t lineColor = jetWs565(backgroundGradientColors
                                           ? backgroundGradientColors[y * (screenHeight / rowCount)]
                                           : backcolor);
                #endif
                const uint32_t lineColor32 = ((uint32_t)lineColor << 16) | lineColor;
                uint32_t* lineStart = framebuffer32 + y * row32;
#if defined(CONFIG_IDF_TARGET_ESP32S3)
                jet_fill_u32x16(lineStart, lineColor32, row32);
#else
                for (int x = 0; x < row32; ++x) lineStart[x] = lineColor32;
#endif
            }
            #if Z_BUFFERING
            // Z-buffer stride matches the rasterizer's depth-buffer layout
            // (see ZBUFFER_STRIDE in Renderer.hpp): half-width when
            // HALF_WIDTH_BUFFERS is on, per-pixel otherwise. Height is the
            // full screen height regardless.
            #if HALF_WIDTH_BUFFERS
            memset(zBuffer, 0xFF, (size_t)(screenWidth / 2) * screenHeight * sizeof(uint16_t));
            #else
            memset(zBuffer, 0xFF, (size_t)screenWidth * screenHeight * sizeof(uint16_t));
            #endif
            #endif
        }
    }
    //memset(scanlinesUpdated, 0, screenHeight * sizeof(bool));

    // Restore the host-visible background state regardless of whether we
    // detoured it for the wireframe-mode clear above.
    if (wireClear) {
        backgroundGradientColors = savedGradient;
        backcolor = savedBack;
    }
}

// The per-object part of prepareFrame for objects [first, last): cull, fade, LOD pick, transform and emit into lane L.
// Reads the frame state prepareFrame set up (camera matrix, cull planes, prepCam) and writes only L and the objects.
void Scene::prepareObjects(PrepareLane& L, size_t first, size_t last) {
    const int32_t camCosX = prepCam[0], camSinX = prepCam[1], camCosY = prepCam[2], camSinY = prepCam[3], camCosZ = prepCam[4], camSinZ = prepCam[5];
    for (size_t oi = first; oi < last; ++oi) {
        Object* obj = objects[oi];
        if (!obj->enabled) continue;
        if (obj->triangles.empty() && obj->lodMeshes.empty()) continue;   // nothing to draw: skip the per-object setup
        // 1) Quick sphere far-cull before the expensive 8-corner AABB test.
        //    distSq to the object centre is computed unconditionally so it
        //    is also available for the fade ramps and LOD pick below,
        //    replacing the old lazy-compute block. The conservative sphere
        //    radius used is the object's longest bounding-box dimension
        //    (always >= the true bounding-sphere radius — never drops a
        //    visible object).
        uint8_t objAlpha = 255;
        const int32_t _ocx = (obj->position.x + obj->centreVolume.x) - camera->position.x;
        const int32_t _ocy = (obj->position.y + obj->centreVolume.y) - camera->position.y;
        const int32_t _ocz = (obj->position.z + obj->centreVolume.z) - camera->position.z;
        int64_t distSq = (int64_t)_ocx*_ocx + (int64_t)_ocy*_ocy + (int64_t)_ocz*_ocz;
        int32_t dist   = -1;
        {
            const int32_t maxExtent = std::max({
                obj->boundingBoxMax.x - obj->boundingBoxMin.x,
                obj->boundingBoxMax.y - obj->boundingBoxMin.y,
                obj->boundingBoxMax.z - obj->boundingBoxMin.z});
            const int64_t farCutoff = static_cast<int64_t>(camera->farPlane) + maxExtent;
            if (distSq > farCutoff * farCutoff) continue;
        }
        // 2) Object-level AABB frustum cull (all 8 corners; full rotation).
        if (cullObject(obj, camCosX, camSinX, camCosY, camSinY, camCosZ, camSinZ))
            continue;
        // 3) Per-object distance fade (two ramps, multiplied):
        //     - fadeFar > 0:   close=opaque, far=invisible (decor fade-out).
        //     - appearFar > 0: close=invisible, far=opaque (LOD impostor
        //                      that pops in at distance).
        //     Both can be combined on the same object if you want a
        //     visibility "band" — opaque only between two distances.
        //     fadeFar==0 / appearFar==0 disable the respective ramp.
        //     Distance is measured in world space from the camera to the
        //     object's centre (position + centreVolume). Beyond fadeFar
        //     OR closer than appearNear the object is skipped entirely
        //     (no transform, no per-tri work), so this is a real perf
        //     win not just a visual fade.
        // distSq is always valid here (computed above for the sphere pre-cull).
        if (obj->fadeFar > 0 || obj->appearFar > 0) {
            if (obj->fadeFar > 0) {
                const int64_t farSq = (int64_t)obj->fadeFar * obj->fadeFar;
                if (distSq >= farSq) continue; // fully past fade-out — skip
                const int64_t nearSq = (int64_t)obj->fadeNear * obj->fadeNear;
                if (distSq > nearSq && obj->fadeFar > obj->fadeNear) {
                    if (dist < 0) dist = (int32_t)sqrtf((float)distSq);
                    const int32_t span = obj->fadeFar - obj->fadeNear;
                    const int32_t over = dist - obj->fadeNear;
                    int32_t a = 255 - (over * 255) / span;
                    if (a < 0) a = 0;
                    if (a > 255) a = 255;
                    objAlpha = (uint8_t)((objAlpha * a) / 255);
                }
            }

            // Appear-in ramp (LOD impostor).
            if (obj->appearFar > 0) {
                const int64_t nearSq = (int64_t)obj->appearNear * obj->appearNear;
                if (distSq <= nearSq) continue; // still too close — skip
                const int64_t farSq = (int64_t)obj->appearFar * obj->appearFar;
                if (distSq < farSq && obj->appearFar > obj->appearNear) {
                    if (dist < 0) dist = (int32_t)sqrtf((float)distSq);
                    const int32_t span = obj->appearFar - obj->appearNear;
                    const int32_t over = dist - obj->appearNear;
                    int32_t a = (over * 255) / span;
                    if (a < 0) a = 0;
                    if (a > 255) a = 255;
                    objAlpha = (uint8_t)((objAlpha * a) / 255);
                }
                // distSq >= farSq: fully appeared, multiplier already 255.
            }

            if (objAlpha == 0) continue;
        }

        // 1c) Global LOD pick. The head Object IS LOD 0; entries in
        //     obj->lodMeshes are LOD 1, 2, ... in order. Beyond the last
        //     available LOD: cull (default) or clamp (`lodPersist`).
        //     The picked Object* contributes ONLY mesh data; the head's
        //     transform / flags / AABB / fade ramps still drive the draw.
        Object* meshSource = obj;
        if (lodScale > 0) {
            if (dist < 0 && distSq >= 0) dist = (int32_t)sqrtf((float)distSq);
            int32_t level = (dist < 0 ? 0 : dist / lodScale);
            level += (int32_t)lodBias + (int32_t)obj->lodBias;
            if (level < 0) level = 0;

            const int availableLODs = (int)obj->lodMeshes.size();
            if (level == 0) {
                meshSource = obj;
            } else if (level <= availableLODs) {
                Object* candidate = obj->lodMeshes[level - 1];
                meshSource = candidate ? candidate : obj;
            } else if (obj->lodPersist) {
                if (availableLODs > 0) {
                    Object* candidate = obj->lodMeshes[availableLODs - 1];
                    meshSource = candidate ? candidate : obj;
                }
                // else: no LOD chain at all, draw the head as-is.
            } else {
                continue; // ran out of LODs and not persisting → cull.
            }
        }

        // 2) Transform + project + per-triangle cull, push into renderQueue
        renderObject(L, obj, camCosX, camSinX, camCosY, camSinY, camCosZ, camSinZ, objAlpha, meshSource);
        ++L.drawnObjs;
    }
}

void Scene::prepareFrame() {
    if (!camera) return;
#if JET_PROFILE
    uint32_t jp0 = jet_prof_now();
#endif
    // renderEvenLines drives the frame-parity selection used by both interlaced
    // and checkerboard modes.  In interlaced mode it selects which rows to draw;
    // in checkerboard mode it selects which (x+y) pixel parity to draw.  When
    // neither mode is active we force it to false so drawTriangle's
    // `yStart = interlacedMode ? (minY + renderEvenLines) : minY` never shifts
    // the first scanline by a stray ±1.
    renderEvenLines = (renderer->interlacedMode || renderer->checkerboardMode)
                      ? (frameCounter % 2 == 0)
                      : false;
    clearBuffers();

#if MAX_PICK_QUERIES > 0
    // Reset pick results for this frame and hand the arrays to the
    // rasterizer. With FIELD_BUFFERS the rasterizer only writes one
    // parity of rows per frame, so a query landing on the "off" parity
    // would never be tested by drawTriangle's per-row pick loop. Snap
    // the y to a row this frame's field actually covers so the host's
    // requested pixel still produces a hit on roughly the right
    // location (off by one row at most). The snapped y is what gets
    // stored in PickResult.y so the caller can render their cursor on
    // the same row the renderer inspected.
    for (int i = 0; i < pickQueryCount; ++i) {
        pickResults[i] = PickResult{};
    #if FIELD_BUFFERS
        if (renderer->interlacedMode && pickQueries[i].y >= 0) {
            const int desiredParity = renderEvenLines ? 0 : 1;
            int sy = pickQueries[i].y;
            if ((sy & 1) != desiredParity) {
                // Prefer nudging up; clamp to a valid row at the bottom.
                if (sy + 1 < screenHeight) sy += 1; else if (sy > 0) sy -= 1;
            }
            pickQueries[i].y = (int16_t)sy;
        }
    #endif
    }
    renderer->pickQueries     = pickQueries;
    renderer->pickResults     = pickResults;
    renderer->pickQueryCount  = pickQueryCount;
#endif

#if LIGHTING
    if (directionalLight) directionalLight->updateViewSpaceDirection(camera);
#endif

    // Thread sky gradient and frame counter into the rasterizer so
    // WATER_REFLECT shading can sample them. Intentionally outside
    // #if LIGHTING — works with LIGHTING=0 on firmware.
    renderer->gradientColors = backgroundGradientColors;
    renderer->gradientSize   = backgroundGradientColors ? screenHeight : 0;
    renderer->frameCounter   = frameCounter;
    renderer->waterTime      = waterTime;

    int32_t camCosX, camSinX, camCosY, camSinY, camCosZ, camSinZ;
    camera->getRotationMatrix(camCosX, camSinX, camCosY, camSinY, camCosZ, camSinZ);

    // Preserve the fixed-point composition and float conversion order,
    // but perform them once per frame instead of once per visible object.
    int32_t camM00, camM01, camM02;
    int32_t camM10, camM11, camM12;
    int32_t camM20, camM21, camM22;
    {
        const int32_t cx = camCosX, sx = camSinX;
        const int32_t cy = camCosY, sy = camSinY;
        const int32_t cz = camCosZ, sz = camSinZ;
        // K = Rx * Ry
        const int32_t k00 = cy;
        const int32_t k01 = 0;
        const int32_t k02 = sy;
        const int32_t k10 = (int32_t)((int64_t)sx * sy / FIXED_POINT_SCALE);
        const int32_t k11 = cx;
        const int32_t k12 = (int32_t)(-(int64_t)sx * cy / FIXED_POINT_SCALE);
        const int32_t k20 = (int32_t)(-(int64_t)cx * sy / FIXED_POINT_SCALE);
        const int32_t k21 = sx;
        const int32_t k22 = (int32_t)((int64_t)cx * cy / FIXED_POINT_SCALE);
        // M = Rz * K
        camM00 = (int32_t)(((int64_t)cz * k00 - (int64_t)sz * k10) / FIXED_POINT_SCALE);
        camM01 = (int32_t)(((int64_t)cz * k01 - (int64_t)sz * k11) / FIXED_POINT_SCALE);
        camM02 = (int32_t)(((int64_t)cz * k02 - (int64_t)sz * k12) / FIXED_POINT_SCALE);
        camM10 = (int32_t)(((int64_t)sz * k00 + (int64_t)cz * k10) / FIXED_POINT_SCALE);
        camM11 = (int32_t)(((int64_t)sz * k01 + (int64_t)cz * k11) / FIXED_POINT_SCALE);
        camM12 = (int32_t)(((int64_t)sz * k02 + (int64_t)cz * k12) / FIXED_POINT_SCALE);
        camM20 = k20;
        camM21 = k21;
        camM22 = k22;
    }
    cameraMatrix[0] = (float)camM00 / FIXED_POINT_SCALE;
    cameraMatrix[1] = (float)camM01 / FIXED_POINT_SCALE;
    cameraMatrix[2] = (float)camM02 / FIXED_POINT_SCALE;
    cameraMatrix[3] = (float)camM10 / FIXED_POINT_SCALE;
    cameraMatrix[4] = (float)camM11 / FIXED_POINT_SCALE;
    cameraMatrix[5] = (float)camM12 / FIXED_POINT_SCALE;
    cameraMatrix[6] = (float)camM20 / FIXED_POINT_SCALE;
    cameraMatrix[7] = (float)camM21 / FIXED_POINT_SCALE;
    cameraMatrix[8] = (float)camM22 / FIXED_POINT_SCALE;

    // Frustum side-plane normal lengths for cullObject's quick sphere
    // test. fovFactor changes at runtime (boost FOV kick) so refresh per
    // frame rather than caching at init.
    {
        const float f  = camera->fovFactor;
        const float hw = (float)screenWidth  * 0.5f;
        const float hh = (float)screenHeight * 0.5f;
        cullPlaneLh = sqrtf(f * f + hw * hw);
        cullPlaneLv = sqrtf(f * f + hh * hh);
    }

    // Compute the screen row of the water/sky horizon from the camera's pitch.
    // Derivation: a world point at the water horizon (infinite Z along forward)
    // projects to sy = screenH/2 + camSinX * fovFactor / FIXED_POINT_SCALE.
    // WATER_REFLECT uses mirrorY = 2*waterlineY - y so reflections align
    // correctly for both near (low on screen) and distant (high on screen) objects.
    renderer->waterlineY = screenHeight / 2
                         + (int)(camSinX * camera->fovFactor / 1024.0f);

    bandListRows = 0;

#if TEXTURE_MAPPING
    textureQueue.clear();
#endif
    // lanes: two disjoint regions of ONE queue, so nothing is copied after the join and lane 1's triangles stay in internal RAM
    uint32_t cap = queueCap;
    auto setup_lanes = [&](uint32_t base1) {
        lane1Begin = base1;
        lanes[0].q.reset(renderQueue.data(), base1);
        lanes[0].bkt.reset(renderBuckets.data(), base1);
        lanes[0].ysp.reset(renderYSpan.data(), 2 * base1);
        lanes[1].q.reset(renderQueue.data() + base1, cap - base1);
        lanes[1].bkt.reset(renderBuckets.data() + base1, cap - base1);
        lanes[1].ysp.reset(renderYSpan.data() + 2 * base1, 2 * (cap - base1));
    };
#if TEXTURE_MAPPING
    lanes[0].textureQueue = &textureQueue; lanes[1].textureQueue = &lane1Texture; lane1Texture.clear();
#endif
    for (PrepareLane& L : lanes) { L.drawnObjs = 0; for (int q = 0; q < 32; ++q) { L.prof_cnt[q] = 0; L.prof_cyc[q] = 0; } }
    prepCam[0] = camCosX; prepCam[1] = camSinX; prepCam[2] = camCosY; prepCam[3] = camSinY; prepCam[4] = camCosZ; prepCam[5] = camSinZ;
    const size_t nObjs = objects.size();
    lastPrepareSplit = 0;
    if (prepareExecutor.start && prepareExecutor.join && nObjs > 1) {
        // split point: half of the enabled objects' vertex count (a cost proxy; the order of the lanes is the serial order)
        uint64_t total = 0;
        for (const Object* o : objects) if (o->enabled) total += o->vertices.size() + o->triangles.size();
        uint64_t acc = 0; size_t k = 0;
        const uint64_t target = (uint64_t)((double)total * (double)prepareSplitFrac);
        for (; k < nObjs; ++k) { const Object* o = objects[k]; if (o->enabled) acc += o->vertices.size() + o->triangles.size(); if (acc >= target) { ++k; break; } }
        if (k > 0 && k < nObjs) {
            lane1First = k; lane1Last = nObjs; lastPrepareSplit = k;
            // Region sizes come from each lane's recent HIGH-WATER emit count plus 20 %, not from a mean: balancing the lanes by TIME
            // makes the emitted counts lopsided (device: 75 vs 300 in one frame), and a mean-sized region overflowed 8.7 % of frames.
            // Any slack left in the buffer is split evenly, so both lanes keep headroom for a heavier frame.
            {
                uint32_t need0 = laneHigh[0] + laneHigh[0] / 3 + 16;
                uint32_t need1 = laneHigh[1] + laneHigh[1] / 3 + 16;
                if (need0 + need1 > cap) {   // shrink both proportionally; an overflow then redoes the frame serially
                    const uint64_t sum = (uint64_t)need0 + need1;
                    need0 = (uint32_t)((uint64_t)need0 * cap / sum);
                    need1 = cap - need0;
                }
                setup_lanes(need0 + (cap - need0 - need1) / 2);
            }
            const auto now = prepareExecutor.now; void* const nu = prepareExecutor.user;
            PrepareTimes& pt = lastPrepareTimes;
            pt.cost0 = acc; pt.cost1 = total - acc;
            const int64_t tp0 = now ? now(nu) : 0;
            // lane 1 writes only pt.lane1 (its own field); the caller reads it after join
            prepareExecutor.start(prepareExecutor.user, [](void* self) {
                Scene* sc = static_cast<Scene*>(self); const auto nw = sc->prepareExecutor.now; void* const u = sc->prepareExecutor.user;
                const int64_t a = nw ? nw(u) : 0;
                sc->prepareObjects(sc->lanes[1], sc->lane1First, sc->lane1Last);
                sc->lastPrepareTimes.lane1 = nw ? nw(u) - a : 0; }, this);
            prepareObjects(lanes[0], 0, k);
            const int64_t tp1 = now ? now(nu) : 0;
            prepareExecutor.join(prepareExecutor.user);
            const int64_t tp2 = now ? now(nu) : 0;
            pt.tris0 = lanes[0].q.size(); pt.tris1 = lanes[1].q.size();
#if TEXTURE_MAPPING
#error "TEXTURE_MAPPING with split prepare needs the uv queue split the same way (it is 0 in this project)"
#endif
            if (now) {
                pt.lane0 = tp1 - tp0; pt.join = tp2 - tp1; pt.merge = now(nu) - tp2;
                // nudge the split toward equal lane times (10 % of the imbalance per frame, clamped): the cost proxy does not
                // track the emit cost, and the device showed lane 1 taking 0.26..0.71 of the emitted triangles at a fixed 0.49 of the proxy
                const double l0 = (double)pt.lane0, l1 = (double)pt.lane1;
                if (l0 > 0 && l1 > 0) {
                    prepareSplitFrac += (float)(0.1 * (l1 - l0) / (l0 + l1)) * prepareSplitFrac;
                    if (prepareSplitFrac < 0.2f) prepareSplitFrac = 0.2f;
                    if (prepareSplitFrac > 0.8f) prepareSplitFrac = 0.8f;
                }
            }
        } else { setup_lanes(cap); prepareObjects(lanes[0], 0, nObjs); }
    } else {
        setup_lanes(cap);
        prepareObjects(lanes[0], 0, nObjs);
    }
    lane0Count = lanes[0].q.size(); lane1Count = lanes[1].q.size();
    // decaying high-water: rises at once for a heavier frame, falls slowly so a single light frame cannot shrink a region
    laneHigh[0] = std::max(lane0Count, laneHigh[0] - laneHigh[0] / 256);
    laneHigh[1] = std::max(lane1Count, laneHigh[1] - laneHigh[1] / 256);
    if (lanes[0].q.over || lanes[1].q.over || lanes[0].bkt.over || lanes[1].bkt.over) {
        // a region filled up: redo the whole frame serially over the entire buffer rather than lose triangles
        ++prepareOverflows;
        // the overflowing lane's count was clamped at its region, so the high-water cannot learn the real need from it: raise it here
        if (lanes[0].q.over || lanes[0].bkt.over) laneHigh[0] += laneHigh[0] / 4 + 16;
        if (lanes[1].q.over || lanes[1].bkt.over) laneHigh[1] += laneHigh[1] / 4 + 16;
        // A region filled up. Redo the frame serially over the whole buffer; if that overflows too the buffer itself is too small,
        // so grow it (counted and reported - a grown buffer can land in PSRAM, which costs 7-8 fps) and redo once more.
        for (int attempt = 0; attempt < 2; ++attempt) {
            for (PrepareLane& L : lanes) { L.drawnObjs = 0; for (int q = 0; q < 32; ++q) { L.prof_cnt[q] = 0; L.prof_cyc[q] = 0; } }
            setup_lanes(cap);
            prepareObjects(lanes[0], 0, nObjs);
            if (!lanes[0].q.over && !lanes[0].bkt.over) break;
            ++queueGrowths;
            queueCap = queueCap + queueCap / 2;
            renderQueue.resize(queueCap); renderBuckets.resize(queueCap); renderYSpan.resize(2 * queueCap);
            renderOrder.reserve(queueCap);
            cap = queueCap;
        }
        lane0Count = lanes[0].q.size(); lane1Count = 0;
    }
    const int drawnObjs = lanes[0].drawnObjs + lanes[1].drawnObjs;
#if JET_PROFILE
    for (const PrepareLane& L : lanes) for (int q = 0; q < JP_N && q < 32; ++q) { jet_prof_cnt[q] += L.prof_cnt[q]; jet_prof_cyc[q] += L.prof_cyc[q]; }
#endif
    lastFrameDrawnObjects   = drawnObjs;
    lastFrameDrawnTriangles = static_cast<int>(lane0Count + lane1Count);

#if JET_PROFILE
    { const uint32_t jp1 = jet_prof_now(); jet_prof_cyc[JP_XFORM] += jp1 - jp0; jp0 = jp1; }
#endif
    // 3) Global painter's sort. Three bands:
    //      0. noWriteZBuffer  — drawn first, so later geometry paints over
    //                           them (e.g. skyboxes).
    //      1. Normal          — main scene, back-to-front by effective Z.
    //      2. ignoreZBuffer   — drawn last, unconditionally on top.
    //
    // Within the normal band, the sort key is `avgZ - zBias * zBiasScale`.
    // Bigger zBias pulls a triangle toward the camera in the sort, so it
    // draws later than coplanar geometry without that bias. The scale has
    // to be large enough to beat within-triangle avgZ variation on
    // typical world-scale geometry (~hundreds of units) so decals reliably
    // win coplanar fights, but not so large that a biased decal draws
    // *over* geometry that's genuinely much closer (where avgZ differences
    // are thousands). Tune if the scale of the world changes significantly.
    // Bucket sort: O(N) vs O(N log N) for std::sort. Separates the three
    // draw bands with one linear pass, then counting-sorts the normal band
    // by depth in K=64 buckets (farther triangles first). Within a bucket
    // (~Z_RANGE/64 depth units) relative order is preserved (stable).
    //
    // The sort scatters 4-byte INDICES into renderOrder rather than moving
    // the RenderTri structs themselves — the queue entries stay where
    // push_back left them and rasterizeBand() draws via renderOrder. This
    // deletes what used to be a full second copy of the queue every frame.
    {
        const int N = static_cast<int>(lane0Count + lane1Count);
        const int32_t r0b = 0, r0e = (int32_t)lane0Count;                       // lane 0's region
        const int32_t r1b = (int32_t)lane1Begin, r1e = r1b + (int32_t)lane1Count; // lane 1's, after the gap
        renderOrder.resize(N);
        if (N > 1) {
            int counts[SortBucketCount] = {};
            // Keys were captured while emitting triangles, when their
            // depth and flags were already live. Both passes now stream
            // bytes instead of revisiting strided, often external-RAM data.
            for (int32_t i = r0b; i < r0e; ++i) ++counts[renderBuckets[i]];
            for (int32_t i = r1b; i < r1e; ++i) ++counts[renderBuckets[i]];
            int pos[SortBucketCount];
            pos[0] = 0;
            for (int i = 1; i < SortBucketCount; ++i)
                pos[i] = pos[i - 1] + counts[i - 1];
            // lane 0 before lane 1 within a bucket: the serial emit order, so the painter's tie-break is unchanged
            for (int32_t i = r0b; i < r0e; ++i) renderOrder[pos[renderBuckets[i]]++] = i;
            for (int32_t i = r1b; i < r1e; ++i) renderOrder[pos[renderBuckets[i]]++] = i;
        } else if (N == 1) {
            renderOrder[0] = lane0Count ? 0 : (int32_t)lane1Begin;
        }
    }
#if JET_PROFILE
    jet_prof_cyc[JP_SORT] += jet_prof_now() - jp0; ++jet_prof_cnt[JP_SORT];
#endif
}  // end prepareFrame()

void Scene::clearBand(int yMin, int yMax) {
    if (!renderer) return;
    renderer->yBandMin = yMin;
    renderer->yBandMax = yMax;
    clearBuffers();
}

void Scene::rasterizeBand(int yMin, int yMax, uint8_t* triangleFlags) {
    const int rasterized = rasterizeBandImpl(yMin, yMax, triangleFlags, nullptr, bandKeyMin, bandKeyMax, true);
    if (!triangleFlags) lastFrameRasterizedTriangles = rasterized;
}

// Thread-safe band entry (multi-core rendering): an explicit framebuffer base and depth gate instead of the
// shared setFramebuffer/setBandDepthGate state, the triangle count returned instead of written to
// lastFrameRasterizedTriangles, and no writes to the global JET_PROFILE counters. Reads only state that
// prepareFrame/buildBandLists left behind, so two threads may run it on disjoint rows at the same time.
int Scene::rasterizeBandInto(int yMin, int yMax, uint16_t* framebufferBase, int32_t keyMin, int32_t keyMax) {
    return rasterizeBandImpl(yMin, yMax, nullptr, framebufferBase, keyMin, keyMax, false);
}

int Scene::rasterizeBandImpl(int yMin, int yMax, uint8_t* triangleFlags, uint16_t* framebufferBase,
                             int32_t keyMin, int32_t keyMax, bool profile) {
#if JET_PROFILE
    const uint32_t jp0 = profile ? jet_prof_now() : 0;
#else
    (void)profile;
#endif
    // Create a thread-local copy of the rasteriser so each band worker has
    // its own yBandMin/yBandMax. Only framebuffer/zbuffer ptrs are shared;
    // writes go to non-overlapping y regions so there is no write race.
    Rasterizer bandRast = *renderer;
    bandRast.yBandMin = yMin;
    bandRast.yBandMax = yMax;
    if (framebufferBase) bandRast.setFramebuffer(framebufferBase);

    // 4) Flush. Count triangles that actually entered the rasterizer
    // (drawTriangle returned true). Triangles dropped by per-tri checks
    // inside drawTriangle (alpha=0, zero-area, near/far Z, degenerate
    // denom) return false and don't count toward the rasterized total.
    int rasterized = 0;
    // Per-band list when the caller built one for this band layout, else the
    // full painter's order with a per-triangle y-span test.
    const bool useList = bandListRows > 0 && yMin % bandListRows == 0 && yMax - yMin <= bandListRows
                         && yMin / bandListRows < MaxBandLists;
    const std::vector<int32_t>& walk = useList ? bandOrder[yMin / bandListRows] : renderOrder;
    for (const int32_t idx : walk) {
        if (!useList && (renderYSpan[2 * idx + 1] < yMin || renderYSpan[2 * idx] >= yMax)) continue;
        const RenderTri& t = renderQueue[idx];
        if (keyMin != INT32_MIN || keyMax != INT32_MAX) {   // setBandDepthGate: far / near split around an externally drawn layer
            const int32_t key = t.avgZ - static_cast<int32_t>(t.zBias) * 256;
            if (key < keyMin || key >= keyMax) continue;
        }
#if JET_FLAT_KERNEL
        if (t.flatOpaque && !bandRast.wireframeMode && !bandRast.interlacedMode && !bandRast.checkerboardMode) {
            // drawTriangleImpl's bounding box (even-aligned) and clamps, verbatim
            const int32_t x1 = t.v1.position.x, x2 = t.v2.position.x, x3 = t.v3.position.x;
            const int32_t y1 = t.v1.position.y, y2 = t.v2.position.y, y3 = t.v3.position.y;
            int32_t minX = std::min({x1, x2, x3}) & ~1, maxX = std::max({x1, x2, x3}) & ~1;
            int32_t minY = std::min({y1, y2, y3}) & ~1, maxY = std::max({y1, y2, y3}) & ~1;
#if SKIP_ZERO_AREA_TRIANGLES
            if (minX == maxX || minY == maxY) continue;
#endif
            minX = std::max<int32_t>(minX, 0);
            maxX = std::min<int32_t>(maxX, screenWidth - 1);
            minY = std::max<int32_t>(minY, yMin);
            maxY = std::min<int32_t>(maxY, std::min(yMax - 1, screenHeight - 1));
            if (minY > maxY) continue;
#if JET_PROFILE
            const uint32_t jpk = jet_prof_now();
#endif
            const int rows = bandRast.drawFlatOpaque(x1, y1, x2, y2, x3, y3, minX, maxX, minY, maxY, t.flatColor);
            if (rows >= 0) {
#if JET_PROFILE
                if (profile) {
                ++jet_prof_cnt[JP_TRI_SETUP];
                jet_prof_cnt[JP_TRI_ROWS] += (uint32_t)rows;
                jet_prof_cyc[JP_TRI_ROWS] += jet_prof_now() - jpk;
                }
#endif
                ++rasterized;
                if (triangleFlags) triangleFlags[idx] = 1;
                continue;
            }
            // huge coordinates / degenerate: fall through to the general path
        }
#endif
#if MAX_PICK_QUERIES > 0
        bandRast.currentPickObject        = t.sourceObject;
        bandRast.currentPickTriangleIndex = t.sourceTriangleIndex;
#endif
        // t.avgZ rides along as the FAST_Z depth hint: emitTri computed the
        // same three-vertex average and already culled it against near/far,
        // so drawTriangle skips both the recompute and the redundant test.
        RenderVertex a = t.v1.expand(), b = t.v2.expand(), c = t.v3.expand();
#if TEXTURE_MAPPING
        if (t.uvIndex != UINT32_MAX) {
            const TriangleUV& uv = textureQueue[t.uvIndex];
            a.uv = uv.a; b.uv = uv.b; c.uv = uv.c;
        }
#endif
        if (bandRast.drawTriangle(a, b, c, t.material,
                                   directionalLight, ambientLight,
                                   renderEvenLines,
                                   t.ignoreZBuffer, t.noWriteZBuffer,
                                   (int)t.zBias, t.objAlpha,
                                   t.brightnessPrecomputed, t.avgZ)) {
            ++rasterized;
            if (triangleFlags) triangleFlags[idx] = 1;
        }
    }
#if JET_PROFILE
    if (profile) { jet_prof_cyc[JP_BAND_WALK] += jet_prof_now() - jp0; ++jet_prof_cnt[JP_BAND_WALK]; }
#endif
    return rasterized;
}

void Scene::render(RasterExecutor executor) {
    if (!camera) return;
    prepareFrame();
    if (executor) executor(*this);
    else rasterizeBand(0, screenHeight);

    // Checkerboard reconstruction: fill in opposite-parity pixels from the
    // freshly-rendered current-parity neighbours so PostFX sees a fully-populated
    // buffer. Done as a separate pass after all triangles are drawn so that
    // painter's algorithm overdraw cannot cause repeated averaging.
    #if defined(CHECKERBOARD_RECONSTRUCTION) && CHECKERBOARD_RECONSTRUCTION
    if (renderer->checkerboardMode) {
        reconstructCheckerboard();
    }
    #endif

    #if POSTFX_ANTIALIASING
    postFX->applyFXAA(framebuffer);
    #endif

    #if POSTFX_BLOOM
    postFX->applyBloom(framebuffer);
    #endif

    #if POSTFX_CRT
    postFX->applyCRT(framebuffer);
    #endif

    #if POSTFX_PIXELATE
    postFX->applyPixelate(framebuffer);
    #endif

    #if POSTFX_CHROMATIC
    postFX->applyChromatic(framebuffer);
    #endif

    #if POSTFX_MOTION_BLUR
    postFX->applyMotionBlur(framebuffer);
    #endif

    drawSprites();

    frameCounter++;
}

void Scene::addSprite(Sprite2D* sprite) {
    if (sprite) sprites.push_back(sprite);
}

// ---------------------------------------------------------------------------
// drawSprites — called at end of render(), after PostFX, before frameCounter++
// ---------------------------------------------------------------------------
// Blends each registered Sprite2D onto the framebuffer in zOrder sequence.
// Two paths:
//   textured  — iterate source rows, colour-key skip, optional alpha blend
//   solid     — span fill with material->color, optional alpha blend
//
// Alpha compositing retains exact per-channel /255 rounding. Shared span
// helpers also implement the byte-swapped full-resolution display pass.
// ---------------------------------------------------------------------------
void Scene::drawSprites() {
    // On HALF_WIDTH_BUFFERS builds, sprites are composited at full output
    // resolution during Display::pushFrame() scanout instead — writing them
    // into the half-width framebuffer here would halve their horizontal
    // resolution and use the wrong stride.
#if HALF_WIDTH_BUFFERS
    return;
#else
    if (sprites.empty()) return;

    // Sort by zOrder each frame (list is typically tiny — insertion sort
    // territory, but std::stable_sort keeps it simple and correct).
    std::stable_sort(sprites.begin(), sprites.end(),
        [](const Sprite2D* a, const Sprite2D* b){ return a->zOrder < b->zOrder; });

    for (Sprite2D* sp : sprites) {
        if (!sp || !sp->enabled || !sp->material) continue;
        const uint8_t matAlpha = sp->material->alpha;
        // Combined alpha: 0 = invisible, 255 = opaque.
        const int combined = ((int)sp->alpha * (int)matAlpha) / 255;
        if (combined == 0) continue;

        const Texture* tex = sp->material->diffuseMap;

        // Clip dest rect to framebuffer bounds.
        const int srcW = tex ? tex->width  : sp->width;
        const int srcH = tex ? tex->height : sp->height;
        if (srcW <= 0 || srcH <= 0) continue;
        const int dstW = srcW * sp->scale;
        const int dstH = srcH * sp->scale;

        const int x0 = sp->x, y0 = sp->y;
        const int x1 = x0 + dstW, y1 = y0 + dstH;
        // Clipped dest rect
        const int dx0 = (x0 < 0) ? 0 : x0;
        const int dy0 = (y0 < 0) ? 0 : y0;
        const int dx1 = (x1 > screenWidth)  ? screenWidth  : x1;
        const int dy1 = (y1 > screenHeight) ? screenHeight : y1;
        if (dx0 >= dx1 || dy0 >= dy1) continue;

        // Source start offsets for the unscaled (scale==1) path.
        const int sx0 = dx0 - x0;
        const int sy0 = dy0 - y0;

        if (tex && sp->scale > 1) {
            // ---- Nearest-neighbour upscale textured blit (scale > 1) ---------
            // fp8 step: advances source coord by (1/scale) per output pixel.
            const int xStep = (srcW << 8) / dstW;
            const int yStep = (srcH << 8) / dstH;
            const bool colorKey = tex->hasAlpha;
            const uint16_t keyColor = tex->alphaColor;
            const uint16_t* src = tex->data;
            int sf_y = (dy0 - y0) * yStep;
            for (int dy = dy0; dy < dy1; ++dy, sf_y += yStep) {
                const int sy  = sf_y >> 8;   // nearest-neighbour: integer texel
                uint16_t* dstRow = framebuffer + dy * screenWidth + dx0;
                int sf_x = (dx0 - x0) * xStep;
                const int w = dx1 - dx0;
                blendRGB565ScaledSpan(dstRow, src + sy * srcW, w, sf_x, xStep,
                    (uint8_t)combined,
                    sp->blendMode == BlendMode::BLEND_ADD ? RGB565BlendMode::Add : RGB565BlendMode::Alpha255,
                    colorKey ? BlendColorKey : 0, keyColor);
            }
        } else if (tex) {
            // ---- Textured sprite blit ----------------------------------------
            const bool colorKey = tex->hasAlpha;
            const uint16_t keyColor = tex->alphaColor;
            const uint16_t* src = tex->data;
            for (int dy = dy0; dy < dy1; ++dy) {
                const int sy = sy0 + (dy - dy0);
                blendRGB565Span(framebuffer + dy * screenWidth + dx0,
                    src + sy * tex->width + sx0, dx1 - dx0, 0, (uint8_t)combined,
                    sp->blendMode == BlendMode::BLEND_ADD ? RGB565BlendMode::Add : RGB565BlendMode::Alpha255,
                    colorKey ? BlendColorKey : 0, keyColor);
            }
            continue;
        } else {
            // ---- Solid rectangle fill ----------------------------------------
            const uint16_t col = sp->material->color;
            for (int dy = dy0; dy < dy1; ++dy)
                blendRGB565Span(framebuffer + dy * screenWidth + dx0, nullptr,
                    dx1 - dx0, col, (uint8_t)combined,
                    sp->blendMode == BlendMode::BLEND_ADD ? RGB565BlendMode::Add : RGB565BlendMode::Alpha255);
            continue;
        }
    }
#endif // !HALF_WIDTH_BUFFERS
}

void Scene::getStatistics(int& objectCount, int& triangleCount, int& vertexCount) {
    objectCount = static_cast<int>(objects.size());
    triangleCount = 0;
    vertexCount = 0;

    for (const auto& obj : objects) {
        if (!obj->enabled) continue;
        triangleCount += static_cast<int>(obj->triangles.size());
        vertexCount += static_cast<int>(obj->vertices.size());
    }
}

void PERF_CRITICAL Scene::renderObject(PrepareLane& L, Object* obj,
                                     int32_t camCosX, int32_t camSinX,
                                     int32_t camCosY, int32_t camSinY,
                                     int32_t camCosZ, int32_t camSinZ,
                                     uint8_t objAlpha,
                                     Object* meshSource) {
#if JET_PROFILE
    const uint32_t jpo0 = jet_prof_now();
#endif
    // meshSource decouples "which mesh do we rasterise" from "where / how
    // does the object live in the world". Defaults to obj itself, so the
    // non-LOD path is unchanged. When the global LOD system picks a
    // reduced-detail mesh, that Object* is passed in here while obj keeps
    // ownership of the transform, flags, AABB and fade state.
    if (!meshSource) meshSource = obj;

    // Reusable scratch buffers — kept across calls so we don't pay for a
    // heap alloc per object per frame. Renderer is single-threaded
    // (one render task), so plain static is fine here. PipelineVertex keeps
    // only transformed attributes; mesh UVs are fetched for visible textured
    // faces below. The loop writes every live field, with no upfront copy.
    std::vector<PipelineVertex>& transformedVertices = L.transformedVertices;   // per lane (was a function static: one render task only)
    LaneArena<RenderTri>& renderQueue = L.q;
    LaneArena<uint8_t>& renderBuckets = L.bkt;
    LaneArena<int16_t>& renderYSpan = L.ysp;
#if TEXTURE_MAPPING
    std::vector<TriangleUV>& textureQueue = *L.textureQueue;
#endif
    const size_t vertCount = meshSource->vertices.size();
    transformedVertices.resize(vertCount);

    Vector3 camPos(camera->position);
    #if FLOAT_CAMERA_ANGLES
    Vector3_f camRotF(camera->rotation);
    #endif
    Vector3 objPos(obj->position);

    float   fovFactor = camera->fovFactor;
    bool isBillboard = obj->isBillboard;
    CullingMode cullingMode = obj->cullingMode;
    bool ignoreZBuffer = obj->ignoreZBuffer;
    bool noWriteZBuffer = obj->noWriteZBuffer;

    // Hoist object rotation trig out of the per-vertex loop — these are
    // constant for every vertex on the object. Also short-circuit the entire
    // rotation block when the object has zero rotation (true for most static
    // scenery), saving 12 mul + 6 div + 9 add per vertex.
    const bool objHasRotation = !isBillboard &&
        (obj->rotation.x != 0 || obj->rotation.y != 0 || obj->rotation.z != 0 || obj->rotationFloat);

    // Composed object rotation matrix (Rz * Ry * Rx, since the previous
    // per-vertex code applied X→Y→Z). Storing all 9 entries at
    // FIXED_POINT_SCALE scale lets the per-vertex transform collapse from
    // three cascaded rotations (12 muls + 12 div per vertex) into a single
    // 3×3 mat-vec (9 muls + 3 div), with the same again saved on the
    // normal when LIGHTING is on. The trig product entries are pre-divided
    // by FIXED_POINT_SCALE so the entire matrix lives at one consistent
    // scale.
    int32_t objM00=FIXED_POINT_SCALE, objM01=0, objM02=0;
    int32_t objM10=0, objM11=FIXED_POINT_SCALE, objM12=0;
    int32_t objM20=0, objM21=0, objM22=FIXED_POINT_SCALE;
    // Float counterparts pre-divided by FIXED_POINT_SCALE. Per-vertex mat-vecs
    // use hardware-FPU fmul-add instead of int64_t multiply+shift. Defaults
    // are identity; vertex loop guards on objHasRotation so they're never
    // consumed when false.
    float fObjM00=1.0f, fObjM01=0.0f, fObjM02=0.0f;
    float fObjM10=0.0f, fObjM11=1.0f, fObjM12=0.0f;
    float fObjM20=0.0f, fObjM21=0.0f, fObjM22=1.0f;
    if (objHasRotation) {
#if FLOAT_CAMERA_ANGLES
        int32_t cx, sx, cy, sy, cz, sz;
        if (obj->rotationFloat) {
            cx = (int32_t)(lookupCos(obj->rotFx)*FIXED_POINT_SCALE);
            sx = (int32_t)(lookupSin(obj->rotFx)*FIXED_POINT_SCALE);
            cy = (int32_t)(lookupCos(obj->rotFy)*FIXED_POINT_SCALE);
            sy = (int32_t)(lookupSin(obj->rotFy)*FIXED_POINT_SCALE);
            cz = (int32_t)(lookupCos(obj->rotFz)*FIXED_POINT_SCALE);
            sz = (int32_t)(lookupSin(obj->rotFz)*FIXED_POINT_SCALE);
        } else
#endif
        {
        const int32_t cx_ = lookupCosI(obj->rotation.x); cx = cx_;
        const int32_t sx_ = lookupSinI(obj->rotation.x); sx = sx_;
        const int32_t cy_ = lookupCosI(obj->rotation.y); cy = cy_;
        const int32_t sy_ = lookupSinI(obj->rotation.y); sy = sy_;
        const int32_t cz_ = lookupCosI(obj->rotation.z); cz = cz_;
        const int32_t sz_ = lookupSinI(obj->rotation.z); sz = sz_;
        }
        // K = Ry * Rx (at FPS scale; trig-product entries divided once).
        const int32_t k00 = cy;
        const int32_t k01 = (int32_t)((int64_t)sy * sx / FIXED_POINT_SCALE);
        const int32_t k02 = (int32_t)((int64_t)sy * cx / FIXED_POINT_SCALE);
        const int32_t k10 = 0;
        const int32_t k11 = cx;
        const int32_t k12 = -sx;
        const int32_t k20 = -sy;
        const int32_t k21 = (int32_t)((int64_t)cy * sx / FIXED_POINT_SCALE);
        const int32_t k22 = (int32_t)((int64_t)cy * cx / FIXED_POINT_SCALE);
        // M = Rz * K (at FPS scale).
        objM00 = (int32_t)(((int64_t)cz * k00 - (int64_t)sz * k10) / FIXED_POINT_SCALE);
        objM01 = (int32_t)(((int64_t)cz * k01 - (int64_t)sz * k11) / FIXED_POINT_SCALE);
        objM02 = (int32_t)(((int64_t)cz * k02 - (int64_t)sz * k12) / FIXED_POINT_SCALE);
        objM10 = (int32_t)(((int64_t)sz * k00 + (int64_t)cz * k10) / FIXED_POINT_SCALE);
        objM11 = (int32_t)(((int64_t)sz * k01 + (int64_t)cz * k11) / FIXED_POINT_SCALE);
        objM12 = (int32_t)(((int64_t)sz * k02 + (int64_t)cz * k12) / FIXED_POINT_SCALE);
        objM20 = k20;
        objM21 = k21;
        objM22 = k22;
        fObjM00=(float)objM00/FIXED_POINT_SCALE; fObjM01=(float)objM01/FIXED_POINT_SCALE; fObjM02=(float)objM02/FIXED_POINT_SCALE;
        fObjM10=(float)objM10/FIXED_POINT_SCALE; fObjM11=(float)objM11/FIXED_POINT_SCALE; fObjM12=(float)objM12/FIXED_POINT_SCALE;
        fObjM20=(float)objM20/FIXED_POINT_SCALE; fObjM21=(float)objM21/FIXED_POINT_SCALE; fObjM22=(float)objM22/FIXED_POINT_SCALE;
    }

    // Composed camera matrix is shared by every object in this frame.
    const float fCamM00=cameraMatrix[0], fCamM01=cameraMatrix[1], fCamM02=cameraMatrix[2];
    const float fCamM10=cameraMatrix[3], fCamM11=cameraMatrix[4], fCamM12=cameraMatrix[5];
    const float fCamM20=cameraMatrix[6], fCamM21=cameraMatrix[7], fCamM22=cameraMatrix[8];

    // Combined object→camera transform, composed ONCE per object:
    //   p_cam = Cam · (Obj·p + objPos − camPos) = (Cam·Obj)·p + Cam·(objPos − camPos)
    // so the per-vertex work collapses from two 3×3 mat-vecs plus two
    // vector adds down to a single mat-vec with the translation folded
    // into the accumulate. fM is also the combined rotation for normals
    // (both factors are pure rotations; translation is excluded there).
    // For unrotated objects (most scenery) fM is just the camera matrix.
    float fM00, fM01, fM02, fM10, fM11, fM12, fM20, fM21, fM22;
    if (objHasRotation) {
        fM00 = fCamM00*fObjM00 + fCamM01*fObjM10 + fCamM02*fObjM20;
        fM01 = fCamM00*fObjM01 + fCamM01*fObjM11 + fCamM02*fObjM21;
        fM02 = fCamM00*fObjM02 + fCamM01*fObjM12 + fCamM02*fObjM22;
        fM10 = fCamM10*fObjM00 + fCamM11*fObjM10 + fCamM12*fObjM20;
        fM11 = fCamM10*fObjM01 + fCamM11*fObjM11 + fCamM12*fObjM21;
        fM12 = fCamM10*fObjM02 + fCamM11*fObjM12 + fCamM12*fObjM22;
        fM20 = fCamM20*fObjM00 + fCamM21*fObjM10 + fCamM22*fObjM20;
        fM21 = fCamM20*fObjM01 + fCamM21*fObjM11 + fCamM22*fObjM21;
        fM22 = fCamM20*fObjM02 + fCamM21*fObjM12 + fCamM22*fObjM22;
    } else {
        fM00 = fCamM00; fM01 = fCamM01; fM02 = fCamM02;
        fM10 = fCamM10; fM11 = fCamM11; fM12 = fCamM12;
        fM20 = fCamM20; fM21 = fCamM21; fM22 = fCamM22;
    }
    const int32_t iDx = objPos.x - camPos.x, iDy = objPos.y - camPos.y, iDz = objPos.z - camPos.z;
    const float fDx = (float)iDx;
    const float fDy = (float)iDy;
    const float fDz = (float)iDz;
    const float fTx = fCamM00*fDx + fCamM01*fDy + fCamM02*fDz;
    const float fTy = fCamM10*fDx + fCamM11*fDy + fCamM12*fDz;
    const float fTz = fCamM20*fDx + fCamM21*fDy + fCamM22*fDz;

#if LIGHTING
    // Object-local lighting precompute eligibility.
    //
    // The view-space normal transform exists only so that the rasterizer
    // can evaluate the view-facing specular term (`N.z < 0` in
    // jetShadeBrightness). For objects whose materials are ALL non-
    // specular (`material->specular == 0`), there is no consumer of the
    // view-space frame — diffuse Lambert is rotation-invariant — so we
    // can skip the per-vertex normal transform entirely and instead
    // transform the world-space light direction into the object's local
    // space ONCE per object, then dot it against the untransformed
    // vertex normal to get brightness. That brightness is cached on the
    // vertex and read straight back out by drawTriangle, which also
    // skips its own jetShadeBrightness call when the flag is set.
    //
    // On the ESP32 with FLAT shading + non-specular materials being the
    // common case (track, ground, parapets, hulls), this deletes 9 muls
    // + 3 shifts per vertex plus the dot/clamp/square cycle from the
    // renderer's per-triangle path — replaced by a single 9-mul + 3-shift
    // light-direction transform amortized over the whole object.
    bool objectLocalLight = false;
    Vector3 objLightDir = {0, 0, 0};
    uint16_t objLightIntensity = 0;
    uint8_t  objDiffuseCoef = 255;
    if (directionalLight && !meshSource->triangles.empty()) {
        bool allNonSpecular = true;                  // hint 1/2: known, skip the walk; hint 0: scan below
        // Per-triangle material walk. Cheap — a handful of byte loads
        // per face — and exits early on the first specular material we
        // hit so specular-heavy objects (vehicles with shiny paint)
        // skip the rest of the scan. If we ever start caching this on
        // the Object itself we can drop the walk entirely; for now the
        // overhead is well below the savings on qualifying objects.
        //
        // Also disqualifies PHONG materials: PHONG interpolates per-
        // pixel from the vertex normals against directionalLight->lightDir
        // (view-space) and there's no place to feed cached brightness
        // back in. FLAT and GOURAUD both consume a per-triangle/vertex
        // scalar brightness so they slot the cache in cleanly.
        if (obj->lightHint == 0)
        for (const auto& tri : meshSource->triangles) {
            if (!tri.material) continue;
            if (tri.material->specular != 0 ||
                tri.material->shadingMode == ShadingMode::PHONG) {
                allNonSpecular = false;
                break;
            }
        }
        if (allNonSpecular) {
            objectLocalLight = true;
            objLightIntensity = directionalLight->intensity;
            // Diffuse coef varies per triangle; we use the first triangle's
            // material as a representative. If diffuse coefs were ever
            // mixed across an object's faces, GOURAUD-style midtones
            // would shift slightly compared to the renderer's path. In
            // practice diffuse is a per-shading-style constant on every
            // material in this project (255 default).
            const Material* m0 = meshSource->triangles[0].material;
            if (m0) objDiffuseCoef = m0->diffuse;

            // Transform worldLightDir into object-local space. With M
            // composed as Rz*Ry*Rx (the same axis order applied to
            // positions and normals above) and orthonormal, the inverse
            // is the transpose: rows of M^T are columns of M, so we
            // dot worldLightDir with the COLUMNS of objM. For identity
            // object rotation we just use worldLightDir directly.
            const Vector3& Lw = directionalLight->worldLightDir;
            if (objHasRotation) {
                objLightDir.assign(
                    (int32_t)(((int64_t)Lw.x * objM00 + (int64_t)Lw.y * objM10 + (int64_t)Lw.z * objM20) / FIXED_POINT_SCALE),
                    (int32_t)(((int64_t)Lw.x * objM01 + (int64_t)Lw.y * objM11 + (int64_t)Lw.z * objM21) / FIXED_POINT_SCALE),
                    (int32_t)(((int64_t)Lw.x * objM02 + (int64_t)Lw.y * objM12 + (int64_t)Lw.z * objM22) / FIXED_POINT_SCALE));
            } else {
                objLightDir = Lw;
            }
        }
    }
#endif

    // Share exactly the same transform/rounding with the clipping slow path.
    // Camera Z already lives in PipelineVertex. Recompute camera X/Y only
    // for straddling faces instead of writing a second array for every vertex.
    auto cameraPosition = [&](const Vector3& sourcePosition) -> Vector3 {
        Vector3 pos(sourcePosition);

        // Y-axis (cylindrical) billboard: pre-rotate the vertex offset
        // by +camRotY around Y so the camera's Y rotation below cancels
        // it out exactly, leaving the offset's X axis mapped to
        // camera-right and Y axis to world-Y. The billboard's world-
        // space POSITION (objPos) goes through the normal camera
        // transform unchanged, so it occupies real 3D space and
        // distance-fades / sorts correctly; only its local orientation
        // is locked to face the camera in yaw. Pitch/roll still apply
        // through the camera transform so the billboard appears
        // tilted exactly as a vertical real object would.
        if (isBillboard) {
            pos.assign((pos.x * camCosY - pos.z * camSinY) / FIXED_POINT_SCALE,
                        pos.y,
                       (pos.x * camSinY + pos.z * camCosY) / FIXED_POINT_SCALE);
        }

        // Object rotation, world translation and camera rotation in ONE
        // 3×3 mat-vec with the translation folded into the accumulate:
        // p_cam = fM·p + fT (see the composition above). Billboards enter
        // here with their yaw pre-rotation already applied and fM equal to
        // the bare camera matrix, which reproduces the old pipeline order
        // exactly.
        if (!objHasRotation && !isBillboard) {
            // Unrotated object: add the world offset in EXACT integer arithmetic
            // before the float camera rotation. Two objects that share a world
            // vertex (adjacent terrain strips: local z = 100 in one, 0 in the
            // next, positions 100 apart) then feed bit-identical floats into the
            // same matrix and truncate to the same camera-space integers. With
            // the translation folded in as a float (fT) the two paths rounded
            // differently by up to one unit, opening sub-pixel cracks along
            // every seam (strings of single background pixels on the glass).
            const float fpx = (float)(pos.x + iDx), fpy = (float)(pos.y + iDy), fpz = (float)(pos.z + iDz);
            pos.assign(
                (int32_t)(fpx * fM00 + fpy * fM01 + fpz * fM02),
                (int32_t)(fpx * fM10 + fpy * fM11 + fpz * fM12),
                (int32_t)(fpx * fM20 + fpy * fM21 + fpz * fM22));
        } else {
            const float fpx = (float)pos.x, fpy = (float)pos.y, fpz = (float)pos.z;
            pos.assign(
                (int32_t)(fpx * fM00 + fpy * fM01 + fpz * fM02 + fTx),
                (int32_t)(fpx * fM10 + fpy * fM11 + fpz * fM12 + fTy),
                (int32_t)(fpx * fM20 + fpy * fM21 + fpz * fM22 + fTz));
        }
        if (pos.z == 0) pos.z = 1;
        return pos;
    };

    // Transform vertices and normals, writing only live projected attributes.
    const Vector3* packedPositions = meshSource->cachedPositions();
#if JET_PROFILE
    L.prof_cnt[JP_XFORM] += (uint32_t)vertCount;
    const uint32_t jpv0 = jet_prof_now();
    L.prof_cyc[JP_OBJ] += jpv0 - jpo0;
#endif
    for (size_t vi = 0; vi < vertCount; ++vi) {
        const Object::Vertex& srcVert = meshSource->vertices[vi];
        PipelineVertex& dst = transformedVertices[vi];
        const Vector3 pos = cameraPosition(packedPositions ? packedPositions[vi] : srcVert.position);
#if LIGHTING
        Vector3 normal(srcVert.normal);
        // Normals use the combined ROTATION only — no translation. The
        // object-local-light path skips the transform entirely and shades
        // from the untransformed mesh-local normal below.
        if (!objectLocalLight) {
            const float fnx = (float)normal.x, fny = (float)normal.y, fnz = (float)normal.z;
            normal.assign(
                (int32_t)(fnx * fM00 + fny * fM01 + fnz * fM02),
                (int32_t)(fnx * fM10 + fny * fM11 + fnz * fM12),
                (int32_t)(fnx * fM20 + fny * fM21 + fnz * fM22));
        }
#endif

        // Perspective projection — float fovFactor lets us use a reciprocal
        // multiply instead of 64-bit integer divide, leveraging the hardware
        // FPU on ESP32-S3/P4 (64-bit div is software-emulated on those cores).
#if defined(__XTENSA__) && defined(ESP_PLATFORM)
        // The LX7 FPU has no divide: 1/z through __divsf3 in ROM costs ~100
        // cycles per vertex. recip0.s seed + two Newton steps gives full
        // float precision in ~10 cycles.
        float invZ;
        {
            const float fz = (float)pos.z;
            float r;
            __asm__("recip0.s %0, %1" : "=f"(r) : "f"(fz));
            r = r * (2.0f - fz * r);
            r = r * (2.0f - fz * r);
            invZ = fovFactor * r;
        }
#else
        const float invZ = fovFactor / (float)pos.z;
#endif
        dst.position.x = (int32_t)(pos.x * invZ) + screenWidth / 2;
        dst.position.y = screenHeight / 2 - (int32_t)(pos.y * invZ);
        dst.position.z = pos.z;

#if LIGHTING
        // Store transformed normal (only consumed by the lit shading paths).
        dst.normal = normal;
        // Object-local-light path: precompute the per-vertex Lambert
        // brightness here using the mesh-local normal + object-local
        // light direction. drawTriangle reads this back and skips its
        // own jetShadeBrightness call. For FLAT triangles all three
        // verts of a face share the same normal (computeFlatNormals
        // stamps identical normals) so the per-triangle FLAT path can
        // pick any one — it uses v1 already.
        dst.lambertBrightness = (objectLocalLight && obj->lightHint != 2)
            ? sceneLambertDiffuse(normal, objLightDir, objLightIntensity, objDiffuseCoef)
            : 0;
#endif
    }

#if JET_PROFILE
    L.prof_cyc[JP_VERTS] += jet_prof_now() - jpv0;
#endif
#if SORT_TRIANGLES
    // Sort the triangles by depth. Only intra-bucket order of the global
    // painter's bucket sort depends on this (buckets are stable by insertion);
    // Object::sortTriangles = false skips it for meshes whose triangles never
    // need intra-bucket ordering (measured 2026-09-12 on an ESP32-S3: 0.7 ms
    // per frame for a 65-object heightfield). Objects in the ignoreZBuffer
    // overlay band rely on it for their own back-to-front order.
    if (obj->sortTriangles)
    std::sort(meshSource->triangles.begin(), meshSource->triangles.end(), [&](const Object::Triangle& a, const Object::Triangle& b) {
        const auto& v1 = transformedVertices[a.v1];
        const auto& v2 = transformedVertices[a.v2];
        const auto& v3 = transformedVertices[a.v3];
        int32_t z1 = v1.position.z;
        int32_t z2 = v2.position.z;
        int32_t z3 = v3.position.z;
        return (z1 + z2 + z3) / 3 > (transformedVertices[b.v1].position.z + transformedVertices[b.v2].position.z + transformedVertices[b.v3].position.z) / 3;
    });
#endif

    // ------------------------------------------------------------------
    // Near-plane clipping helpers
    // ------------------------------------------------------------------
    // Without these, a triangle with even one vertex behind the near plane
    // is discarded whole (projected x/y would be garbage for that vertex),
    // producing visible holes in geometry right under the camera. We clip
    // straddling triangles to z==nearPlane and project the resulting new
    // vertices fresh, with UVs/normals interpolated along each clipped edge.
    const int32_t nz = camera->nearPlane;

    auto lerpI32 = [](int32_t a, int32_t b, int32_t tFixed) -> int32_t {
        return a + (int32_t)(((int64_t)tFixed * (b - a)) / FIXED_POINT_SCALE);
    };

    // Given a camera-space position, produce a RenderVertex with screen-
    // space x/y, camera-space z kept in position.z, and the provided normal
    // and uv (both interpolated upstream in camera / texture space). The
    // normal/uv/brightness parameters are only stored when the configured
    // pipeline carries those fields.
    auto projectVertex = [&](const Vector3& cam,
                             const Vector3& normalCamSpace,
                             const Vector2& uv,
                             uint16_t lambertBrightness = 0) -> RenderVertex {
        RenderVertex v;
        const float invZ = (cam.z != 0) ? fovFactor / (float)cam.z : 0.0f;
        v.position.x = (int32_t)(cam.x * invZ) + screenWidth / 2;
        v.position.y = screenHeight / 2 - (int32_t)(cam.y * invZ);
        v.position.z = cam.z;
#if LIGHTING
        v.normal = normalCamSpace;
        v.lambertBrightness = lambertBrightness;
#else
        (void)normalCamSpace; (void)lambertBrightness;
#endif
#if TEXTURE_MAPPING
        v.uv = uv;
#else
        (void)uv;
#endif
        return v;
    };

    // Clip edge from A (behind near plane) → B (in front). Returns vertex on
    // the near plane with all attributes interpolated.
    auto clipEdge = [&](const RenderVertex& A, const RenderVertex& B,
                        const Vector3& camA, const Vector3& camB) -> RenderVertex {
        int32_t dz = camB.z - camA.z;
        if (dz == 0) dz = 1;
        int32_t t = (int32_t)(((int64_t)(nz - camA.z) * FIXED_POINT_SCALE) / dz);
        Vector3 camNew(lerpI32(camA.x, camB.x, t),
                       lerpI32(camA.y, camB.y, t),
                       nz);
#if LIGHTING
        Vector3 n(lerpI32(A.normal.x, B.normal.x, t),
                  lerpI32(A.normal.y, B.normal.y, t),
                  lerpI32(A.normal.z, B.normal.z, t));
        // Interpolate the precomputed Lambert brightness along the clipped
        // edge too. Without this, clipped vertices land in projectVertex
        // with the default lambertBrightness=0 and the brightnessPrecomputed
        // path downstream reads pure black — for FLAT that turns the whole
        // face black whenever v1 happens to be the clipped vertex; for
        // GOURAUD it gradient-blends toward black at the clip seam,
        // producing the dark wedge artefacts visible on triangles whose
        // vertices straddle the near plane.
        const uint16_t lb = (uint16_t)(A.lambertBrightness +
            (int32_t)(((int64_t)t * (int32_t)(B.lambertBrightness - A.lambertBrightness)) / FIXED_POINT_SCALE));
#else
        Vector3 n(0, 0, 0);
        const uint16_t lb = 0;
#endif
#if TEXTURE_MAPPING
        Vector2 uv(lerpI32(A.uv.x, B.uv.x, t),
                   lerpI32(A.uv.y, B.uv.y, t));
#else
        Vector2 uv(0, 0);
#endif
        return projectVertex(camNew, n, uv, lb);
    };

    // Emit a projected triangle into renderQueue (does screen-bounds + backface
    // cull). Reused by both the fast path and the clipped path.
    auto emitTri = [&](const auto& a,
                       const auto& b,
                       const auto& c,
                       Material* mat
#if TEXTURE_MAPPING
                       , const Vector2* uvA, const Vector2* uvB, const Vector2* uvC
#endif
#if MAX_PICK_QUERIES > 0
                       , int32_t srcTriIdx
#endif
                       ) {
        // Per-triangle near/far cull on the average camera-space Z.
        // Object cull rejects entirely-outside boxes; this catches the
        // remaining far-plane tris on objects that straddle it (large
        // ground tiles, big cliff faces). We do this BEFORE the off-
        // screen XY tests + shoelace + queue-push so a doomed tri pays
        // none of those costs (and never enters the painter's-sort).
        // depthFogFar == farPlane in the current build, so this also
        // subsumes the depth-fog alpha=0 early-out that drawTriangle
        // would have done after a full setup.
#if JET_PROFILE
        struct JpEmitScope { uint32_t t0; uint32_t* cyc; uint32_t* cnt; ~JpEmitScope() { cyc[JP_EMIT] += jet_prof_now() - t0; ++cnt[JP_EMIT]; } } jpe{jet_prof_now(), L.prof_cyc, L.prof_cnt};
#endif
        const int32_t avgZ = (a.position.z + b.position.z + c.position.z) / 3;
        if (avgZ > camera->farPlane || avgZ < camera->nearPlane) return;

        if (a.position.x < 0 && b.position.x < 0 && c.position.x < 0) return;
        if (a.position.x > screenWidth && b.position.x > screenWidth && c.position.x > screenWidth) return;
        if (a.position.y < 0 && b.position.y < 0 && c.position.y < 0) return;
        if (a.position.y > screenHeight && b.position.y > screenHeight && c.position.y > screenHeight) return;

        // 64-bit shoelace: projected coords from a near-plane-clipped vertex
        // can be tens of thousands of units, which would overflow a 32-bit
        // signed product and accidentally flip the backface-cull sign. That
        // was causing large floor triangles near the camera to vanish
        // (typically in the bottom-left quadrant where cam.x/cam.y are most
        // negative).
        int64_t shoelaceArea = (int64_t)a.position.x * (b.position.y - c.position.y) +
                               (int64_t)b.position.x * (c.position.y - a.position.y) +
                               (int64_t)c.position.x * (a.position.y - b.position.y);

        bool shouldCull = false;
        switch (cullingMode) {
            case CullingMode::CULL_BACKFACES: shouldCull = (shoelaceArea <= 0); break;
            case CullingMode::CULL_FRONTFACES: shouldCull = (shoelaceArea >= 0); break;
            case CullingMode::NO_CULLING: break;
        }
        if (shouldCull) return;
#if JET_CULL_SLIVERS
        // Hairline triangles: a face seen almost edge-on projects to a long
        // sliver a fraction of a pixel thick; point sampling paints it at full
        // intensity wherever an integer x falls inside, so a lit quarter-pixel
        // crest facet drew a dotted bright line across the shadowed slope 18
        // units behind it (2026-09-13, esp-flightscene2: area 3.5 px^2 over
        // 31 px). Dropping such triangles punched the same dotted line in the
        // sky colour along every ridge silhouette, and blending them by
        // coverage did the same more faintly: at a silhouette the crest
        // facet really owns those pixel centres. What works is DEPTH ORDER:
        // a sub-pixel triangle is sorted jetSliverPushZ farther back, so any
        // neighbouring face within that distance paints over it while a
        // distant background behind a silhouette does not. Only below
        // jetSliverPushMaxThickness (0.5 px): a 0.5-1 px crest facet paints a
        // nearly continuous one-pixel line, which is what a lit crest looks
        // like; pushing it behind a slope that covers only some of its pixels
        // turned that line into dots. Thickness = 2*area/L for the longest
        // edge L; below jetSliverMinThickness the triangle is dropped
        // (degenerate, or a back face the integer shoelace flipped). Cheap
        // gate first: 2*area >= w+h >= L means thickness >= 1, which is
        // nearly every triangle.
        int32_t sliverPush = 0;
        {
            const int32_t minx = std::min({a.position.x, b.position.x, c.position.x});
            const int32_t maxx = std::max({a.position.x, b.position.x, c.position.x});
            const int32_t miny = std::min({a.position.y, b.position.y, c.position.y});
            const int32_t maxy = std::max({a.position.y, b.position.y, c.position.y});
            const int64_t a2 = shoelaceArea < 0 ? -shoelaceArea : shoelaceArea;
            if (a2 < (int64_t)(maxx - minx) + (maxy - miny)) {
                const int64_t dx1 = b.position.x - a.position.x, dy1 = b.position.y - a.position.y;
                const int64_t dx2 = c.position.x - b.position.x, dy2 = c.position.y - b.position.y;
                const int64_t dx3 = a.position.x - c.position.x, dy3 = a.position.y - c.position.y;
                int64_t l2 = dx1 * dx1 + dy1 * dy1;
                const int64_t l2b = dx2 * dx2 + dy2 * dy2, l2c = dx3 * dx3 + dy3 * dy3;
                if (l2b > l2) l2 = l2b;
                if (l2c > l2) l2 = l2c;
                if (a2 * a2 < l2) {                                   // thinner than one pixel
                    const float th = (float)a2 / sqrtf((float)l2);
                    if (th < jetSliverMinThickness) return;
                    if (th < jetSliverPushMaxThickness) sliverPush = jetSliverPushZ;
                }
            }
        }
#endif
#if JET_PROFILE
        const uint32_t jpe1 = jet_prof_now(); L.prof_cyc[JP_EMIT_CULL] += jpe1 - jpe.t0;
#endif

        RenderTri& rt = renderQueue.emplace_back();   // built in place in this lane's region; overflow returns a sink and is counted
        const bool reverse = cullingMode == CullingMode::NO_CULLING && shoelaceArea < 0;
        if (reverse) {
            rt.v1.assign(c); rt.v2.assign(b); rt.v3.assign(a);
        } else {
            rt.v1.assign(a); rt.v2.assign(b); rt.v3.assign(c);
        }
#if TEXTURE_MAPPING
        rt.uvIndex = UINT32_MAX;
        // Fetch mesh UVs only after culling, and only for textured faces.
        if (mat && mat->diffuseMap) {
            rt.uvIndex = (uint32_t)textureQueue.size();
            textureQueue.push_back(reverse ? TriangleUV{*uvC, *uvB, *uvA}
                                           : TriangleUV{*uvA, *uvB, *uvC});
        }
#endif
        rt.material       = mat;
        rt.ignoreZBuffer  = ignoreZBuffer;
        rt.noWriteZBuffer = noWriteZBuffer;
        rt.zBias          = obj->zBias;
        rt.objAlpha       = objAlpha;
        rt.avgZ           = avgZ;
        rt.brightnessPrecomputed = false;
#if LIGHTING
        rt.brightnessPrecomputed = objectLocalLight;
#endif
#if MAX_PICK_QUERIES > 0
        rt.sourceObject        = obj;
        rt.sourceTriangleIndex = srcTriIdx;
#endif
#if JET_FLAT_KERNEL && LIGHTING && !Z_BUFFERING && !HALF_WIDTH_BUFFERS && !FIELD_BUFFERS && \
    !DEBUG_OVERDRAW && !POSTFX_CELLSHADING && !Z_BRIGHTNESS && !TEXTURE_MAPPING && \
    !SCREEN_DOOR_ALPHA && MAX_PICK_QUERIES == 0 && !PERSPECTIVE_CORRECT_TEXTURES
        // Same gates as drawTriangleImpl's flat path, evaluated once per
        // triangle per frame instead of once per band: opaque, no water/additive
        // blend, no engine depth fog, and either emissive/UNLIT or FLAT with the
        // object-local Lambert already in v1.
        rt.flatOpaque = false;
        if (mat && rt.objAlpha == 255 && mat->alpha == 255 &&           // rt.objAlpha: a coverage-blended sliver takes the general path
            mat->shadingMode != ShadingMode::WATER_REFLECT && mat->shadingMode != ShadingMode::ADDITIVE
#if DEPTH_ALPHA_BLEND && FAST_Z
            && avgZ <= depthFogNear
#endif
           ) {
            const bool emissive = mat->emissive || mat->shadingMode == ShadingMode::UNLIT;
            const bool flatShaded = (!directionalLight && !ambientLight) || mat->shadingMode == ShadingMode::FLAT;
            if (emissive) {
                rt.flatColor = jetWs565(mat->color);
                rt.flatOpaque = true;
            } else if (flatShaded && (rt.brightnessPrecomputed || !directionalLight)) {
                // drawTriangleImpl: no lights at all -> full brightness; ambient only -> 0 (tint)
                const uint16_t brightness = directionalLight ? rt.v1.lambertBrightness : (ambientLight ? 0 : 255);
                const uint8_t ambR = ambientLight ? ambientLight->color.r : 0;
                const uint8_t ambG = ambientLight ? ambientLight->color.g : 0;
                const uint8_t ambB = ambientLight ? ambientLight->color.b : 0;
                rt.flatColor = jetWs565(jetModulateRGB565(mat->color, brightness, ambR, ambG, ambB,
                                                          (uint16_t)(255u + mat->specular)));
                rt.flatOpaque = true;
            }
        }
#endif
#if JET_PROFILE
        const uint32_t jpe2 = jet_prof_now(); L.prof_cyc[JP_EMIT_BUILD] += jpe2 - jpe1; ++L.prof_cnt[JP_EMIT_BUILD];
#endif
        // Preserve the old stable 64-bucket ordering exactly, including
        // noWriteZBuffer taking precedence when both special flags are set.
        uint8_t bucket = 0;
        if (!noWriteZBuffer) {
            bucket = SortBucketCount - 1;
            if (!ignoreZBuffer) {
                constexpr int32_t zBiasScale = 256;
                constexpr int K = SortBucketCount - 2;
                int32_t key = avgZ - static_cast<int32_t>(obj->zBias) * zBiasScale;
#if JET_CULL_SLIVERS
                key += sliverPush;                                    // hairlines sort behind their neighbours
#endif
                const int32_t range = std::max<int32_t>(camera->farPlane - camera->nearPlane, 1);
                // Exact old bucket formula; the 32-bit hardware divide when the
                // product fits (always, for world-scale keys), int64 otherwise.
                // (A Q16 reciprocal moved keys across bucket edges: reviewer finding.)
                const int32_t rel = key - camera->nearPlane;
                int b = (rel > -(INT32_MAX / K) && rel < INT32_MAX / K)
                      ? (rel * K) / range
                      : static_cast<int>((static_cast<int64_t>(rel) * K) / range);
                b = std::max(0, std::min(b, K - 1));
                bucket = static_cast<uint8_t>(K - b);
            }
        }
        renderBuckets.push_back(bucket);
        {
            const int32_t ylo = std::min({a.position.y, b.position.y, c.position.y});
            const int32_t yhi = std::max({a.position.y, b.position.y, c.position.y});
            renderYSpan.push_back((int16_t)std::max<int32_t>(-32768, std::min<int32_t>(32767, ylo)));
            renderYSpan.push_back((int16_t)std::max<int32_t>(-32768, std::min<int32_t>(32767, yhi)));
        }
#if JET_PROFILE
        L.prof_cyc[JP_EMIT_TAIL] += jet_prof_now() - jpe2;
#endif
    };

    // Render triangles with backface culling and shading
    for (size_t triIdx = 0; triIdx < meshSource->triangles.size(); ++triIdx) {
        const auto& triangle = meshSource->triangles[triIdx];
        const auto& vA = transformedVertices[triangle.v1];
        const auto& vB = transformedVertices[triangle.v2];
        const auto& vC = transformedVertices[triangle.v3];

        // Classify each vertex against the near plane.
        const int outMask = (vA.position.z < nz ? 1 : 0)
                          | (vB.position.z < nz ? 2 : 0)
                          | (vC.position.z < nz ? 4 : 0);

        if (outMask == 7) continue;               // fully behind near plane

#if TEXTURE_MAPPING
        #define JET_UV_ARGS(A, B, C) , (A), (B), (C)
#else
        #define JET_UV_ARGS(A, B, C)
#endif
#if MAX_PICK_QUERIES > 0
        const int32_t srcTriIdx = (int32_t)triIdx;
        #define JET_EMIT_TRI(A, B, C, M, U, V, W)  emitTri((A), (B), (C), (M) JET_UV_ARGS(U, V, W), srcTriIdx)
#else
        #define JET_EMIT_TRI(A, B, C, M, U, V, W)  emitTri((A), (B), (C), (M) JET_UV_ARGS(U, V, W))
#endif

        if (outMask == 0) {                       // fast path: fully in front
            Material* effectiveMat = triangle.material;
            if (triangle.colorBaked) {
                s_bakedMat.color = triangle.bakedColor;
                effectiveMat = &s_bakedMat;
            }
            JET_EMIT_TRI(vA, vB, vC, effectiveMat,
                         &meshSource->vertices[triangle.v1].uv,
                         &meshSource->vertices[triangle.v2].uv,
                         &meshSource->vertices[triangle.v3].uv);
            continue;
        }

        // Straddling near plane — produce a clipped polygon (3 or 4 verts)
        // while preserving winding order of the original triangle.
        const Vector3 cA = cameraPosition(meshSource->vertices[triangle.v1].position);
        const Vector3 cB = cameraPosition(meshSource->vertices[triangle.v2].position);
        const Vector3 cC = cameraPosition(meshSource->vertices[triangle.v3].position);
        RenderVertex clippedInput[3] = { vA.expand(), vB.expand(), vC.expand() };
#if TEXTURE_MAPPING
        if (!triangle.colorBaked && triangle.material && triangle.material->diffuseMap) {
            clippedInput[0].uv = meshSource->vertices[triangle.v1].uv;
            clippedInput[1].uv = meshSource->vertices[triangle.v2].uv;
            clippedInput[2].uv = meshSource->vertices[triangle.v3].uv;
        }
#endif
        const RenderVertex* vs[3]  = { &clippedInput[0], &clippedInput[1], &clippedInput[2] };
        const Vector3*        cvs[3] = { &cA, &cB, &cC };
        const bool in[3] = { (outMask & 1) == 0,
                             (outMask & 2) == 0,
                             (outMask & 4) == 0 };

        RenderVertex poly[4];
        int polyN = 0;
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3;
            if (in[i]) poly[polyN++] = *vs[i];
            if (in[i] != in[j]) {
                // One endpoint in, one out — add the near-plane intersection.
                if (in[i])
                    poly[polyN++] = clipEdge(*vs[j], *vs[i], *cvs[j], *cvs[i]);
                else
                    poly[polyN++] = clipEdge(*vs[i], *vs[j], *cvs[i], *cvs[j]);
            }
        }

        if (polyN >= 3) {
            Material* effectiveMat = triangle.material;
            if (triangle.colorBaked) { s_bakedMat.color = triangle.bakedColor; effectiveMat = &s_bakedMat; }
            JET_EMIT_TRI(poly[0], poly[1], poly[2], effectiveMat, &poly[0].uv, &poly[1].uv, &poly[2].uv);
        }
        if (polyN == 4) JET_EMIT_TRI(poly[0], poly[2], poly[3], triangle.colorBaked ? &s_bakedMat : triangle.material, &poly[0].uv, &poly[2].uv, &poly[3].uv);
        #undef JET_EMIT_TRI
        #undef JET_UV_ARGS
    }
}

} // namespace Renderer
