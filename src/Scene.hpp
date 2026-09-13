#ifndef SCENE_HPP
#define SCENE_HPP

#include <vector>
#include <algorithm>
#include "Object.hpp"
#include "Camera.hpp"
#include "Light.hpp"
#include "Renderer.hpp"
#include "JetConfig.hpp"
#include "PostFX.hpp"
#include "Sprite2D.hpp"

namespace Renderer {

/// @brief Top-level container that owns the scene graph and drives rendering.
///
/// Holds the active camera, lights, object list and per-frame state, and
/// exposes a single `render()` entry point that runs the full
/// transform/cull/raster/post-FX pipeline.
extern float   jetSliverMinThickness;   // JET_CULL_SLIVERS: drop triangles thinner than this many pixels (Scene.cpp)
extern float   jetSliverPushMaxThickness; // JET_CULL_SLIVERS: only triangles thinner than this (px) are pushed back
extern int32_t jetSliverPushZ;          // JET_CULL_SLIVERS: sort sub-pixel triangles this many units farther back

class Scene {
public:
    /// @brief Construct a scene bound to caller-owned framebuffers.
    /// @param framebuffer RGB565 colour buffer of size screenWidth*screenHeight.
    /// @param zBuffer Depth buffer of size ZBUFFER_STRIDE(screenWidth)*screenHeight; pass nullptr when Z_BUFFERING is disabled.
    /// @param screenWidth Output width in pixels.
    /// @param screenHeight Output height in pixels.
    Scene(uint16_t* framebuffer, uint16_t* zBuffer, int screenWidth, int screenHeight);
    ~Scene();

    int   frameCounter = 0;       ///< Incremented once per render(); useful for animations and dither parity.
    float waterTime    = 0.0f;    ///< Accumulated wall-clock seconds; set each frame by the caller before render/prepareFrame.

    /// @name Per-frame counters populated by render()
    /// @{
    /// `lastFrameDrawnObjects` is the number of enabled objects that
    /// survived the AABB frustum cull. `lastFrameDrawnTriangles` is the
    /// number of triangles submitted to the rasteriser (renderQueue size
    /// after all culling). `lastFrameRasterizedTriangles` is the subset of
    /// those that produced rasterizer work (drawTriangle returned true).
    int lastFrameDrawnObjects        = 0;
    int lastFrameDrawnTriangles      = 0;
    int lastFrameRasterizedTriangles = 0;
    /// @}

    /// @brief Add an object to the scene.
    /// @param obj Object to add. Pointer is borrowed; caller retains ownership.
    void addObject(Object* obj);

    /// @brief Add a point light to the scene.
    /// @param light Light to add. Pointer is borrowed; caller retains ownership.
    void addPointLight(PointLight* light);

    /// @brief Register a 2D screen-space overlay to be drawn after every render().
    /// @param sprite Sprite to add. Pointer is borrowed; caller retains ownership.
    void addSprite(Sprite2D* sprite);

    /// @brief Set the active camera.
    /// @param cam Camera pointer (borrowed).
    void setCamera(Camera* cam);
    /// @brief Get the active camera.
    /// @return Pointer to the current camera, or nullptr if none is set.
    Camera* getCamera() { return camera; }

    /// @brief Set the active directional light.
    /// @param light Directional light (borrowed). May be nullptr.
    void setDirectionalLight(DirectionalLight* light);
    /// @brief Get the active directional light.
    DirectionalLight* getDirectionalLight() { return directionalLight; }

    /// @brief Set the active ambient light.
    /// @param light Ambient light (borrowed). May be nullptr.
    void setAmbientLight(AmbientLight* light);
    /// @brief Get the active ambient light.
    AmbientLight* getAmbientLight() { return ambientLight; }

    using RasterExecutor = void (*)(Scene&);
    /// @brief Run the full pipeline for one frame: cull, transform, rasterise, post-FX.
    /// An optional frontend executor replaces the full-screen raster pass.
    /// It must join its workers and publish statistics before returning.
    void render(RasterExecutor executor = nullptr);

    /// @brief Phase 1 of split rendering: clear [yBandMin, yBandMax) on the rasteriser,
    ///        transform and depth-sort all objects. Does NOT rasterise triangles.
    ///
    ///        Call this once per frame before any rasterizeBand() calls.
    ///        The rasteriser's yBandMin/yBandMax gate which rows clearBuffers() clears
    ///        so the framebuffer pointer can be a virtual base (adjusted for band offset).
    void prepareFrame();

    /// @brief Phase 2 of split rendering: rasterise the sorted render queue for
    ///        rows [yMin, yMax) only. Uses a thread-local copy of the rasteriser so
    ///        concurrent calls with non-overlapping y ranges are safe when Z_BUFFERING==0.
    ///
    ///        May be called from multiple threads simultaneously with disjoint bands.
    // Parallel callers supply separate zeroed flags, then OR them after joining to count unique triangles.
    // The array must hold triangleFlagsBytes() entries, NOT lastFrameDrawnTriangles: split prepare leaves the queue SPARSE
    // (lane 0 fills from 0, lane 1 from lane1RegionBegin()), and the flags are indexed by the PHYSICAL queue slot. A 559-triangle
    // frame reached physical index 597; sizing by the drawn count is a heap overflow (reviewer finding 3).
    // With flags supplied this does not write shared frame statistics.
    void rasterizeBand(int yMin, int yMax, uint8_t* triangleFlags = nullptr);
    /// @brief Depth gate for rasterizeBand(): only queued triangles whose painter's
    ///        key (avgZ - zBias * 256, camera-space units) lies in [min, max) are drawn.
    ///        Lets a caller split one prepared frame into a far pass, something drawn
    ///        by other means (a column-rendered terrain), and a near pass. Default: all.
    void setBandDepthGate(int32_t keyMin, int32_t keyMax) { bandKeyMin = keyMin; bandKeyMax = keyMax; }
    /// @brief Multi-core prepareFrame: `start` runs fn(arg) on another core and returns at once, `join` blocks until it has
    /// returned. With both set, prepareFrame splits the enabled objects into two ranges by vertex count (serial order kept).
    struct PrepareExecutor { void (*start)(void* user, void (*fn)(void*), void* arg) = nullptr; void (*join)(void* user) = nullptr; void* user = nullptr;
                             int64_t (*now)(void* user) = nullptr; };   // optional µs clock: fills lastPrepareTimes (diagnostics only)
    /// @brief Last split prepareFrame, µs: lane 0 on the calling core, lane 1 on the executor (timed there), the caller's join wait,
    /// the lane-1 append; cost = the enabled vertices + triangles each lane was given (the split proxy), tris = triangles each lane emitted.
    struct PrepareTimes { int64_t lane0 = 0, lane1 = 0, join = 0, merge = 0; uint64_t cost0 = 0, cost1 = 0; uint32_t tris0 = 0, tris1 = 0; };
    PrepareTimes lastPrepareTimes;
    /// @brief Lane-0 share of the split cost proxy. Starts at 0.5 and, when PrepareExecutor::now is set, is nudged each frame
    /// toward equal lane times (the proxy misses the per-object emit cost). The split point never changes what is emitted, only
    /// which lane emits it, so frames stay identical.
    float prepareSplitFrac = 0.5f;
    /// @brief Where the queues actually are right now: a vector that outgrows its reserve reallocates, and on the S3 a big
    /// reallocation lands in PSRAM, which the boot-time address cannot show.
    const void* queueData() const { return renderQueue.data(); }
    size_t queueCapacity() const { return renderQueue.capacity(); }
    const void* lane1QueueData() const { return renderQueue.data() + lane1Begin; }
    size_t lane1QueueCapacity() const { return queueCap - lane1Begin; }
    uint32_t lane1RegionBegin() const { return lane1Begin; }
    uint32_t queueCapTriangles() const { return queueCap; }
    uint32_t prepareOverflowCount() const { return prepareOverflows; }   // frames redone serially (a lane region filled up)
    uint32_t queueGrowthCount() const { return queueGrowths; }           // buffer reallocations: must stay 0 on the S3 (PSRAM)
    /// Split-prepare balance target: lane 1's time is steered toward ratio x lane 0's (1 = equal; < 1 moves work to the caller's lane).
    void setPrepareLaneRatio(float ratio) { prepareLaneRatio = ratio > 0.05f ? ratio : 0.05f; }
    uint32_t prepareTruncationCount() const { return prepareTruncations; } // frames that STILL overflowed after 8 growths: triangles were dropped
    /// @brief Size a rasterizeBand() triangleFlags array. Indices are physical queue slots, which split prepare leaves sparse.
    int triangleFlagsBytes() const { return static_cast<int>(queueCap); }
    void setPrepareExecutor(const PrepareExecutor& e) { prepareExecutor = e; }
    PrepareExecutor prepareExecutor;
    size_t lastPrepareSplit = 0;   // objects index where lane 1 started (0 = serial), for diagnostics
    /// @brief Thread-safe rasterizeBand for multi-core rendering: explicit framebuffer base (row 0 = engine row 0,
    /// like setFramebuffer) and depth gate, returns the rasterized-triangle count, touches no shared state.
    int rasterizeBandInto(int yMin, int yMax, uint16_t* framebufferBase, int32_t keyMin = INT32_MIN, int32_t keyMax = INT32_MAX);

    /// @brief Clear only the rows [yMin, yMax) of the current framebuffer without
    ///        re-running the transform or sort pipeline. Use this for bands 1+ when the
    ///        render queue from the preceding prepareFrame() call is still valid.
    ///
    ///        Sets the rasteriser's yBandMin/yBandMax before clearing so the
    ///        band-aware clear writes only into the correct region.
    void clearBand(int yMin, int yMax);

    /// @brief Advance the internal frame counter by one. Normally called automatically
    ///        by render(); use this when driving the pipeline via prepareFrame()/rasterizeBand().
    void advanceFrameCounter() { frameCounter++; }

    /// @brief Get total scene statistics (independent of camera position).
    /// @param objectCount Out: number of enabled objects.
    /// @param triangleCount Out: total triangle count across enabled objects.
    /// @param vertexCount Out: total vertex count across enabled objects.
    void getStatistics(int& objectCount, int& triangleCount, int& vertexCount);

    /// @brief Set the colour used to clear the framebuffer.
    /// @param color RGB565 clear colour.
    void setBackcolor(uint16_t color) { backcolor = color; }

    /// @brief Replace the colour buffer pointer (without changing dimensions).
    /// @param framebuffer New caller-owned RGB565 buffer.
    void setFramebuffer(uint16_t *framebuffer);

    /// @brief Pre-size the per-frame render queues for n triangles. Call once
    ///        after construction so the vectors are allocated where you want
    ///        them (e.g. before external-RAM fallback kicks in on an ESP32-S3).
    void reserveQueues(size_t n);
    /// @brief After prepareFrame(): split renderOrder into per-band index lists for
    ///        bands of `bandRows` rows (screenHeight / bandRows + 1 lists, painter's
    ///        order preserved). rasterizeBand(y0, y1) then walks only the list of the
    ///        band that starts at y0 (when y0 % bandRows == 0 and y1 - y0 <= bandRows)
    ///        instead of testing every queued triangle against the band.
    void buildBandLists(int bandRows);
    /// @brief Storage address of the render queue (placement diagnostics).
    const void* queueStorage() const { return renderQueue.data(); }
    /// @brief Storage address of the lane-1 render queue (multi-core prepareFrame placement diagnostics).
    const void* lane1QueueStorage() const { return renderQueue.data() + lane1Begin; }

    /// @brief Enable or disable per-frame framebuffer clearing.
    /// @param clear True to clear before rendering, false to preserve previous content.
    void setClearBuffer(bool clear) { clearRenderBuffer = clear; }

    /// @brief Hot-swap framebuffer, z-buffer and dimensions (e.g. on window resize).
    ///
    /// The caller owns both buffers and is responsible for freeing the old
    /// ones AFTER this call returns. PostFX is recreated internally to
    /// pick up the new dimensions.
    /// @param newFramebuffer New caller-owned colour buffer.
    /// @param newZBuffer New caller-owned depth buffer.
    /// @param newWidth New width in pixels.
    /// @param newHeight New height in pixels.
    void resize(uint16_t* newFramebuffer, uint16_t* newZBuffer,
                int newWidth, int newHeight);

    /// @brief Get the underlying rasteriser.
    Rasterizer* getRenderer() { return renderer; }
    /// @brief Get the mutable list of scene objects.
    std::vector<Object*>& getObjects() { return objects; }
    /// @brief Get the mutable list of point lights.
    std::vector<PointLight*>& getPointLights() { return pointLights; }
    /// @brief Get the mutable list of materials owned by the scene.
    std::vector<Material*>& getMaterials() { return materials; }
    /// @brief Get the list of registered 2D sprites.
    /// On HALF_WIDTH_BUFFERS builds the display layer composites sprites
    /// during scanout at full resolution; expose the list so it can do so.
    std::vector<Sprite2D*>& getSprites() { return sprites; }

#if MAX_PICK_QUERIES > 0
    /// @brief Submit screen-space pick points to be tested during the next render().
    ///
    /// Excess queries beyond MAX_PICK_QUERIES are silently dropped. Pass
    /// count == 0 (or just don't call this) to disable picking. The renderer
    /// reads queries during render() and writes the matching slots in the
    /// pick result array. Both arrays are owned by Scene; the host should
    /// copy queries in by value and read results back after render().
    /// @param queries Caller-owned array of pick queries.
    /// @param count Number of valid entries in @p queries.
    void setPickQueries(const PickQuery* queries, int count);

    /// @brief Get the pick results from the most recent render() call.
    /// @return Pointer to an internal array of MAX_PICK_QUERIES results.
    const PickResult* getPickResults() const { return pickResults; }

    /// @brief Get the number of active pick queries set for the next render().
    int getPickQueryCount() const { return pickQueryCount; }
#endif


    uint16_t* backgroundGradientColors = nullptr;   ///< Optional per-row background gradient (screenHeight entries) used during clear.
    /// If > 0, the clear fills only rows [0, backgroundClearMaxRows): below the
    /// horizon the ground plane rewrites every pixel each frame, so clearing
    /// them is pure PSRAM bandwidth waste (used by the flight scene).
    int backgroundClearMaxRows = 0;

    /// @name Distance-based level of detail (LOD)
    /// @brief Global LOD selection driven by camera-to-object distance.
    ///
    /// `lodScale` is the world-units-per-LOD-step. Setting it to 0 (the
    /// default) disables global LOD and every Object renders its own
    /// mesh as before. With `lodScale = 4096`, an Object with two LOD
    /// meshes attached renders LOD 0 below 4096 units, LOD 1 from 4096
    /// to 8191, LOD 2 from 8192 onward (or fades out / persists past
    /// the last LOD depending on the Object's `lodPersist` flag).
    ///
    /// `lodBias` is added to the computed level globally — a bias of -1
    /// pushes everything one LOD step higher in detail, +1 cheaper. This
    /// composes with the per-Object `lodBias` and is intended for runtime
    /// quality knobs (e.g. perf-driven dynamic adjustment).
    /// @{
    int32_t lodScale = 0;   ///< World units per LOD step; 0 disables global LOD.
    int8_t  lodBias  = 0;   ///< Scene-wide LOD level offset (added to each object's choice).
    /// @}

private:
    // UVs are immutable mesh attributes, not transformed attributes. Keep
    // them out of the scratch vertices and the common triangle queue.
    // JET_QUEUE_FLAT_ONLY: the queued vertex keeps only what the flat kernel reads (position). The normal and the
    // Lambert brightness are what the GENERAL path needs, and they cost 14 of the 28 bytes per queued vertex, i.e.
    // 42 of RenderTri's 112. Dropping them fits ~1400 triangles in the S3's internal RAM instead of ~900, and the
    // queue living in PSRAM costs 7-8 fps (measured). A scene with any non-flat triangle must NOT build with this:
    // emitTri counts such triangles in nonFlatEmitted and the caller is expected to check it (the valley's are all flat).
// JET_FLAT_KERNEL is opt-in and has no default definition anywhere, so `#if JET_FLAT_KERNEL` silently reads 0 in any translation
// unit that forgets -D. That is not a harmless default: the general path reads live materials and lights while rasterising, so a
// host tool built without it proves the WRONG path (stepearly: 3177 of 3600 frames differ, single-threaded, by order alone).
// Defining it here lets callers test and report the value rather than mis-evaluate it.
#ifndef JET_FLAT_KERNEL
#define JET_FLAT_KERNEL 0
#endif
#ifndef JET_QUEUE_FLAT_ONLY
#define JET_QUEUE_FLAT_ONLY 0
#endif
    struct PipelineVertex {
        Vector3 position;
#if LIGHTING
        Vector3 normal;
        uint16_t lambertBrightness = 0;
#endif
        template<class V> void assign(const V& v) {
            position = v.position;
#if LIGHTING
            normal = v.normal;
            lambertBrightness = v.lambertBrightness;
#endif
        }
        RenderVertex expand() const {
            RenderVertex v;
            v.position = position;
#if LIGHTING
            v.normal = normal;
            v.lambertBrightness = lambertBrightness;
#endif
            return v;
        }
    };
    // What the QUEUE stores per vertex. Under JET_QUEUE_FLAT_ONLY that is position only: the flat kernel reads nothing
    // else, and the 14 bytes of normal + brightness per vertex are 42 of RenderTri's 112. The transform scratch above
    // keeps them - lighting still runs per vertex before emit, it just is not carried into the queue.
#if JET_QUEUE_FLAT_ONLY
    // Position only. The three Lambert brightnesses live in RenderTri::lb[] instead of beside each position: a 12 B position plus a
    // 2 B brightness aligns to 16 B, wasting 2 B per vertex, and together with flatColor placed next to them that padding is what
    // took the device entry from 56 B to 64 B. At 64 B the x4 detail queue (~2 032 entries) cannot fit the S3's largest internal
    // RAM region (120 KiB, measured at boot) and lands in PSRAM; at 56 B 2 100 entries are 117 600 B and do.
    struct QueuedVertex {
        Vector3 position;
        template<class V> void assign(const V& v) { position = v.position; }
    };
#else
    using QueuedVertex = PipelineVertex;
#endif
#if TEXTURE_MAPPING
    struct TriangleUV { Vector2 a, b, c; };
    std::vector<TriangleUV> textureQueue;
#endif
    struct RenderTri {
        QueuedVertex v1, v2, v3;
#if JET_QUEUE_FLAT_ONLY
#if LIGHTING
        uint16_t lb[3] = {0, 0, 0};   // Lambert brightness of v1, v2, v3 (see QueuedVertex for why they are not inside it)
#endif
        uint16_t flatColor = 0;       // placed here, not after avgZ, so the small fields fill the gap before `material` (56 B on Xtensa)
#endif
#if TEXTURE_MAPPING
        uint32_t uvIndex;
#endif
        Material* material;
        int32_t avgZ;
#if !JET_QUEUE_FLAT_ONLY
        // JET_FLAT_KERNEL: FLAT/UNLIT, alpha 255, colour fixed at emit time (RenderTri is 100 B on Xtensa)
        // (already in wire order): rasterizeBand draws it without drawTriangle.
        uint16_t flatColor = 0;
#endif
        bool flatOpaque = false;
        // Pack the three booleans together so the UV index replaces
        // padding rather than growing the total payload of textured faces.
        bool ignoreZBuffer : 1;
        bool noWriteZBuffer : 1;
        // When true, v1/v2/v3.lambertBrightness has been precomputed in
        // object-local space by renderObject (see "objectLocalLight" path
        // in Scene.cpp). drawTriangle skips its own jetShadeBrightness
        // calls in that case and reads the cached values directly. Only
        // ever set for objects whose materials are all non-specular.
        bool brightnessPrecomputed : 1;
        // JET_CULL_SLIVERS: the sort key carried jetSliverPushZ (a hairline sorted behind its
        // neighbours); setBandDepthGate classifies by the same key (reviewer: the gate rebuilt
        // the key without it, so a pushed sliver could land in the other pass).
        bool sliverPushed : 1;
        int8_t zBias;
        // Per-object alpha multiplier (255 = no per-object fade); folded
        // into the per-pixel screen-door alpha at raster time.
        uint8_t objAlpha;
        // Layout-independent access to the three queued vertices (flat-only keeps brightness in lb[], the wide queue in the vertex).
        template<class V> void setV(int i, const V& v) {
            QueuedVertex& q = i == 0 ? v1 : (i == 1 ? v2 : v3);
            q.assign(v);
#if JET_QUEUE_FLAT_ONLY && LIGHTING
            lb[i] = v.lambertBrightness;
#endif
        }
        RenderVertex expandV(int i) const {
            const QueuedVertex& q = i == 0 ? v1 : (i == 1 ? v2 : v3);
#if JET_QUEUE_FLAT_ONLY
            RenderVertex v; v.position = q.position;
#if LIGHTING
            v.lambertBrightness = lb[i];
#endif
            return v;
#else
            return q.expand();
#endif
        }
#if LIGHTING
        uint16_t brightness1() const {
#if JET_QUEUE_FLAT_ONLY
            return lb[0];
#else
            return v1.lambertBrightness;
#endif
        }
#endif
#if MAX_PICK_QUERIES > 0
        // Source object + ORIGINAL triangle index (in obj->triangles) for
        // pick attribution. Carried through the painter sort.
        Object* sourceObject;
        int32_t sourceTriangleIndex;
#endif
    };
    int32_t bandKeyMin = INT32_MIN, bandKeyMax = INT32_MAX;   // setBandDepthGate
    std::vector<RenderTri> renderQueue;
    // One byte per triangle: background, 64 far-to-near depth buckets,
    // then overlay. Sorting never needs to fetch the full triangle payload.
    static constexpr int SortBucketCount = 66;
    std::vector<uint8_t> renderBuckets;
    // Per-queued-triangle screen y extent (min, max) clamped to int16, captured
    // at emit time so rasterizeBand() can reject out-of-band triangles without
    // touching the (possibly external-RAM) RenderTri itself.
    std::vector<int16_t> renderYSpan;
    // Multi-core prepareFrame: objects [0, k) emit into lane 0 (the members above), objects [k, N) into lane 1 on
    // another core; lane 1 is appended to lane 0 afterwards, so insertion order (the painter's tie order) is exactly
    // the serial order. Each lane has its own transform scratch and profile counters.
    /// @brief One lane's slice of a shared queue: fixed storage, its own cursor, never reallocates (a reallocation on the S3
    /// lands in PSRAM, which costs 7-8 fps). Overflow is counted, never a silently dropped triangle: prepareFrame redoes the
    /// frame serially over the whole buffer when it happens.
    template <class T>
    struct LaneArena {
        T* p = nullptr; uint32_t n = 0, cap = 0, over = 0; T sink{};
        void reset(T* base, uint32_t capacity) { p = base; n = 0; cap = capacity; over = 0; }
        T& emplace_back() { if (n < cap) return p[n++]; ++over; return sink; }
        void push_back(const T& v) { if (n < cap) p[n++] = v; else ++over; }
        uint32_t size() const { return n; }
    };
    struct PrepareLane {
        LaneArena<RenderTri> q;
        LaneArena<uint8_t> bkt;
        LaneArena<int16_t> ysp;
#if TEXTURE_MAPPING
        std::vector<TriangleUV>* textureQueue = nullptr;
#endif
        std::vector<PipelineVertex> transformedVertices;
        uint32_t prof_cnt[32] = {}, prof_cyc[32] = {};   // JET_PROFILE counters of this lane, added to the globals after the merge
        int drawnObjs = 0;
    };
    PrepareLane lanes[2];
    uint32_t queueCap = 0;        // triangles the single queue holds; lane 0 fills [0, lane1Begin), lane 1 [lane1Begin, queueCap)
    uint32_t lane0Count = 0, lane1Begin = 0, lane1Count = 0;
    uint32_t prepareOverflows = 0;   // frames redone serially because a lane region filled up
    uint32_t nonFlatEmitted = 0;     // JET_QUEUE_FLAT_ONLY: triangles the queue cannot describe (must stay 0)
    uint32_t queueGrowths = 0;       // times the buffer itself was too small and had to grow (must stay 0: a grown buffer can land in PSRAM)
    float prepareLaneRatio = 1.0f;   // see setPrepareLaneRatio
    uint32_t prepareStamp = 0;       // bumped once per renderPrepare; Object::trianglesSortedStamp keys the per-frame depth sort off it
    uint32_t prepareTruncations = 0; // frames abandoned after 8 growths without fitting: the ONLY path on which a triangle is lost
    uint32_t laneHigh[2] = {0, 0};   // decaying high-water of each lane's emitted triangles; the regions are sized from these, not from a mean
#if TEXTURE_MAPPING
    std::vector<TriangleUV> lane1Texture;
#endif
    int32_t prepCam[6] = {};   // camera cos/sin of the frame, for lane 1
    size_t lane1First = 0, lane1Last = 0;
    void prepareObjects(PrepareLane& L, size_t first, size_t last);
    // Painter's-sort output as indices into renderQueue, rebuilt by
    // prepareFrame() each frame. Sorting (scattering) 4-byte indices
    // instead of whole RenderTri structs avoids a full second copy of the
    // queue per frame; rasterizeBand() walks this to draw in depth order.
    std::vector<int32_t> renderOrder;
    static constexpr int MaxBandLists = 8;
    std::vector<int32_t> bandOrder[MaxBandLists];   // buildBandLists() output
    int bandListRows = 0;                           // rows per band the lists were built for (0 = none)

    Camera* camera;
    DirectionalLight* directionalLight;
    AmbientLight* ambientLight;
    Rasterizer* renderer = nullptr;
    PostFX* postFX = nullptr;

    uint16_t* framebuffer;
    uint16_t* zBuffer;
    int screenWidth;
    int screenHeight;
    bool* scanlinesUpdated;

    std::vector<Object*> objects;
    std::vector<PointLight*> pointLights;
    std::vector<Material*> materials;
    std::vector<Sprite2D*> sprites;

    uint16_t backcolor = 0;
    bool clearRenderBuffer = true;
    bool renderEvenLines = false;

    float cameraMatrix[9] = {}; // Same composed camera transform for the entire frame.

    // Frustum side-plane normal lengths for the quick sphere cull in
    // cullObject(): |(fovFactor, ±screenW/2)| and |(fovFactor, ±screenH/2)|.
    // Recomputed once per frame in prepareFrame() because fovFactor
    // changes at runtime (boost FOV kick).
    float cullPlaneLh = 1.0f;
    float cullPlaneLv = 1.0f;

    bool cullObject(Object* obj,
                    int32_t camCosX, int32_t camSinX,
                    int32_t camCosY, int32_t camSinY,
                    int32_t camCosZ, int32_t camSinZ) const;

    int rasterizeBandImpl(int yMin, int yMax, uint8_t* triangleFlags, uint16_t* framebufferBase, int32_t keyMin, int32_t keyMax, bool profile);
    void renderObject(PrepareLane& L, Object* obj,
                      int32_t camCosX, int32_t camSinX,
                      int32_t camCosY, int32_t camSinY,
                      int32_t camCosZ, int32_t camSinZ,
                      uint8_t objAlpha,
                      Object* meshSource = nullptr);
    void reconstructCheckerboard();
    void clearBuffers();

public:
    /// @brief Composite all enabled sprites onto the current framebuffer.
    ///        Normally called automatically by render(); call explicitly when
    ///        driving the pipeline via prepareFrame()/rasterizeBand().
    void drawSprites();

private:

#if MAX_PICK_QUERIES > 0
    PickQuery  pickQueries[MAX_PICK_QUERIES];
    PickResult pickResults[MAX_PICK_QUERIES];
    int        pickQueryCount = 0;
#endif
};

} // namespace Renderer

#endif // SCENE_HPP
