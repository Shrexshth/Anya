/**
 * blend.js – Expression parameter blending and smooth transition interpolation.
 *
 * Public API (attached to window.anya after avatar.js initialises):
 *   anya.setEmotion(weights, transitionMs)
 *       weights  – plain object { emotionName: 0‑1, … }  (values are normalised internally)
 *       transitionMs – ease duration in milliseconds (default 600)
 *
 * Internally the system:
 *   1. Loads every exp3.json file listed in expressions.map.json once at startup.
 *   2. Each animation frame (beforeModelUpdate), for every model parameter that
 *      appears in *any* loaded expression it:
 *        a. Multiplies the parameter's value in each expression by the expression's weight.
 *        b. Sums the weighted values across all active expressions.
 *        c. Clamps to the parameter's [min, max] range read from the moc.
 *        d. Eases the *current* value toward the *target* value (no snapping).
 *      eye_open_cap from the map is applied on top of the blink layer; animation.js
 *      reads window.anya.eyeOpenCap for this.
 *   3. Blend-emotions (jealous, caring, proud) are expanded from their weighted lists.
 */

(function () {
    "use strict";

    const MAP_URL  = "/assets/model/expressions.map.json";
    const BASE_URL = "/assets/model/";

    /** @type {Map<string, {params: Map<string, number>}>} emotion name → param map */
    const _loadedExpressions = new Map();

    /** @type {Map<string, {file:string|null, eye_open_cap:number, default_intensity:number}>} */
    let _emotionMap = {};

    /** @type {Map<string, {weights: Object<string,number>}>} blend definitions */
    let _blendDefs = {};

    /**
     * Current target weights per *base* emotion name (neutral, happy, sad, …).
     * @type {Object<string, number>}
     */
    let _targetWeights  = {};

    /**
     * Currently interpolated weights (what is applied this frame).
     * @type {Object<string, number>}
     */
    let _currentWeights = {};

    /** Ease speed: fraction of the gap closed per ms. Derived from transitionMs. */
    let _easeAlphaPerMs = 0.004; // default ~250 ms half-life

    /** True once the map and all expression files are fully loaded. */
    let _ready = false;

    /** Unified param range cache: paramId → {min, max} read from coreModel. */
    let _paramRanges = {};

    // ──────────────────────────────────────────────────────────────────────────
    // Boot: wait for avatar.js to finish, then load everything.
    // ──────────────────────────────────────────────────────────────────────────

    function _boot() {
        if (window.anya && window.anya.model) {
            _init(window.anya.model);
        } else {
            window.addEventListener("anya:modelReady", (ev) => _init(ev.detail.model));
        }
    }

    async function _init(model) {
        try {
            // 1. Load the emotion → file map.
            const mapResp = await fetch(MAP_URL);
            if (!mapResp.ok) throw new Error("Failed to fetch expressions.map.json");
            const mapData = await mapResp.json();

            _blendDefs = mapData["_blends"] || {};
            delete mapData["_blends"];
            delete mapData["_comment"];
            _emotionMap = mapData;

            // 2. Collect unique files to load.
            const fileSet = new Set();
            for (const [emotion, entry] of Object.entries(_emotionMap)) {
                if (entry.file) fileSet.add(entry.file);
            }

            // 3. Fetch all expression files concurrently.
            await Promise.all([...fileSet].map(async (relPath) => {
                try {
                    const resp = await fetch(BASE_URL + relPath.replace(/^expressions\//, "expressions/"));
                    if (!resp.ok) {
                        console.warn(`[Anya/blend] Could not load ${relPath} (${resp.status})`);
                        return;
                    }
                    const data = await resp.json();
                    const params = new Map();
                    for (const p of (data.Parameters || [])) {
                        params.set(p.Id, p.Value);
                    }
                    _loadedExpressions.set(relPath, { params });
                } catch (err) {
                    console.warn(`[Anya/blend] Error loading ${relPath}:`, err);
                }
            }));

            // 4. Build param range cache from moc core model.
            _buildParamRanges(model);

            // 5. Register beforeModelUpdate hook.
            const internal = model.internalModel;
            if (internal && typeof internal.on === "function") {
                internal.on("beforeModelUpdate", () => _applyBlend(model));
            } else {
                // Fallback: use PIXI ticker if the event is unavailable in this cubism build.
                console.warn("[Anya/blend] beforeModelUpdate unavailable – falling back to ticker.");
                if (window.anya && window.anya.app) {
                    window.anya.app.ticker.add(() => _applyBlend(model));
                }
            }

            _ready = true;
            console.log(`[Anya/blend] Loaded ${_loadedExpressions.size} expression files. Ready.`);

            // Expose the public API on window.anya.
            _exposeAPI();

        } catch (err) {
            console.error("[Anya/blend] Initialisation failed:", err);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Parameter range helpers
    // ──────────────────────────────────────────────────────────────────────────

    function _buildParamRanges(model) {
        const core = model.internalModel && model.internalModel.coreModel;
        if (!core) return;
        try {
            const count = core.getParameterCount();
            for (let i = 0; i < count; i++) {
                const id  = core.getParameterId(i);
                const min = core.getParameterMinimumValue(i);
                const max = core.getParameterMaximumValue(i);
                _paramRanges[id] = { min, max };
            }
            console.log(`[Anya/blend] Cached ${count} parameter ranges.`);
        } catch (e) {
            console.warn("[Anya/blend] Could not build param ranges:", e);
        }
    }

    function _clamp(val, min, max) {
        return Math.max(min, Math.min(max, val));
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Weight resolution (expand blends into base emotions)
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Resolve a user-supplied weight map (which may contain blend names) into
     * a flat map of base-emotion → weight, then normalise to sum ≤ 1.
     * @param {Object<string,number>} raw
     * @returns {Object<string,number>}
     */
    function _resolveWeights(raw) {
        const resolved = {};

        function addWeight(emotion, amount) {
            if (_blendDefs[emotion]) {
                // Expand blend.
                const def = _blendDefs[emotion];
                const total = Object.values(def.weights).reduce((a, b) => a + b, 0) || 1;
                for (const [sub, w] of Object.entries(def.weights)) {
                    addWeight(sub, amount * (w / total));
                }
            } else {
                resolved[emotion] = (resolved[emotion] || 0) + amount;
            }
        }

        for (const [emotion, weight] of Object.entries(raw)) {
            addWeight(emotion, Math.max(0, weight));
        }

        // Normalise so the sum does not exceed 1.
        const total = Object.values(resolved).reduce((a, b) => a + b, 0);
        if (total > 1) {
            for (const k of Object.keys(resolved)) resolved[k] /= total;
        }

        return resolved;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Per-frame blend application
    // ──────────────────────────────────────────────────────────────────────────

    let _lastTs = 0;

    function _applyBlend(model) {
        if (!_ready) return;

        const now = performance.now();
        const dt  = _lastTs ? Math.min(now - _lastTs, 50) : 16; // cap at 50 ms
        _lastTs   = now;

        // 1. Ease current weights toward target weights.
        const alpha = 1 - Math.pow(1 - _easeAlphaPerMs, dt); // exponential ease
        const allKeys = new Set([...Object.keys(_targetWeights), ...Object.keys(_currentWeights)]);
        for (const k of allKeys) {
            const target  = _targetWeights[k]  || 0;
            const current = _currentWeights[k] || 0;
            const next = current + (target - current) * alpha;
            if (Math.abs(next) < 0.0001 && target === 0) {
                delete _currentWeights[k];
            } else {
                _currentWeights[k] = next;
            }
        }

        // 2. Accumulate weighted parameter deltas.
        const paramAccum = {}; // paramId → accumulated value

        for (const [emotion, weight] of Object.entries(_currentWeights)) {
            if (weight < 0.0001) continue;
            const entry = _emotionMap[emotion];
            if (!entry || !entry.file) continue;

            const expData = _loadedExpressions.get(entry.file);
            if (!expData) continue;

            const intensity = (entry.default_intensity ?? 1.0) * weight;
            for (const [paramId, value] of expData.params) {
                paramAccum[paramId] = (paramAccum[paramId] || 0) + value * intensity;
            }
        }

        // 3. Compute eye_open_cap for the current blend (animation.js reads this).
        let eyeCap = 1.0;
        let totalW = 0;
        for (const [emotion, weight] of Object.entries(_currentWeights)) {
            const entry = _emotionMap[emotion];
            if (!entry) continue;
            const cap = entry.eye_open_cap ?? 1.0;
            eyeCap = eyeCap + (cap - eyeCap) * weight; // weighted interpolation toward lowest cap
            totalW += weight;
        }
        if (window.anya) window.anya.eyeOpenCap = eyeCap;

        // 4. Apply to model parameters (clamp to range).
        const core = model.internalModel && model.internalModel.coreModel;
        if (!core) return;

        for (const [paramId, rawValue] of Object.entries(paramAccum)) {
            // Skip eye-open params – animation.js owns those (it will apply eyeOpenCap).
            if (paramId === "ParamEyeLOpen" || paramId === "ParamEyeROpen") continue;

            const range = _paramRanges[paramId];
            if (!range) continue;
            const clamped = _clamp(rawValue, range.min, range.max);
            try {
                core.setParameterValueById(paramId, clamped);
            } catch (_) { /* param may not exist in this moc version */ }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Public API
    // ──────────────────────────────────────────────────────────────────────────

    function _exposeAPI() {
        if (!window.anya) window.anya = {};

        /**
         * Set the current emotion blend.
         * @param {Object<string,number>} weights  e.g. { sad: 0.6, shy: 0.3 }
         * @param {number} [transitionMs=600]      ease duration in milliseconds
         */
        window.anya.setEmotion = function (weights, transitionMs = 600) {
            if (!_ready) {
                console.warn("[Anya/blend] setEmotion called before blend engine is ready.");
                return;
            }
            _targetWeights  = _resolveWeights(weights || {});
            // Convert transition duration to per-ms ease rate.
            // alpha = 1 - (1 - easeAlphaPerMs)^dt  →  target: 95 % done in transitionMs
            _easeAlphaPerMs = transitionMs > 0
                ? 1 - Math.pow(0.05, 1 / transitionMs)
                : 1; // instant
            console.log("[Anya/blend] setEmotion", _targetWeights, "transition:", transitionMs, "ms");
        };

        /**
         * Read the currently interpolated weights (useful for debugging).
         * @returns {Object<string,number>}
         */
        window.anya.getCurrentWeights = function () {
            return { ..._currentWeights };
        };

        /**
         * Return the loaded emotion map (for lab.html and external tooling).
         * @returns {Object}
         */
        window.anya.getEmotionMap = function () {
            return _emotionMap;
        };

        /** Expose loaded exp data for the lab blend mixer. */
        window.anya._loadedExpressions = _loadedExpressions;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Kick off once DOM is ready
    // ──────────────────────────────────────────────────────────────────────────

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", _boot);
    } else {
        _boot();
    }
})();
