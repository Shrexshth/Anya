// Live2D avatar controller – initialises PIXI, pixi-live2d-display, and audio lip-sync queue.
// Model is always fitted fully inside the canvas using originalWidth/Height from
// the internalModel (not model.width, which tracks the scaled size).

(async function () {
    // ── PIXI Application ──────────────────────────────────────────────────────
    // resizeTo: document.body keeps the canvas filling the viewport at all times.
    const app = new PIXI.Application({
        resizeTo: window,
        transparent: true,
        backgroundAlpha: 0,
        autoDensity: true,
        resolution: window.devicePixelRatio || 1,
        antialias: true,
    });

    // Attach the renderer view as the only child of <body>.
    document.body.appendChild(app.view);

    // ── Runtime config from Python ────────────────────────────────────────────
    let MODEL_SCALE   = 0.9; // fraction of the smaller canvas dimension
    let MODEL_OFFSET_Y = 0;  // CSS px downward shift from centre

    try {
        const configResp = await fetch("/api/config");
        if (configResp.ok) {
            const cfg = await configResp.json();
            if (cfg.fps_active)   PIXI.Ticker.shared.maxFPS = cfg.fps_active;
            if (cfg.model_scale  != null) MODEL_SCALE    = cfg.model_scale;
            if (cfg.model_offset_y != null) MODEL_OFFSET_Y = cfg.model_offset_y;
            if (cfg.debug) console.log("[Anya] /api/config:", cfg);
        }
    } catch (err) {
        console.warn("[Anya] Could not fetch /api/config:", err);
        PIXI.Ticker.shared.maxFPS = 30;
    }

    // ── QWebChannel bridge ────────────────────────────────────────────────────
    function setupWebChannel() {
        if (typeof QWebChannel !== "undefined" && window.qt && window.qt.webChannelTransport) {
            new QWebChannel(qt.webChannelTransport, function (channel) {
                window.bridge = channel.objects.bridge;
                console.log("[Anya] QWebChannel bridge connected.");
                if (window.anya && window.anya.model && window.bridge && window.bridge.modelReady) {
                    window.bridge.modelReady();
                }
            });
        }
    }
    setupWebChannel();

    // ── Load Live2D Model ─────────────────────────────────────────────────────
    const modelUrl = "/assets/model/Anya_model.model3.json";
    console.log("[Anya] Loading Live2D Model from:", modelUrl);

    try {
        const { Live2DModel } = PIXI.live2d;
        const model = await Live2DModel.from(modelUrl, { autoInteract: false });

        app.stage.addChild(model);

        // Set anchor once – (0.5, 0.5) means the model pivot is its centre.
        model.anchor.set(0.5, 0.5);

        // ── Fit model to canvas ───────────────────────────────────────────────
        function fitModel() {
            if (!model || !model.internalModel) return;

            const viewW = app.screen.width;
            const viewH = app.screen.height;

            const baseW = model.internalModel.originalWidth  || model.internalModel.width;
            const baseH = model.internalModel.originalHeight || model.internalModel.height;

            const scale = Math.min(viewW / baseW, viewH / baseH) * MODEL_SCALE;

            model.scale.set(scale);
            model.x = viewW / 2;
            model.y = viewH / 2 + MODEL_OFFSET_Y;

            console.log("[Anya] fitModel:", {
                viewW, viewH, baseW, baseH, scale,
                x: model.x, y: model.y,
                dpr: window.devicePixelRatio,
                rendererW: app.renderer.width,
                rendererH: app.renderer.height,
            });
        }

        fitModel();

        // ── Single AudioContext & LipSync Subsystem (WP4) ──────────────────────
        let _audioCtx = null;
        let _analyser = null;
        let _timeDomainData = null;
        let _currentSource = null;
        let _audioQueue = []; // array of { url, utteranceId, isLast }
        let _activeClip = null;
        let _isPlayingAudio = false;

        let _smoothedRms = 0;
        let _targetMouth = 0;
        let _currentMouth = 0;

        function getAudioContext() {
            if (!_audioCtx) {
                const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
                _audioCtx = new AudioCtxClass();
                _analyser = _audioCtx.createAnalyser();
                _analyser.fftSize = 512;
                _analyser.smoothingTimeConstant = 0.25;
                _timeDomainData = new Float32Array(_analyser.fftSize);
                _analyser.connect(_audioCtx.destination);
            }
            if (_audioCtx.state === "suspended") {
                _audioCtx.resume().catch(() => {});
            }
            return _audioCtx;
        }

        // Resume AudioContext on user interaction
        window.addEventListener("pointerdown", () => {
            if (_audioCtx && _audioCtx.state === "suspended") {
                _audioCtx.resume().catch(() => {});
            }
        });

        function _setMouthParam(val) {
            if (!model || !model.internalModel || !model.internalModel.coreModel) return;
            try {
                model.internalModel.coreModel.setParameterValueById("ParamMouthOpenY", val);
            } catch (_) {}
        }

        function _updateLipSync() {
            if (_isPlayingAudio && _analyser && _timeDomainData) {
                _analyser.getFloatTimeDomainData(_timeDomainData);
                let sumSq = 0;
                for (let i = 0; i < _timeDomainData.length; i++) {
                    const sample = _timeDomainData[i];
                    sumSq += sample * sample;
                }
                const rms = Math.sqrt(sumSq / _timeDomainData.length);
                _smoothedRms = _smoothedRms * 0.65 + rms * 0.35;

                // Lip sync mapping: silence floor ~0.015, normal speech opens mouth ~0.2 to 0.9
                if (_smoothedRms < 0.015) {
                    _targetMouth = 0.0;
                } else {
                    const norm = (_smoothedRms - 0.015) / (0.18 - 0.015);
                    const clampedNorm = Math.max(0, Math.min(1, norm));
                    _targetMouth = 0.20 + clampedNorm * (0.90 - 0.20);
                }
                _currentMouth += (_targetMouth - _currentMouth) * 0.55;
            } else {
                _smoothedRms = 0;
                _targetMouth = 0;
                _currentMouth += (0 - _currentMouth) * 0.65;
                if (_currentMouth < 0.01) {
                    _currentMouth = 0;
                }
            }

            _setMouthParam(_currentMouth);
        }

        app.ticker.add(_updateLipSync);

        if (model.internalModel && typeof model.internalModel.on === "function") {
            model.internalModel.on("beforeModelUpdate", () => {
                _setMouthParam(_currentMouth);
            });
        }

        async function _playNextClip() {
            if (_audioQueue.length === 0) {
                _isPlayingAudio = false;
                _activeClip = null;
                _currentMouth = 0;
                _targetMouth = 0;
                _setMouthParam(0);
                if (window.bridge && typeof window.bridge.set_speaking === "function") {
                    window.bridge.set_speaking(false);
                }
                if (window.anya && typeof window.anya.setSpeaking === "function") {
                    window.anya.setSpeaking(false);
                }
                return;
            }

            const clip = _audioQueue.shift();
            _activeClip = clip;
            _isPlayingAudio = true;

            if (window.bridge && typeof window.bridge.set_speaking === "function") {
                window.bridge.set_speaking(true);
            }
            if (window.anya && typeof window.anya.setSpeaking === "function") {
                window.anya.setSpeaking(true);
            }

            const ctx = getAudioContext();
            try {
                const resp = await fetch(clip.url);
                if (!resp.ok) {
                    console.warn(`[Anya/audio] Failed to fetch clip from ${clip.url} (${resp.status})`);
                    _handleClipEnded(clip);
                    return;
                }
                const arrayBuffer = await resp.arrayBuffer();
                const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

                // Check if playback was stopped while decoding
                if (!_isPlayingAudio || _activeClip !== clip) {
                    return;
                }

                const source = ctx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(_analyser);
                _currentSource = source;

                source.onended = () => {
                    if (_currentSource === source) {
                        _currentSource = null;
                        _handleClipEnded(clip);
                    }
                };

                source.start(0);
                console.log(`[Anya/audio] Playing clip: ${clip.url} (isLast=${clip.isLast})`);
            } catch (err) {
                console.error("[Anya/audio] Error playing audio clip:", err);
                _handleClipEnded(clip);
            }
        }

        function _handleClipEnded(clip) {
            // Close mouth between clips
            _currentMouth = 0;
            _targetMouth = 0;
            _smoothedRms = 0;
            _setMouthParam(0);

            if (clip && clip.isLast) {
                const utteranceId = clip.utteranceId || "";
                console.log(`[Anya/audio] Utterance finished (id: ${utteranceId})`);
                if (window.bridge && typeof window.bridge.audioEnded === "function") {
                    window.bridge.audioEnded(utteranceId);
                }
            }

            if (_audioQueue.length > 0) {
                _playNextClip();
            } else {
                _isPlayingAudio = false;
                _activeClip = null;
                if (window.bridge && typeof window.bridge.set_speaking === "function") {
                    window.bridge.set_speaking(false);
                }
                if (window.anya && typeof window.anya.setSpeaking === "function") {
                    window.anya.setSpeaking(false);
                }
            }
        }

        function queueAudio(url, utteranceId = "", isLast = true) {
            getAudioContext();
            _audioQueue.push({ url, utteranceId, isLast: !!isLast });
            if (!_isPlayingAudio) {
                _playNextClip();
            }
        }

        function playSound(name) {
            const filename = (name && name.endsWith(".wav")) ? name : `${name}.wav`;
            const url = `/assets/sounds/${filename}`;
            console.log(`[Anya/audio] playSound requested: ${name} -> ${url}`);
            queueAudio(url, `sound_${name}`, true);
        }

        function stopSpeech() {
            _audioQueue = [];
            if (_currentSource) {
                try {
                    _currentSource.onended = null;
                    _currentSource.stop(0);
                } catch (_) {}
                _currentSource = null;
            }
            _isPlayingAudio = false;
            _activeClip = null;
            _currentMouth = 0;
            _targetMouth = 0;
            _smoothedRms = 0;
            _setMouthParam(0);
            if (window.bridge && typeof window.bridge.set_speaking === "function") {
                window.bridge.set_speaking(false);
            }
            if (window.anya && typeof window.anya.setSpeaking === "function") {
                window.anya.setSpeaking(false);
            }
            console.log("[Anya/audio] stopSpeech executed – queue cleared, mouth closed.");
        }

        // ── window.anya interface ─────────────────────────────────────────────
        window.anya = {
            model,
            app,
            fitModel,
            eyeOpenCap: 1.0,

            // Audio & LipSync APIs (WP4)
            getAudioContext,
            queueAudio,
            playAudio: queueAudio,
            playSound,
            stopSpeech,

            /** Compatibility shim – prefer anya.setEmotion() from blend.js. */
            setExpression(name) {
                if (!name) { this.clearExpression(); return; }
                try {
                    if (typeof model.expression === "function") {
                        model.expression(name);
                        console.log("[Anya] setExpression (compat):", name);
                    }
                } catch (e) {
                    console.error("[Anya] setExpression failed:", name, e);
                }
            },

            clearExpression() {
                try {
                    const expMgr = model.internalModel
                        && model.internalModel.motionManager
                        && model.internalModel.motionManager.expressionManager;
                    if (expMgr && typeof expMgr.resetExpression === "function") {
                        expMgr.resetExpression();
                    } else if (typeof model.expression === "function") {
                        model.expression("");
                    }
                    console.log("[Anya] Expression cleared.");
                } catch (e) {
                    console.error("[Anya] clearExpression failed:", e);
                }
            },
        };

        // ── Notify Python bridge ──────────────────────────────────────────────
        if (window.bridge && typeof window.bridge.modelReady === "function") {
            window.bridge.modelReady();
        }

        // ── Notify blend.js / animation.js ────────────────────────────────────
        window.dispatchEvent(new CustomEvent("anya:modelReady", { detail: { model } }));

        // ── Re-fit on resize ──────────────────────────────────────────────────
        app.renderer.on("resize", fitModel);

    } catch (error) {
        console.error("[Anya] Critical error loading Live2D model:", error);
    }
})();
