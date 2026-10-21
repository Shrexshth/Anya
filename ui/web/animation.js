/**
 * animation.js – Automatic blink, breathing, body sway, head tilt, eye tracking,
 * random glances, idle fidgets, react() and setSleep() APIs.
 *
 * Reads:
 *   window.anya.model        – Live2D model
 *   window.anya.app          – PIXI Application
 *   window.anya.eyeOpenCap   – set each frame by blend.js (1.0 = fully open)
 *
 * Exposes on window.anya:
 *   anya.react(kind)          – 'blush' | 'shake' | 'droop' | 'bounce'
 *   anya.setSleep(bool)       – sleep mode (eye closed, slow breath, 10 FPS)
 *   anya.setCursorNorm(x, y)  – called by bridge with cursor in [0,1]²
 *
 * FPS management:
 *   Active FPS (FPS_ACTIVE) is read from window.anya.app.ticker.maxFPS.
 *   After 10 s of no speech & no mouse movement the ticker drops to FPS_IDLE (10).
 *   On any mouse move or bridge.setSpeaking(true) it snaps back.
 */

(function () {
    "use strict";

    // ──────────────────────────────────────────────────────────────────────────
    // Config constants (all tuneable; nothing hard-coded)
    // ──────────────────────────────────────────────────────────────────────────
    const FPS_IDLE             = 10;
    const IDLE_TIMEOUT_MS      = 10_000;   // 10 s

    const BLINK_INTERVAL_MIN   = 2.5;      // seconds between blink starts
    const BLINK_INTERVAL_MAX   = 5.5;
    const BLINK_CLOSE_SPEED    = 0.18;     // fraction per frame (at 30 FPS)
    const BLINK_OPEN_SPEED     = 0.12;

    const BREATH_PERIOD_S      = 3.8;      // seconds per breath cycle
    const BREATH_AMPLITUDE     = 0.4;      // ParamBreath peak value
    const BREATH_SLEEP_PERIOD  = 6.0;
    const BREATH_SLEEP_AMP     = 0.25;

    const SWAY_PERIOD_S        = 9.0;      // body sway cycle
    const SWAY_AMPLITUDE_X     = 0.08;     // ParamBodyAngleX peak
    const SWAY_AMPLITUDE_Z     = 0.06;     // ParamBodyAngleZ peak

    const TILT_FOLLOW_SPEED    = 0.03;     // how fast head chases cursor
    const EYE_FOLLOW_SPEED     = 0.06;     // how fast eyes chase cursor
    const EYE_FOLLOW_SCALE     = 0.6;      // max eyeball deflection (0-1)

    const GLANCE_INTERVAL_MIN  = 8;        // seconds between random glances
    const GLANCE_INTERVAL_MAX  = 18;
    const GLANCE_DURATION_S    = 1.2;
    const GLANCE_STRENGTH      = 0.5;

    const FIDGET_INTERVAL_MIN  = 15;       // seconds between idle fidgets
    const FIDGET_INTERVAL_MAX  = 35;
    const FIDGET_DURATION_S    = 1.8;

    const REACT_BLUSH_DURATION = 2500;     // ms
    const REACT_SHAKE_AMP      = 6;        // degrees
    const REACT_SHAKE_CYCLES   = 3;
    const REACT_SHAKE_MS       = 400;
    const REACT_BOUNCE_AMP     = 8;        // px
    const REACT_BOUNCE_MS      = 500;
    const REACT_DROOP_AMOUNT   = -0.5;
    const REACT_DROOP_MS       = 800;

    // ──────────────────────────────────────────────────────────────────────────
    // State
    // ──────────────────────────────────────────────────────────────────────────

    let _model   = null;
    let _core    = null;
    let _app     = null;
    let _fpsActive = 30;

    // Blink state machine: 'open' | 'closing' | 'opening'
    let _blinkState     = "open";
    let _blinkValue     = 1.0;   // current eye openness 0-1
    let _nextBlinkTime  = 0;     // epoch ms

    // Breath
    let _breathPhase = 0;

    // Body sway
    let _swayPhase = 0;

    // Head & eye tracking
    let _cursorNormX = 0.5;
    let _cursorNormY = 0.5;
    let _headX = 0, _headY = 0;  // current interpolated head angle
    let _eyeX  = 0, _eyeY  = 0;  // current interpolated eyeball pos

    // Glance
    let _glanceActive  = false;
    let _glanceStartMs = 0;
    let _glanceTargetX = 0;
    let _glanceTargetY = 0;
    let _nextGlanceTime = 0;

    // Fidget
    let _fidgetActive  = false;
    let _fidgetStartMs = 0;
    let _fidgetParamId = "ParamBodyAngleZ";
    let _fidgetAmp     = 0.12;
    let _nextFidgetTime = 0;

    // React transients
    let _blushActive  = false;
    let _blushEndMs   = 0;
    let _shakeActive  = false;
    let _shakeStartMs = 0;
    let _bounceActive = false;
    let _bounceStartMs = 0;
    let _droopActive  = false;
    let _droopStartMs = 0;

    // Sleep
    let _isSleeping = false;

    // Idle FPS management
    let _lastActivityMs = 0;
    let _isIdle = false;
    let _isSpeaking = false;

    // ──────────────────────────────────────────────────────────────────────────
    // Boot
    // ──────────────────────────────────────────────────────────────────────────

    function _boot() {
        if (window.anya && window.anya.model) {
            _setup(window.anya.model, window.anya.app);
        } else {
            window.addEventListener("anya:modelReady", (ev) => {
                _setup(ev.detail.model, window.anya && window.anya.app);
            });
        }
    }

    function _setup(model, app) {
        _model   = model;
        _core    = model.internalModel && model.internalModel.coreModel;
        _app     = app;
        _fpsActive = (app && app.ticker && app.ticker.maxFPS) || 30;

        _lastActivityMs  = performance.now();
        _nextBlinkTime   = _now() + _randBetween(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX);
        _nextGlanceTime  = _now() + _randBetween(GLANCE_INTERVAL_MIN, GLANCE_INTERVAL_MAX);
        _nextFidgetTime  = _now() + _randBetween(FIDGET_INTERVAL_MIN, FIDGET_INTERVAL_MAX);

        // Register per-frame tick.
        if (app && app.ticker) {
            app.ticker.add(_tick);
        }

        // Mouse-move listener → update activity & cursor position.
        window.addEventListener("mousemove", (ev) => {
            _wakeActivity();
            _cursorNormX = ev.clientX / window.innerWidth;
            _cursorNormY = ev.clientY / window.innerHeight;
        });

        _exposeAPI();
        console.log("[Anya/animation] Ready.");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Per-frame tick
    // ──────────────────────────────────────────────────────────────────────────

    function _tick(deltaTime) {
        if (!_core) return;

        const nowS  = _now();
        const nowMs = performance.now();
        const dt    = deltaTime / 60; // PIXI deltaTime is frames at 60 FPS

        // Idle FPS management.
        _manageIdleFPS(nowMs);

        // Blend all layers.
        _tickBlink(nowS, dt);
        _tickBreath(nowS);
        _tickSway(nowS);
        _tickHeadAndEyes(dt);
        _tickGlance(nowS);
        _tickFidget(nowS);
        _tickReacts(nowMs);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Idle FPS management
    // ──────────────────────────────────────────────────────────────────────────

    function _manageIdleFPS(nowMs) {
        const idle = (nowMs - _lastActivityMs) > IDLE_TIMEOUT_MS && !_isSpeaking;
        if (idle !== _isIdle) {
            _isIdle = idle;
            if (_app && _app.ticker) {
                _app.ticker.maxFPS = idle ? FPS_IDLE : _fpsActive;
            }
            console.log("[Anya/animation] FPS →", idle ? FPS_IDLE : _fpsActive);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Blink
    // ──────────────────────────────────────────────────────────────────────────

    function _tickBlink(nowS, dt) {
        const eyeCap = (window.anya && window.anya.eyeOpenCap != null)
            ? window.anya.eyeOpenCap : 1.0;

        if (_isSleeping) {
            _blinkValue = Math.max(0, _blinkValue - BLINK_CLOSE_SPEED * 2);
        } else {
            switch (_blinkState) {
                case "open":
                    if (nowS >= _nextBlinkTime) {
                        _blinkState    = "closing";
                        _nextBlinkTime = nowS + _randBetween(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX);
                    }
                    break;

                case "closing":
                    _blinkValue -= BLINK_CLOSE_SPEED;
                    if (_blinkValue <= 0) {
                        _blinkValue = 0;
                        _blinkState = "opening";
                    }
                    break;

                case "opening":
                    _blinkValue += BLINK_OPEN_SPEED;
                    if (_blinkValue >= 1) {
                        _blinkValue = 1;
                        _blinkState = "open";
                    }
                    break;
            }
        }

        // Apply eye_open_cap on top of blink.
        const finalEyeOpen = _blinkValue * eyeCap;
        _setParam("ParamEyeLOpen", finalEyeOpen);
        _setParam("ParamEyeROpen", finalEyeOpen);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Breathing
    // ──────────────────────────────────────────────────────────────────────────

    function _tickBreath(nowS) {
        const period = _isSleeping ? BREATH_SLEEP_PERIOD : BREATH_PERIOD_S;
        const amp    = _isSleeping ? BREATH_SLEEP_AMP    : BREATH_AMPLITUDE;
        const phase  = (nowS % period) / period * Math.PI * 2;
        const breath = (Math.sin(phase) * 0.5 + 0.5) * amp;
        _setParam("ParamBreath", breath);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Body sway
    // ──────────────────────────────────────────────────────────────────────────

    function _tickSway(nowS) {
        if (_isSleeping) return;
        const phaseX = (nowS % SWAY_PERIOD_S)      / SWAY_PERIOD_S      * Math.PI * 2;
        const phaseZ = (nowS % (SWAY_PERIOD_S * 1.3)) / (SWAY_PERIOD_S * 1.3) * Math.PI * 2;
        _setParam("ParamBodyAngleX", Math.sin(phaseX) * SWAY_AMPLITUDE_X);
        _setParam("ParamBodyAngleZ", Math.sin(phaseZ) * SWAY_AMPLITUDE_Z);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Head tilt & eye follow (cursor driven)
    // ──────────────────────────────────────────────────────────────────────────

    function _tickHeadAndEyes(dt) {
        if (_isSleeping) return;

        // Map cursor [0,1] → [-1,+1]
        const targetX = (_cursorNormX - 0.5) * 2;
        const targetY = (_cursorNormY - 0.5) * 2;

        // Ease head angles.
        const alpha = Math.min(1, TILT_FOLLOW_SPEED * 60 * dt);
        _headX += (targetX * 30 - _headX) * alpha;
        _headY += (-targetY * 15 - _headY) * alpha;

        // Ease eyeball position.
        const eyeAlpha = Math.min(1, EYE_FOLLOW_SPEED * 60 * dt);
        _eyeX += (targetX * EYE_FOLLOW_SCALE - _eyeX) * eyeAlpha;
        _eyeY += (-targetY * EYE_FOLLOW_SCALE - _eyeY) * eyeAlpha;

        // Skip head params if physics group owns them; just do the angle group.
        _setParam("ParamAngleX", _headX);
        _setParam("ParamAngleY", _headY);
        _setParam("ParamEyeBallX", _eyeX);
        _setParam("ParamEyeBallY", _eyeY);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Random glances
    // ──────────────────────────────────────────────────────────────────────────

    function _tickGlance(nowS) {
        if (_isSleeping) return;

        if (!_glanceActive && nowS >= _nextGlanceTime) {
            _glanceActive   = true;
            _glanceStartMs  = performance.now();
            _glanceTargetX  = _randBetween(-GLANCE_STRENGTH, GLANCE_STRENGTH);
            _glanceTargetY  = _randBetween(-GLANCE_STRENGTH * 0.5, GLANCE_STRENGTH * 0.5);
            _nextGlanceTime = nowS + _randBetween(GLANCE_INTERVAL_MIN, GLANCE_INTERVAL_MAX);
        }

        if (_glanceActive) {
            const elapsed = (performance.now() - _glanceStartMs) / 1000;
            const t = Math.min(elapsed / GLANCE_DURATION_S, 1);
            const env = t < 0.5 ? t * 2 : (1 - t) * 2; // triangle envelope
            const offset = env * _glanceTargetX;
            // Add glance offset on top of cursor-driven values.
            _setParam("ParamEyeBallX", _clamp(_eyeX + offset, -1, 1));
            _setParam("ParamEyeBallY", _clamp(_eyeY + _glanceTargetY * env, -1, 1));
            if (t >= 1) _glanceActive = false;
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Idle fidgets
    // ──────────────────────────────────────────────────────────────────────────

    const _FIDGET_PARAMS = [
        { id: "ParamBodyAngleZ",  amp: 0.15 },
        { id: "ParamAngleZ",      amp: 4.0  },
        { id: "ParamBodyAngleX",  amp: 0.1  },
    ];

    function _tickFidget(nowS) {
        if (_isSleeping) return;

        if (!_fidgetActive && nowS >= _nextFidgetTime) {
            _fidgetActive   = true;
            _fidgetStartMs  = performance.now();
            const choice    = _FIDGET_PARAMS[Math.floor(Math.random() * _FIDGET_PARAMS.length)];
            _fidgetParamId  = choice.id;
            _fidgetAmp      = choice.amp;
            _nextFidgetTime = nowS + _randBetween(FIDGET_INTERVAL_MIN, FIDGET_INTERVAL_MAX);
        }

        if (_fidgetActive) {
            const elapsed = (performance.now() - _fidgetStartMs) / 1000;
            const t = Math.min(elapsed / FIDGET_DURATION_S, 1);
            const env = Math.sin(t * Math.PI) * Math.sin(t * Math.PI * 4);
            _setParam(_fidgetParamId, env * _fidgetAmp);
            if (t >= 1) _fidgetActive = false;
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // React transients
    // ──────────────────────────────────────────────────────────────────────────

    function _tickReacts(nowMs) {
        // Blush: handled by blend.js (Param14), we just need to unset it.
        if (_blushActive && nowMs > _blushEndMs) {
            _blushActive = false;
            if (window.anya && window.anya.setEmotion) {
                window.anya.setEmotion({}, 800); // fade out
            }
        }

        // Shake: oscillate ParamAngleZ.
        if (_shakeActive) {
            const elapsed = nowMs - _shakeStartMs;
            const t = elapsed / REACT_SHAKE_MS;
            if (t < 1) {
                const osc = Math.sin(t * Math.PI * 2 * REACT_SHAKE_CYCLES) * (1 - t);
                _setParam("ParamAngleZ", osc * REACT_SHAKE_AMP);
            } else {
                _setParam("ParamAngleZ", 0);
                _shakeActive = false;
            }
        }

        // Bounce: move model.y up then back.
        if (_bounceActive && _model) {
            const elapsed = nowMs - _bounceStartMs;
            const t = elapsed / REACT_BOUNCE_MS;
            if (t < 1) {
                const offset = Math.sin(t * Math.PI) * REACT_BOUNCE_AMP;
                _model.y = (_model._baseY || _model.y) - offset;
            } else {
                if (_model._baseY != null) _model.y = _model._baseY;
                _bounceActive = false;
            }
        }

        // Droop: lower brows temporarily.
        if (_droopActive) {
            const elapsed = nowMs - _droopStartMs;
            const t = Math.min(elapsed / REACT_DROOP_MS, 1);
            const env = t < 0.5 ? t * 2 : (1 - t) * 2;
            _setParam("ParamBrowLY", env * REACT_DROOP_AMOUNT);
            _setParam("ParamBrowRY", env * REACT_DROOP_AMOUNT);
            if (t >= 1) _droopActive = false;
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────────

    function _setParam(id, value) {
        if (!_core) return;
        try { _core.setParameterValueById(id, value); } catch (_) {}
    }

    function _now() { return performance.now() / 1000; }

    function _randBetween(min, max) {
        return min + Math.random() * (max - min);
    }

    function _clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

    function _wakeActivity() {
        _lastActivityMs = performance.now();
        if (_isIdle) {
            _isIdle = false;
            if (_app && _app.ticker) _app.ticker.maxFPS = _fpsActive;
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Public API
    // ──────────────────────────────────────────────────────────────────────────

    function _exposeAPI() {
        if (!window.anya) window.anya = {};

        /**
         * Trigger a short reactive animation.
         * @param {'blush'|'shake'|'droop'|'bounce'} kind
         */
        window.anya.react = function (kind) {
            switch (kind) {
                case "blush":
                    _blushActive = true;
                    _blushEndMs  = performance.now() + REACT_BLUSH_DURATION;
                    if (window.anya.setEmotion) {
                        window.anya.setEmotion({ romantic: 0.6 }, 300);
                    }
                    break;

                case "shake":
                    _shakeActive  = true;
                    _shakeStartMs = performance.now();
                    break;

                case "droop":
                    _droopActive  = true;
                    _droopStartMs = performance.now();
                    break;

                case "bounce":
                    if (_model) _model._baseY = _model.y;
                    _bounceActive  = true;
                    _bounceStartMs = performance.now();
                    break;

                default:
                    console.warn("[Anya/animation] Unknown react kind:", kind);
            }
        };

        /**
         * Enter or exit sleep mode.
         * @param {boolean} bool
         */
        window.anya.setSleep = function (bool) {
            _isSleeping = !!bool;
            if (_app && _app.ticker) {
                _app.ticker.maxFPS = bool ? FPS_IDLE : _fpsActive;
            }
            console.log("[Anya/animation] Sleep:", bool);
        };

        /**
         * Called by bridge.py every 200 ms with normalised cursor coords.
         * No polling in JS – Python pushes.
         * @param {number} nx  0‑1 (left→right)
         * @param {number} ny  0‑1 (top→bottom)
         */
        window.anya.setCursorNorm = function (nx, ny) {
            _cursorNormX = nx;
            _cursorNormY = ny;
            _wakeActivity();
        };

        /**
         * Called by bridge.py to mark speech start/end (affects idle timer).
         * @param {boolean} speaking
         */
        window.anya.setSpeaking = function (speaking) {
            _isSpeaking = speaking;
            if (speaking) _wakeActivity();
        };
    }

    // Boot.
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", _boot);
    } else {
        _boot();
    }
})();
