// =====================================================================
// Chat-session finite state machine (refactor 2026-05-20).
//
// The chat UI used to expose six overlapping buttons (開始對話/停止對話/
// 開始錄音/停止錄音/強制送出/取消) plus ad-hoc state flags
// (`isRecording`, `isThinking`, `isStartingRecording`). The button state
// drifted from the actual WS/mic state — after `vad_commit`, the
// recordBtn label said "🎤 開始錄音" but the status pill still said
// "Recording...", so users had no idea what was happening.
//
// This rewrite collapses everything onto ONE primary button whose label
// is a pure function of the FSM state, and one auto-continue checkbox.
//
// STATES
//   IDLE         — page just loaded, no WS, no mic. Primary disabled.
//   CONNECTING   — ws.readyState === CONNECTING. Primary disabled.
//   READY        — WS open, awaiting user gesture. Primary → "開始說話".
//   LISTENING    — mic streaming, server VAD active. Primary → "送出".
//   THINKING     — server has the audio, ASR/LLM running, no TTS yet.
//                  Primary → "中斷".
//   SPEAKING     — TTS audio playing on client. Primary → "中斷".
//
// LEGAL TRANSITIONS (trigger → next state)
//   IDLE         → CONNECTING        connect() on page load
//   CONNECTING   → READY             ws.onopen + worklet ready
//   CONNECTING   → IDLE              ws.onerror | ws.onclose
//   READY        → LISTENING         primary click | Space key
//   LISTENING    → THINKING          primary click ('送出')
//                                    | vad_commit msg from server
//   LISTENING    → READY             Esc key (cancel without commit)
//   THINKING     → SPEAKING          first tts_start msg
//   THINKING     → READY             llm_done (no TTS played)
//                                    | llm_cancelled | llm_error
//   SPEAKING     → READY             llm_done AND all tts_done received
//                                    AND auto-continue is OFF
//   SPEAKING     → LISTENING         llm_done AND all tts_done received
//                                    AND auto-continue is ON
//   THINKING     → LISTENING         llm_done with empty text AND
//                                    auto-continue is ON (no TTS path)
//   THINKING|SPEAKING → READY        primary click ('中斷') sends cancel
//   ANY (except IDLE)  → IDLE        ws.onclose
//
// Auto-continue is default ON. The user can disable it via the
// "自動繼續對話" checkbox; that puts them back in classic
// push-to-talk mode.
// =====================================================================

// System status poller moved to _status_bar.js (RFC_M6 Phase 0-pre
// review #28). This page registers a SYS_ON_UPDATE hook that lets the
// chat-specific FSM gate the primary button when training is running
// OR TTS/ASR aren't ready.
window.SYS_ON_UPDATE = window.SYS_ON_UPDATE || [];
window.SYS_ON_UPDATE.push(function () {
    if (typeof renderUI === 'function') renderUI();
});

window.onerror = function(msg, url, line, col, error) {
    log('GLOBAL ERROR: ' + msg + ' at line ' + line + ' col ' + col);
    return false;
};
window.UI_VERSION = '2026-05-20-v27-fsm-refactor';
console.log('UI Version: ' + window.UI_VERSION);
const _uiVerEl = document.getElementById('uiVersion');
if (_uiVerEl) _uiVerEl.textContent = '[v27]';

const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/asr';

// FSM state constants. Exposed on `window` so contract tests can grep.
const STATE = Object.freeze({
    IDLE: 'IDLE',
    CONNECTING: 'CONNECTING',
    READY: 'READY',
    LISTENING: 'LISTENING',
    THINKING: 'THINKING',
    SPEAKING: 'SPEAKING',
});
window.CHAT_STATE = STATE;

let state = STATE.IDLE;
let ws = null;
let audioContext = null;
let scriptProcessor = null;
let micSource = null;  // MediaStreamAudioSourceNode — held so cleanup can disconnect it
let utteranceId = null;
let ttsText = '';
let ttsEmotion = null;
let recordingStream = null;
let accumulatedChunks = [];
let selectedVersionId = null;
let ttsStartCount = 0;
let ttsDoneCount = 0;
let llmDoneSeen = false;
let isStartingRecording = false;
// Set to true by the AudioWorklet when its ring buffer drains from
// non-empty → empty. Reset to false on every tts_start (new audio
// coming). Used by isResponseDone() — server's `tts_done` fires when
// it finishes sending PCM, but the client still has ~5s of audio
// buffered locally. Without waiting for actual drain, auto-continue
// would re-arm the mic mid-playback and the mobile OS would duck
// the speaker output (user report 2026-05-26).
let audioDrained = true;

// Single cleanup helper for mic teardown — used by all the sites that
// stop recording (vad_commit, ws.onclose, stopRecordingAndSend,
// onPrimaryClick LISTENING branch). Previously each site did
// scriptProcessor.disconnect() + recordingStream.tracks.stop() but
// none disconnected the MediaStreamAudioSourceNode (which kept a
// dangling reference to the stream and prevented Android/iOS from
// releasing the mic indicator after stop). Centralizing avoids future
// drift between sites.
function teardownMic() {
    if (scriptProcessor) {
        try { scriptProcessor.disconnect(); } catch (_) {}
        scriptProcessor = null;
    }
    if (micSource) {
        try { micSource.disconnect(); } catch (_) {}
        micSource = null;
    }
    if (recordingStream) {
        try { recordingStream.getTracks().forEach(t => t.stop()); } catch (_) {}
        recordingStream = null;
    }
}

let audioWorkletNode = null;
let workletInitialized = false;
let workletNodeCreated = false;

async function initAudioWorklet() {
    if (workletInitialized) {
        log('Worklet already initialized');
        return true;
    }
    log('initAudioWorklet: starting');

    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
        log('Created AudioContext');
    }
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
        log('Resumed AudioContext');
    }
    if (!audioContext.audioWorklet) {
        log('AudioWorklet not supported');
        return false;
    }
    try {
        const workletCode = `
            class PCMPlayer extends AudioWorkletProcessor {
                constructor() {
                    super();
                    this.ringBuffer = new Float32Array(24000 * 20);
                    this.writePos = 0;
                    this.readPos = 0;
                    this.samplesInBuffer = 0;
                    this.isPlaying = true;
                    this.pendingFlush = false;
                    this.port.onmessage = (e) => {
                        if (e.data.type === 'pcm') {
                            const int16Data = new Int16Array(e.data.buffer);
                            let dropped = 0;
                            for (let i = 0; i < int16Data.length; i++) {
                                this.ringBuffer[this.writePos] = int16Data[i] / 32768.0;
                                this.writePos = (this.writePos + 1) % this.ringBuffer.length;
                                this.samplesInBuffer++;
                                if (this.samplesInBuffer > this.ringBuffer.length) {
                                    this.readPos = (this.readPos + 1) % this.ringBuffer.length;
                                    this.samplesInBuffer--;
                                    dropped++;
                                }
                            }
                            if (dropped > 0) {
                                this.port.postMessage({ type: 'log', msg: 'BUF DROPPED ' + dropped + ' samples, buf=' + this.samplesInBuffer });
                            }
                        } else if (e.data.type === 'flush') {
                            // Soft flush: only clear if buffer is small.
                            // Used on tts_start to avoid wiping ongoing
                            // playback if a new sentence overlaps.
                            if (this.samplesInBuffer < 24000) {
                                this.writePos = 0;
                                this.readPos = 0;
                                this.samplesInBuffer = 0;
                            }
                        } else if (e.data.type === 'abort') {
                            // Hard flush: unconditionally clear. Used on
                            // user-initiated cancel (中斷 button) to kill
                            // mid-sentence TTS playback so the mic can
                            // re-arm without the OS ducking the still-
                            // playing audio (user report 2026-05-26 —
                            // 中斷 left mic indicator on).
                            this.writePos = 0;
                            this.readPos = 0;
                            this.samplesInBuffer = 0;
                        } else if (e.data.type === 'stop') {
                            this.isPlaying = false;
                        } else if (e.data.type === 'log') {
                            console.log('Worklet: ' + e.data.msg);
                        }
                    };
                }
                process(inputs, outputs, parameters) {
                    const output = outputs[0];
                    if (!output || output.length === 0) return true;
                    const out = output[0];
                    if (!out) return true;
                    if (!this.isPlaying) {
                        for (let i = 0; i < out.length; i++) out[i] = 0;
                        return true;
                    }
                    const wasNonEmpty = this.samplesInBuffer > 0;
                    for (let i = 0; i < out.length; i++) {
                        if (this.samplesInBuffer > 0) {
                            out[i] = this.ringBuffer[this.readPos];
                            this.readPos = (this.readPos + 1) % this.ringBuffer.length;
                            this.samplesInBuffer--;
                        } else {
                            out[i] = 0;
                        }
                    }
                    // Notify main thread once when the buffer transitions
                    // from non-empty → empty. Lets the auto-continue logic
                    // wait for actual audio drain (vs just tts_done from
                    // the server, which fires before client-side playback
                    // finishes). User reported 2026-05-26 mid-TTS ducking
                    // even on single-sentence responses — that was the
                    // mic re-arming after tts_done while ~5s of buffered
                    // audio was still playing.
                    if (wasNonEmpty && this.samplesInBuffer === 0) {
                        this.port.postMessage({ type: 'drained' });
                    }
                    return true;
                }
            }
            registerProcessor('pcm-player', PCMPlayer);
        `;
        const blob = new Blob([workletCode], { type: 'application/javascript' });
        const workletUrl = URL.createObjectURL(blob);
        log('Adding worklet module...');
        await audioContext.audioWorklet.addModule(workletUrl);
        workletInitialized = true;
        log('Worklet module registered, AudioContext state: ' + audioContext.state);
        return true;
    } catch (e) {
        log('Worklet init error: ' + e.name);
        return false;
    }
}

async function ensureWorklet() {
    if (workletNodeCreated && audioWorkletNode) {
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
            log('Resumed suspended AudioContext');
        }
        return true;
    }
    const ok = await initAudioWorklet();
    if (ok && !workletNodeCreated) {
        audioWorkletNode = new AudioWorkletNode(audioContext, 'pcm-player');
        audioWorkletNode.connect(audioContext.destination);
        audioWorkletNode.port.onmessage = (e) => {
            if (e.data.type === 'log') log('Worklet: ' + e.data.msg);
            if (e.data.type === 'drained') {
                audioDrained = true;
                log('Worklet: audio buffer drained');
                maybeFinishResponse();
            }
        };
        workletNodeCreated = true;
        log('AudioWorkletNode created, state=' + audioContext.state);
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
            log('Resumed after node creation, state=' + audioContext.state);
        }
    }
    return ok;
}

const statusEl = document.getElementById('status');
const primaryBtn = document.getElementById('primaryBtn');
const autoContinueChk = document.getElementById('autoContinueChk');
const convEl = document.getElementById('conversation');
const debugEl = document.getElementById('debug');
const debugContent = document.getElementById('debugContent');
const listenerEl = document.getElementById('listener');
const personaEl = document.getElementById('persona');

function transition(next, reason) {
    if (state === next) return;
    const prev = state;
    state = next;
    log('FSM: ' + prev + ' → ' + next + ' (' + (reason || '') + ')');
    renderUI();
}

function isResponseDone() {
    // Need ALL THREE:
    //   1. LLM finished streaming text (no more sentences coming)
    //   2. Server sent tts_done for every tts_start it emitted
    //   3. Client's audio worklet buffer has actually drained (otherwise
    //      we'd re-arm the mic mid-playback and the OS would duck audio)
    return llmDoneSeen
        && (ttsStartCount === 0 || ttsDoneCount >= ttsStartCount)
        && (ttsStartCount === 0 || audioDrained);
}

// Multi-sentence TTS gotcha: server sends N tts_start / tts_done pairs.
// Between sentence k's tts_done and sentence k+1's tts_start, counts
// transiently match (k == k). If llm_done already arrived, the OLD
// isResponseDone() check returned true and re-armed the mic — which
// then caused mobile OS audio-session ducking when the NEXT sentence's
// TTS audio started playing (user reported 2026-05-26 "TTS speaks for
// a few seconds, then volume drops + recording icon appears"). Adding
// a debounce: only consider the response actually finished if no new
// tts_start has arrived for `RESPONSE_DONE_GRACE_MS` after the last
// terminating signal. Also defers the maybeFinishResponse callback.
const RESPONSE_DONE_GRACE_MS = 1500;
let _maybeFinishTimer = null;

function _maybeFinishResponseImmediate() {
    if (!isResponseDone()) return;
    if (state !== STATE.THINKING && state !== STATE.SPEAKING) return;
    if (autoContinueChk && autoContinueChk.checked) {
        log('Auto-continue: re-arming mic (after grace period)');
        startRecording();
    } else {
        transition(STATE.READY, 'response finished, auto-continue off');
    }
}

function maybeFinishResponse() {
    if (!isResponseDone()) {
        if (_maybeFinishTimer) { clearTimeout(_maybeFinishTimer); _maybeFinishTimer = null; }
        return;
    }
    if (state !== STATE.THINKING && state !== STATE.SPEAKING) return;
    // Restart the grace timer every time we get here. If a new tts_start
    // arrives within the grace window, it'll bump ttsStartCount and the
    // next isResponseDone() check will return false, cancelling the timer.
    if (_maybeFinishTimer) clearTimeout(_maybeFinishTimer);
    _maybeFinishTimer = setTimeout(() => {
        _maybeFinishTimer = null;
        _maybeFinishResponseImmediate();
    }, RESPONSE_DONE_GRACE_MS);
}

function renderUI() {
    if (!primaryBtn) return;

    const pillByState = {
        [STATE.IDLE]:       { cls: 'disconnected', text: 'Disconnected' },
        [STATE.CONNECTING]: { cls: 'connecting',   text: 'Connecting…' },
        [STATE.READY]:      { cls: 'ready',        text: 'Connected · Ready' },
        [STATE.LISTENING]:  { cls: 'listening',    text: 'Listening…' },
        [STATE.THINKING]:   { cls: 'thinking',     text: 'AI thinking…' },
        [STATE.SPEAKING]:   { cls: 'speaking',     text: 'AI speaking…' },
    };
    const pill = pillByState[state] || pillByState[STATE.IDLE];
    statusEl.className = 'status ' + pill.cls;
    statusEl.textContent = pill.text;

    const btnByState = {
        // IDLE = disconnected. Click → reconnect. Distinct from CONNECTING.
        [STATE.IDLE]:       { txt: '🔌 Reconnect',     disabled: false, cls: 'primary' },
        [STATE.CONNECTING]: { txt: 'Connecting…',     disabled: true,  cls: 'primary' },
        [STATE.READY]:      { txt: '🎤 Start Speaking', disabled: false, cls: 'primary' },
        [STATE.LISTENING]:  { txt: '✋ Send',           disabled: false, cls: 'primary recording' },
        [STATE.THINKING]:   { txt: '⏹ Interrupt',      disabled: false, cls: 'primary danger' },
        [STATE.SPEAKING]:   { txt: '⏹ Interrupt',      disabled: false, cls: 'primary danger' },
    };
    const btn = btnByState[state] || btnByState[STATE.IDLE];
    primaryBtn.textContent = btn.txt;
    primaryBtn.className = btn.cls;

    const sys = window.SYS || {};
    const sysBlocks = sys.trainingActive || sys.ttsReady === false || sys.asrReady === false;
    primaryBtn.disabled = btn.disabled || sysBlocks;
    primaryBtn.classList.toggle('gated', sysBlocks);
    primaryBtn.title = sys.trainingActive
        ? 'Training in progress — chat disabled'
        : (sys.ttsReady === false ? 'TTS engine loading…'
        : (sys.asrReady === false ? 'ASR engine loading…' : ''));

    const locked = state === STATE.LISTENING || state === STATE.THINKING || state === STATE.SPEAKING;
    [listenerEl, personaEl, document.getElementById('vad'), document.getElementById('tts_model')].forEach(el => {
        if (el) el.disabled = locked;
    });
}

function addMessage(role, text, emotion) {
    const placeholder = convEl.querySelector('.placeholder');
    if (placeholder) placeholder.remove();
    const div = document.createElement('div');
    div.className = 'message ' + role;
    div.textContent = text;
    if (emotion) {
        const e = document.createElement('div');
        e.className = 'emotion';
        e.textContent = 'Emotion: ' + emotion;
        div.appendChild(e);
    }
    convEl.appendChild(div);
    convEl.scrollTop = convEl.scrollHeight;
}

function log(msg, level = 'info') {
    const d = document.createElement('div');
    d.className = 'debug-entry' + (level === 'error' ? ' error' : '');
    d.textContent = new Date().toISOString().substr(11,8) + ' [' + level.toUpperCase() + '] ' + msg;
    if (debugContent) debugContent.prepend(d);
}

function toggleDebug() {
    debugEl.classList.toggle('show');
}

function clearConversation() {
    convEl.innerHTML = '<div class="placeholder">Press [🎤 Start Speaking] or Space to begin...</div>';
    document.getElementById('thinkingIndicator').style.display = 'none';
    log('Conversation cleared');
}

// Heartbeat ping + auto-reconnect state. Cloudflared (and many corporate
// proxies) kill WebSockets after ~100s of no traffic. A periodic
// no-op message keeps the connection warm; auto-reconnect with
// exponential backoff handles the failures that slip through.
let _heartbeatTimer = null;
let _reconnectAttempts = 0;
const HEARTBEAT_INTERVAL_MS = 30 * 1000;  // 30s — well below the 100s cutoff
const MAX_RECONNECT_DELAY_MS = 10 * 1000;

function _stopHeartbeat() {
    if (_heartbeatTimer) {
        clearInterval(_heartbeatTimer);
        _heartbeatTimer = null;
    }
}

function _scheduleReconnect() {
    const delay = Math.min(
        MAX_RECONNECT_DELAY_MS,
        500 * Math.pow(2, _reconnectAttempts),
    );
    _reconnectAttempts++;
    log('Auto-reconnect scheduled in ' + delay + 'ms (attempt ' + _reconnectAttempts + ')');
    setTimeout(() => {
        if (state === STATE.IDLE) connect();
    }, delay);
}

function connect() {
    if (ws) {
        try { ws.close(); } catch (e) {}
    }
    _stopHeartbeat();
    transition(STATE.CONNECTING, 'connect()');
    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        // Browser autoplay policy blocks AudioContext.resume() before a user
        // gesture. Previously we did `await ensureWorklet()` here on page
        // load — if it threw or stayed pending, the FSM never reached READY
        // and the UI showed "連線中…" forever. Fix: transition to READY
        // immediately; the worklet is initialized lazily on the first
        // primary-button click (which IS a user gesture).
        log('WS connected');
        // Send config WITHOUT touching audio context. sample_rate hint is
        // 24000 — if the audio context ends up at a different rate after
        // user-gesture initialization, we resend config in primary click.
        try {
            ws.send(JSON.stringify({
                type: 'config',
                audio: { sample_rate: 24000, channels: 1, format: 'pcm' },
                persona_id: personaEl.value,
                listener_id: listenerEl.value,
                model: 'gpt-4o-mini',
                vad: document.getElementById('vad').value,
                // tts_model dropdown removed in ea55cac — server tolerates
                // missing field. If a future UI re-adds it, this guard
                // picks up whatever value it sets.
                tts_model: (document.getElementById('tts_model') || {}).value || '1.7B',
            }));
            log('Config sent (worklet deferred to user-gesture)');
        } catch (e) {
            log('Config send failed: ' + e.message);
        }
        transition(STATE.READY, 'ws.onopen');
        // Heartbeat to keep the cloudflared/proxy WebSocket from being
        // killed at ~100s of inactivity. Server tolerates unknown msg
        // types — no server change needed.
        _reconnectAttempts = 0;
        _stopHeartbeat();
        _heartbeatTimer = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                try { ws.send(JSON.stringify({type: 'ping'})); }
                catch (e) { log('ping failed: ' + e.message); }
            }
        }, HEARTBEAT_INTERVAL_MS);
    };

    ws.onmessage = async (e) => {
        if (typeof e.data === 'string') {
            const msg = JSON.parse(e.data);
            handleMessage(msg);
        } else if (e.data instanceof ArrayBuffer) {
            const buf = new Int16Array(e.data);
            if (buf.length > 0 && audioWorkletNode) {
                const now = performance.now();
                const gap = ws._lastBinaryTime ? Math.round(now - ws._lastBinaryTime) : 0;
                ws._lastBinaryTime = now;
                log('WS binary: ' + buf.length + ' samples, gap=' + gap + 'ms');
                if (audioContext.state === 'suspended') await audioContext.resume();
                audioWorkletNode.port.postMessage({ type: 'pcm', buffer: buf.buffer }, [buf.buffer]);
            } else if (buf.length === 0) {
                log('WS binary: empty chunk');
            } else if (!audioWorkletNode) {
                log('WS binary: no audioWorkletNode');
            }
        }
    };

    ws.onclose = () => {
        _stopHeartbeat();
        if (state === STATE.LISTENING) {
            teardownMic();
            accumulatedChunks = [];
        }
        if (audioWorkletNode) {
            try { audioWorkletNode.port.postMessage({ type: 'flush' }); } catch(e) {}
        }
        transition(STATE.IDLE, 'ws.onclose');
        log('WS disconnected');
        // Auto-reconnect with backoff. User can still click "重新連線" in
        // IDLE to skip the wait. Reset attempts counter on successful
        // onopen so a long-stable connection that drops once doesn't
        // start at a huge backoff.
        _scheduleReconnect();
    };

    ws.onerror = (e) => log('WS error: ' + JSON.stringify(e));
}

async function loadVersions() {
    const versionSelect = document.getElementById('versionSelect');
    try {
        const personaId = document.getElementById('persona').value;
        const response = await fetch(`/api/training/versions?persona_id=${personaId}`);
        const data = await response.json();
        const versions = data.versions || [];

        const activeRes = await fetch(`/api/training/active?persona_id=${personaId}`);
        const activeData = await activeRes.json();
        const activeVersionId = activeData.version?.version_id;

        versionSelect.innerHTML = '<option value="">System Default</option>';
        const fmt = window.formatVersionName || ((x) => x);
        versions.forEach(v => {
            if (v.status === 'ready') {
                const pretty = fmt(v.version_id);
                const label = v.nickname ? `${pretty} — ${v.nickname}` : pretty;
                const selected = v.version_id === activeVersionId ? 'selected' : '';
                versionSelect.innerHTML += `<option value="${v.version_id}" ${selected} title="${v.version_id}">${label}</option>`;
            }
        });

        if (activeVersionId) {
            versionSelect.value = activeVersionId;
            selectedVersionId = activeVersionId;
        } else {
            versionSelect.value = '';
            selectedVersionId = null;
        }
        updateVersionInfo(activeData.version);
    } catch (e) {
        log('Failed to load versions: ' + e.message);
    }
}

function updateVersionInfo(version) {
    const versionInfo = document.getElementById('versionInfo');
    if (!version) { versionInfo.textContent = ''; return; }
    const loss = version.final_loss ? ` loss: ${version.final_loss.toFixed(4)}` : '';
    const date = version.completed_at ? new Date(version.completed_at).toLocaleDateString('en-US') : '';
    versionInfo.textContent = `${loss} ${date}`;
}

async function onVersionChange() {
    const versionSelect = document.getElementById('versionSelect');
    const versionId = versionSelect.value;
    selectedVersionId = versionId || null;
    if (versionId) {
        try {
            await fetch(`/api/training/versions/${versionId}/activate`, { method: 'POST' });
            log(`Version ${versionId} activated`);
            showToast(`Switched to: ${versionSelect.options[versionSelect.selectedIndex].text}`, 'success');
            loadVersions();
            // Trigger an immediate status-bar refresh so the top pill
            // reflects the new active version within ~1s instead of
            // waiting for the next 5s poll. Same pattern as the
            // persona-change handler. User reported 2026-05-26: picked
            // v7 in the version dropdown, status bar kept showing v9
            // until the next scheduled poll.
            setTimeout(() => {
                if (window.SYS_FORCE_POLL) window.SYS_FORCE_POLL();
            }, 500);
        } catch (e) {
            log(`Failed to activate version: ${e.message}`, 'error');
            showToast('Failed to switch version', 'error');
        }
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer') || document.body;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    toast.offsetHeight;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => { if (toast.parentNode) toast.remove(); }, 300);
    }, 3000);
}

async function onPrimaryClick() {
    // First click on this page IS a user gesture — kick the worklet/audio
    // context awake here (deferred from ws.onopen to avoid autoplay-policy
    // hang). Subsequent clicks no-op via the workletNodeCreated guard.
    try {
        await ensureWorklet();
    } catch (e) {
        log('ensureWorklet failed (will retry on next click): ' + e.message);
    }
    switch (state) {
        case STATE.IDLE:
            // Disconnected — user clicked "重新連線" to manually reconnect.
            log('Manual reconnect from IDLE');
            connect();
            break;
        case STATE.READY:
            startRecording();
            break;
        case STATE.LISTENING:
            stopRecordingAndSend();
            break;
        case STATE.THINKING:
        case STATE.SPEAKING:
            sendCancel();
            break;
        default:
            log('Primary click ignored in state=' + state);
    }
}

function sendCancel() {
    // Hard abort: unconditionally clear the worklet's ring buffer (the
    // soft 'flush' only clears if buffer < 1s, which doesn't help for
    // mid-TTS cancel). Without this the audio kept playing after 中斷
    // and the mic re-arm immediately after triggered OS-level ducking.
    if (audioWorkletNode) {
        try { audioWorkletNode.port.postMessage({ type: 'abort' }); } catch(e) {}
    }
    // Audio is now flushed, so the worklet won't emit a 'drained' event
    // (transition is buffer → empty, not non-empty → empty when we just
    // wiped it). Mark drained ourselves so isResponseDone is consistent.
    audioDrained = true;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'control', action: 'cancel' }));
        log('Sent cancel');
    }
    // Optimistic UI transition — the user clicked 中斷 to escape THINKING/
    // SPEAKING regardless of whether the server has anything to cancel.
    // Server may not emit `llm_cancelled` if the LLM task already
    // completed (server log: `had_task=False had_event=False`) — without
    // this optimistic transition the FSM would stay in THINKING forever.
    if (state === STATE.THINKING || state === STATE.SPEAKING) {
        ttsText = '';
        ttsEmotion = '';
        const ti = document.getElementById('thinkingIndicator');
        if (ti) ti.style.display = 'none';
        // Reset response-tracking counters so a stray late `llm_done` /
        // `tts_done` doesn't re-fire `maybeFinishResponse` into the next
        // utterance.
        llmDoneSeen = true;
        ttsStartCount = ttsDoneCount;
        if (autoContinueChk && autoContinueChk.checked) {
            startRecording();
        } else {
            transition(STATE.READY, 'sendCancel optimistic');
        }
    }
}

function onTtsModelChange() {
    const hintEl = document.getElementById('ttsModelHint');
    const modelEl = document.getElementById('tts_model');
    if (!modelEl) return;
    const modelValue = modelEl.value;
    const modelLabel = modelValue === '0.6B' ? '0.6B (fast)' : '1.7B (high quality)';
    if (hintEl) {
        const hintValEl = document.getElementById('ttsModelHintValue');
        if (hintValEl) hintValEl.textContent = modelLabel;
        hintEl.style.display = 'block';
        hintEl.style.color = '#00ccff';
        setTimeout(() => { hintEl.style.display = 'none'; }, 3000);
    }
    log('TTS model preference set to: ' + modelValue);
}

function onVadChange() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const vadValue = document.getElementById('vad').value;
    const ttsModelEl = document.getElementById('tts_model');
    ws.send(JSON.stringify({
        type: 'config',
        audio: { sample_rate: 24000, channels: 1, format: 'pcm' },
        persona_id: personaEl.value,
        listener_id: listenerEl.value,
        model: 'gpt-4o-mini',
        vad: vadValue,
        tts_model: ttsModelEl ? ttsModelEl.value : '1.7B'
    }));
    log('VAD sensitivity changed to: ' + vadValue);
}

function handleMessage(msg) {
    log('MSG: ' + JSON.stringify(msg));

    if (msg.type === 'asr_result') {
        log('ASR result: is_final=' + msg.is_final + ' text=' + (msg.text || '(empty)'));
        addMessage('user', msg.text || '(no text)');
    }

    if (msg.type === 'vad_commit') {
        log('VAD commit: silence detected, auto-committing utterance');
        if (state === STATE.LISTENING) {
            if (scriptProcessor && scriptProcessor._localIsRecording) {
                scriptProcessor._localIsRecording.value = false;
            }
            teardownMic();
            ws.send(JSON.stringify({ type: 'control', action: 'commit_utterance' }));
            accumulatedChunks = [];
            log('VAD commit: sent commit_utterance (audio already streamed)');
            transition(STATE.THINKING, 'vad_commit');
        }
    }

    if (msg.type === 'llm_start') {
        log('LLM started: utterance_id=' + msg.utterance_id);
        if (audioWorkletNode) {
            try { audioWorkletNode.port.postMessage({ type: 'flush' }); } catch(e) {}
        }
        ttsText = '';
        ttsStartCount = 0;
        ttsDoneCount = 0;
        llmDoneSeen = false;
        audioDrained = true;  // no audio yet, no need to wait for drain
        document.getElementById('thinkingIndicator').style.display = 'block';
        log('Thinking indicator shown');
        if (state === STATE.LISTENING) {
            transition(STATE.THINKING, 'llm_start (skipped vad_commit)');
        } else if (state === STATE.READY) {
            transition(STATE.THINKING, 'llm_start');
        }
    }

    if (msg.type === 'llm_token') {
        document.getElementById('thinkingIndicator').style.display = 'none';
        let content = msg.content || '';
        if (content === '[' || content === '情' || content === '感' || content === ':' ||
            content === '默' || content === ' 幽' || content.match(/^\[情感/)) {
            // skip emotion-tag fragments
        } else {
            ttsText += content;
        }
        let displayText = ttsText.replace(/\[情感[:：][^\]]*\]/g, '');
        if (displayText.startsWith('[')) {
            displayText = displayText.replace(/^\[情感[:：][^\]]*/, '');
        }
        let aiMsg = convEl.querySelector('.message.ai:last-child');
        if (!aiMsg) {
            aiMsg = document.createElement('div');
            aiMsg.className = 'message ai';
            convEl.appendChild(aiMsg);
        }
        aiMsg.textContent = displayText;
        if (msg.emotion) {
            let e = aiMsg.querySelector('.emotion');
            if (!e) { e = document.createElement('div'); e.className = 'emotion'; aiMsg.appendChild(e); }
            e.textContent = 'Emotion: ' + msg.emotion;
        }
        convEl.scrollTop = convEl.scrollHeight;
    }

    if (msg.type === 'tts_start') {
        ttsStartCount += 1;
        audioDrained = false;  // new audio coming, ignore stale drain signals
        // Cancel any pending "response finished" grace timer — a new
        // sentence just arrived, so we're definitely not done yet.
        if (_maybeFinishTimer) {
            clearTimeout(_maybeFinishTimer);
            _maybeFinishTimer = null;
            log('Cancelled pending auto-continue: new tts_start arrived during grace');
        }
        if (audioWorkletNode) {
            audioWorkletNode.port.postMessage({ type: 'flush' });
        }
        if (audioContext && audioContext.state === 'suspended') {
            audioContext.resume();
        }
        log('TTS stream started: sentence_idx=' + msg.sentence_idx + ' (count=' + ttsStartCount + ')');
        if (state === STATE.THINKING) {
            transition(STATE.SPEAKING, 'tts_start');
        }
    }

    if (msg.type === 'tts_done') {
        ttsDoneCount += 1;
        log('TTS stream done: sentence_idx=' + msg.sentence_idx + ' (' + ttsDoneCount + '/' + ttsStartCount + ')');
        maybeFinishResponse();
    }

    if (msg.type === 'llm_done') {
        log('LLM done: text=' + (msg.text || '').substring(0, 80) + ' total_tokens=' + (msg.total_tokens || '?'));
        ttsText = '';
        ttsEmotion = '';
        document.getElementById('thinkingIndicator').style.display = 'none';
        llmDoneSeen = true;
        maybeFinishResponse();
    }

    if (msg.type === 'llm_cancelled') {
        if (audioWorkletNode) {
            try { audioWorkletNode.port.postMessage({ type: 'flush' }); } catch(e) {}
        }
        ttsText = '';
        ttsEmotion = '';
        document.getElementById('thinkingIndicator').style.display = 'none';
        log('LLM cancelled: partial=' + (msg.partial_text || '').substring(0, 50));
        llmDoneSeen = true;
        ttsStartCount = ttsDoneCount;
        if (state === STATE.THINKING || state === STATE.SPEAKING) {
            if (autoContinueChk && autoContinueChk.checked) {
                startRecording();
            } else {
                transition(STATE.READY, 'llm_cancelled');
            }
        }
    }

    if (msg.type === 'llm_error') {
        log('LLM ERROR: ' + (msg.error || 'unknown'));
        document.getElementById('thinkingIndicator').style.display = 'none';
        llmDoneSeen = true;
        ttsStartCount = ttsDoneCount;
        if (state === STATE.THINKING || state === STATE.SPEAKING) {
            transition(STATE.READY, 'llm_error');
        }
    }

    if (msg.type === 'asr_error') {
        log('ASR ERROR: ' + (msg.error || 'unknown'), 'error');
        if (state === STATE.LISTENING || state === STATE.THINKING) {
            transition(STATE.READY, 'asr_error');
        }
    }

    if (msg.type === 'vad_error') {
        log('VAD ERROR: ' + (msg.error || 'unknown'), 'error');
    }

    if (msg.type === 'tts_error') {
        log('TTS stream error: sentence_idx=' + (msg.sentence_idx || '?') + ' error=' + (msg.error || 'unknown'), 'error');
        ttsDoneCount += 1;
        maybeFinishResponse();
    }
}

async function startRecording() {
    if (isStartingRecording) return;
    isStartingRecording = true;
    if (state === STATE.LISTENING) {
        isStartingRecording = false;
        return;
    }
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log('startRecording: WS not open, cannot start');
        isStartingRecording = false;
        return;
    }
    try {
        // MediaTrackConstraints. AGC + echo cancellation + noise
        // suppression ENABLED (2026-05-20 VAD investigation): without AGC
        // raw laptop mics deliver RMS ~0.005-0.025 which Silero can't
        // distinguish from silence. Echo cancellation prevents TTS
        // playback feedback from triggering false barge-in cancels.
        // Tear down any prior mic graph BEFORE acquiring a new stream.
        // If we do it after, teardownMic() would stop the tracks of the
        // stream we JUST acquired (because we already set
        // recordingStream = stream) → onaudioprocess delivers zeros →
        // VAD never triggers. Observed 2026-05-26 with all chunks
        // reporting avg_abs=0.0000.
        teardownMic();

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 24000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            }
        });
        recordingStream = stream;

        if (!audioContext || audioContext.state === 'closed') {
            audioContext = new AudioContext({ sampleRate: 24000 });
        } else if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        log('Using onaudioprocess for raw PCM at ' + audioContext.sampleRate + ' Hz');

        let debugSampleCount = 0;
        const localIsRecording = { value: true };

        scriptProcessor.onaudioprocess = (e) => {
            if (!localIsRecording.value) return;
            const inputData = e.inputBuffer.getChannelData(0);
            if (debugSampleCount < 3) {
                let sum = 0;
                for (let i = 0; i < inputData.length; i++) sum += Math.abs(inputData[i]);
                debugSampleCount++;
                log('audio chunk ' + debugSampleCount + ': avg_abs=' + (sum/inputData.length).toFixed(4));
            }
            const int16 = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const s = Math.max(-1, Math.min(1, inputData[i]));
                int16[i] = s < 0 ? s * 32768 : s * 32767;
            }
            if (ws && ws.readyState === WebSocket.OPEN) ws.send(int16.buffer);
            accumulatedChunks.push(int16.buffer);
        };

        micSource = audioContext.createMediaStreamSource(stream);
        micSource.connect(scriptProcessor);
        // ScriptProcessorNode requires connection to a destination for
        // onaudioprocess to fire. Tried routing via a gain=0 GainNode
        // to avoid the OS-level duplex audio detection (2026-05-26
        // commit 561e5c2), but mobile Chrome optimizes silent-output
        // paths and stopped delivering input samples — VAD went dead
        // (server saw rms=0.0000 across all chunks). Reverted to direct
        // destination connection; the audio output is silent anyway
        // because the actual playback path is the AudioWorklet, not
        // this scriptProcessor. The mic-indicator behavior on mobile
        // still benefits from the other fixes (micSource disconnect,
        // recordingStream.tracks.stop, hard abort on cancel).
        scriptProcessor.connect(audioContext.destination);

        accumulatedChunks = [];
        ttsText = '';
        scriptProcessor._localIsRecording = localIsRecording;

        ws.send(JSON.stringify({ type: 'control', action: 'start_speech' }));
        log('start_speech sent');
        transition(STATE.LISTENING, 'startRecording');
    } catch (e) {
        // No alert() — modal popup mid-demo would be catastrophic on
        // stage if Chrome happens to throw a permission/notfound error.
        // Toast + log instead so the audience sees nothing dramatic and
        // the operator can retry by clicking the button again.
        log('Mic error: ' + e.message);
        showToast('Mic failed to start — try again: ' + e.message, 'error');
        transition(STATE.READY, 'startRecording failed');
    } finally {
        isStartingRecording = false;
    }
}

function stopRecordingAndSend() {
    if (state !== STATE.LISTENING) return;
    if (scriptProcessor && scriptProcessor._localIsRecording) {
        scriptProcessor._localIsRecording.value = false;
    }
    teardownMic();
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'control', action: 'commit_utterance' }));
    }
    accumulatedChunks = [];
    transition(STATE.THINKING, 'force commit');
}

listenerEl.addEventListener('change', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        const ttsModelEl = document.getElementById('tts_model');
        ws.send(JSON.stringify({
            type: 'config',
            audio: { sample_rate: 24000, channels: 1, format: 'pcm' },
            persona_id: personaEl.value,
            listener_id: listenerEl.value,
            model: 'gpt-4o-mini',
            vad: document.getElementById('vad').value,
            tts_model: ttsModelEl ? ttsModelEl.value : '1.7B'
        }));
    }
    loadVersions();
});

function resendConfig(reason) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const ttsModelEl = document.getElementById('tts_model');
    ws.send(JSON.stringify({
        type: 'config',
        audio: { sample_rate: 24000, channels: 1, format: 'pcm' },
        persona_id: personaEl.value,
        listener_id: listenerEl.value,
        model: 'gpt-4o-mini',
        vad: document.getElementById('vad').value,
        tts_model: ttsModelEl ? ttsModelEl.value : '1.7B'
    }));
    log('Config resent (' + reason + '): persona=' + personaEl.value + ' listener=' + listenerEl.value);
}

// Nuclear reset for live demos — if mid-demo state goes sideways
// (stuck SPEAKING, runaway TTS, sticky mic indicator, ghost cancel
// state), this button restores READY in one click without a full
// page refresh (which would drop the WS + lose the worklet warmup).
// Stops everything, clears the conversation, re-sends config to the
// server, returns FSM to READY.
async function demoReset() {
    log('=== DEMO RESET ===');
    try {
        // 1. Kill any in-flight TTS audio
        if (audioWorkletNode) {
            try { audioWorkletNode.port.postMessage({ type: 'abort' }); } catch (_) {}
        }
        audioDrained = true;
        // 2. Cancel any pending finish-response grace timer
        if (typeof _maybeFinishTimer !== 'undefined' && _maybeFinishTimer) {
            clearTimeout(_maybeFinishTimer);
            _maybeFinishTimer = null;
        }
        // 3. Tell server to drop any in-flight LLM/TTS task
        if (ws && ws.readyState === WebSocket.OPEN) {
            try { ws.send(JSON.stringify({ type: 'control', action: 'cancel' })); } catch (_) {}
        }
        // 4. Tear down mic completely
        teardownMic();
        // 5. Reset response-tracking counters so a stale late event can't fire callbacks
        ttsStartCount = 0;
        ttsDoneCount = 0;
        llmDoneSeen = false;
        ttsText = '';
        ttsEmotion = '';
        accumulatedChunks = [];
        // 6. Clear the conversation DOM
        if (convEl) {
            convEl.innerHTML = '<div style="text-align:center;color:#666;padding:24px;">Conversation reset</div>';
            setTimeout(() => { if (convEl) convEl.innerHTML = ''; }, 1500);
        }
        // 7. Re-send config so the server's session state is in sync
        resendConfig('demo reset');
        // 8. Force FSM back to READY (or LISTENING-ready if ws is closed)
        if (ws && ws.readyState === WebSocket.OPEN) {
            transition(STATE.READY, 'demo reset');
        } else {
            transition(STATE.IDLE, 'demo reset (ws closed)');
        }
        log('Demo reset complete');
    } catch (e) {
        log('Demo reset error: ' + e.message);
    }
}
window.demoReset = demoReset;

// Update the visible tone chip ("正在以 X 的口氣對 Y 說話") whenever
// the persona OR listener selection changes. Pure visual reinforcement
// for the demo audience — proves that the dropdowns aren't decorative
// and that the AI's relationship to the listener is real state.
function updateToneChip() {
    const pEl = document.getElementById('toneChipPersona');
    const lEl = document.getElementById('toneChipListener');
    if (!pEl || !lEl) return;
    const pOpt = personaEl.options[personaEl.selectedIndex];
    const lOpt = listenerEl.options[listenerEl.selectedIndex];
    pEl.textContent = pOpt ? (pOpt.textContent || pOpt.value) : '—';
    lEl.textContent = lOpt ? (lOpt.textContent || lOpt.value) : '—';
}

personaEl.addEventListener('change', () => {
    // The server uses persona_id from the LAST config message to resolve
    // which TTS model to activate (custom_voice vs voice_clone). Without
    // resending, picking "Test" in the dropdown leaves the server still
    // synthesizing for "xiao_s" — the user hears the wrong voice no
    // matter how many times they re-activate the version. Observed
    // 2026-05-21 as "v9 sounds female in chat" after the activate_version
    // path fix landed. The fix here is the second half: server-side
    // routing uses the right persona, client-side notifies the server
    // when persona changes.
    resendConfig('persona changed');
    loadVersions();
    updateToneChip();
    log('Persona switched to ' + personaEl.value + ' — new relationship-block applied for listener=' + listenerEl.value);
    // Server's eager-activation on config receipt swaps the model
    // synchronously, so a quick /api/system/status hit ~500ms later
    // will reflect the new active_version. The status bar normally
    // polls every 5s; this just shortens the apparent latency.
    setTimeout(() => {
        if (window.SYS_FORCE_POLL) window.SYS_FORCE_POLL();
    }, 500);
});

listenerEl.addEventListener('change', () => {
    resendConfig('listener changed');
    updateToneChip();
    log('Listener switched to ' + listenerEl.value + ' — persona will adapt tone (same voice)');
});

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

    if (e.code === 'Space' && !e.repeat) {
        e.preventDefault();
        if (!primaryBtn.disabled) onPrimaryClick();
    }
    if (e.code === 'Escape') {
        e.preventDefault();
        if (state === STATE.LISTENING) {
            if (scriptProcessor && scriptProcessor._localIsRecording) scriptProcessor._localIsRecording.value = false;
            teardownMic();
            accumulatedChunks = [];
            sendCancel();
            transition(STATE.READY, 'Esc cancel during listening');
        } else if (state === STATE.THINKING || state === STATE.SPEAKING) {
            sendCancel();
        }
    }
    if (e.code === 'KeyK' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        clearConversation();
        log('Ctrl+K: conversation cleared');
    }
});

window.toggleDebug = toggleDebug;
window.clearConversation = clearConversation;
window.onPrimaryClick = onPrimaryClick;
window.onTtsModelChange = onTtsModelChange;
window.onVadChange = onVadChange;
window.loadVersions = loadVersions;
window.onVersionChange = onVersionChange;

// Persona + listener dropdowns were originally hardcoded in
// standalone.html with the legacy defaults. New personas / listeners
// created from the recordings page (e.g. "test" persona for a child's
// voice) never appeared in the chat dropdowns. Fetch + populate from
// the same APIs the recordings page uses so the chat sees everything.
async function loadPersonasIntoSelect() {
    try {
        const res = await fetch('/api/personas/');
        if (!res.ok) return;
        const data = await res.json();
        const personas = data.personas || [];
        if (personas.length === 0) return;
        const prev = personaEl.value || 'xiao_s';
        personaEl.innerHTML = '';
        for (const p of personas) {
            const opt = document.createElement('option');
            opt.value = p.persona_id;
            opt.textContent = p.name || p.persona_id;
            personaEl.appendChild(opt);
        }
        // Restore previous selection if it still exists, else default to xiao_s, else first.
        const has = (id) => personas.some(p => p.persona_id === id);
        personaEl.value = has(prev) ? prev : (has('xiao_s') ? 'xiao_s' : personas[0].persona_id);
        log(`Loaded ${personas.length} personas into dropdown`);
    } catch (e) {
        log('Failed to load personas: ' + e.message);
    }
}

async function loadListenersIntoSelect() {
    try {
        const res = await fetch('/api/listeners/');
        if (!res.ok) return;
        const data = await res.json();
        const listeners = data.listeners || [];
        if (listeners.length === 0) return;
        const prev = listenerEl.value || 'default';
        listenerEl.innerHTML = '';
        for (const l of listeners) {
            const opt = document.createElement('option');
            opt.value = l.listener_id;
            opt.textContent = l.name || l.listener_id;
            listenerEl.appendChild(opt);
        }
        const has = (id) => listeners.some(l => l.listener_id === id);
        listenerEl.value = has(prev) ? prev : (has('default') ? 'default' : listeners[0].listener_id);
        log(`Loaded ${listeners.length} listeners into dropdown`);
    } catch (e) {
        log('Failed to load listeners: ' + e.message);
    }
}

renderUI();
connect();
// Populate dropdowns first so loadVersions sees the right persona.
(async () => {
    await Promise.all([loadPersonasIntoSelect(), loadListenersIntoSelect()]);
    loadVersions();
    updateToneChip();  // initial chip render once dropdowns are populated
})();
