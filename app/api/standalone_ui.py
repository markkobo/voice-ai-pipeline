"""
Standalone Voice AI UI — Simple HTML/JS page served by FastAPI.

No Gradio dependency. Connects to WebSocket and plays audio directly.
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

UI_HTML = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice AI — 小S</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 20px; }
        .container { background: white; border-radius: 16px; padding: 24px; width: 100%; max-width: 700px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        h1 { font-size: 24px; margin-bottom: 4px; color: #333; }
        .subtitle { color: #888; margin-bottom: 20px; font-size: 14px; }
        .status { padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: bold; display: inline-block; margin-bottom: 20px; }
        .status.disconnected { background: #eee; color: #888; }
        .status.connecting { background: #fff3cd; color: #856404; }
        .status.connected { background: #d4edda; color: #155724; }
        .status.recording { background: #f8d7da; color: #721c24; animation: pulse 1s infinite; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
        .config-row { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
        .config-item { flex: 1; min-width: 150px; }
        label { display: block; font-size: 12px; color: #666; margin-bottom: 4px; font-weight: 500; }
        select, button { width: 100%; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; outline: none; }
        select:focus, button:focus { border-color: #007bff; }
        button.primary { background: #007bff; color: white; border: none; cursor: pointer; font-weight: 500; }
        button.primary:hover { background: #0056b3; }
        button.primary:disabled { background: #ccc; cursor: not-allowed; }
        .controls { display: flex; gap: 8px; margin-bottom: 16px; }
        .conversation { background: #f8f9fa; border-radius: 12px; padding: 16px; min-height: 200px; max-height: 400px; overflow-y: auto; margin-bottom: 16px; }
        .message { margin-bottom: 12px; padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.5; }
        .message.user { background: #007bff; color: white; margin-left: 20px; border-bottom-right-radius: 4px; }
        .message.ai { background: #e9ecef; color: #333; margin-right: 20px; border-bottom-left-radius: 4px; }
        .message .emotion { font-size: 11px; color: #888; margin-top: 4px; }
        .debug { background: #1e1e1e; color: #ddd; border-radius: 8px; padding: 12px; font-family: 'Courier New', monospace; font-size: 12px; max-height: 200px; overflow-y: auto; margin-top: 16px; display: none; }
        .debug.show { display: block; }
        .debug-toggle { font-size: 12px; color: #888; cursor: pointer; margin-top: 8px; }
        .debug-entry { padding: 2px 0; border-bottom: 1px solid #333; }
        .debug-entry.error { color: #ff4444; }
        .placeholder { color: #aaa; text-align: center; padding: 40px; }
        .thinking {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            background: #e9ecef;
            border-radius: 12px;
            margin-right: 20px;
            border-bottom-left-radius: 4px;
            color: #666;
            font-size: 14px;
        }
        .thinking-dot {
            width: 8px;
            height: 8px;
            background: #007bff;
            border-radius: 50%;
            animation: thinking-bounce 1.4s infinite ease-in-out both;
        }
        .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
        .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes thinking-bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        .keyboard-hint {
            font-size: 11px;
            color: #aaa;
            margin-top: 8px;
        }
        .keyboard-hint kbd {
            background: #ddd;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 10px;
        }
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
            z-index: 9999;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
            max-width: 300px;
            margin-top: 8px;
        }
        .toast.show { opacity: 1; }
        .toast.success { background: #1a4a2e; color: #00ff88; border: 1px solid #00ff88; }
        .toast.error { background: #4a1a1a; color: #ff4444; border: 1px solid #ff4444; }
        .toast.info { background: #1a2a4a; color: #00ccff; border: 1px solid #00ccff; }
    </style>
</head>
<body>
    <div id="toastContainer"></div>
    <div class="container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>Voice AI — 小S <span id="uiVersion" style="font-size:12px;color:#888">[v26]</span></h1>
                <p class="subtitle">選擇對象，開始對話</p>
            </div>
            <button onclick="clearConversation()" style="background:#555;color:#fff;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:13px;">🗑 清除對話</button>
        </div>
        <div id="status" class="status disconnected">Disconnected</div>

        <div class="config-row">
            <div class="config-item">
                <label>對象 (Listener)</label>
                <select id="listener">
                    <option value="child">小孩</option>
                    <option value="mom">媽媽</option>
                    <option value="friend">朋友</option>
                    <option value="default" selected>一般</option>
                </select>
            </div>
            <div class="config-item">
                <label>人格 (Persona)</label>
                <select id="persona">
                    <option value="xiao_s" selected>小S</option>
                    <option value="caregiver">照護者</option>
                    <option value="elder_gentle">長輩-溫柔</option>
                    <option value="elder_playful">長輩-活潑</option>
                </select>
            </div>
            <div class="config-item">
                <label>VAD 靈敏度</label>
                <select id="vad" onchange="onVadChange()">
                    <option value="low">低</option>
                    <option value="medium" selected>中</option>
                    <option value="high">高</option>
                </select>
            </div>
            <div class="config-item">
                <label>TTS 模型</label>
                <select id="tts_model" onchange="onTtsModelChange()">
                    <option value="0.6B">0.6B (快速)</option>
                    <option value="1.7B" selected>1.7B (高品質)</option>
                </select>
            </div>
        </div>
        <div id="ttsModelHint" style="font-size: 0.8rem; color: #666; margin-bottom: 12px; display: none;">
            ✓ 已切換為 <span id="ttsModelHintValue"></span>，下次對話時生效
        </div>

        <!-- Active version indicator -->
        <div id="versionIndicator" style="background: #1a1a2e; border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; font-size: 0.85rem; display: flex; align-items: center; gap: 8px;">
            <span style="color: #888;">🎙️ 聲音版本:</span>
            <select id="versionSelect" onchange="onVersionChange()" style="background: #1a1a2e; color: #00ccff; border: 1px solid #333; border-radius: 4px; padding: 4px 8px; font-size: 0.85rem; max-width: 220px;">
                <option value="">系統預設</option>
            </select>
            <span id="versionInfo" style="color: #666; font-size: 0.75rem;"></span>
            <button onclick="loadVersions()" style="background: #333; color: #aaa; border: none; padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; margin-left: auto;">🔄</button>
        </div>

        <div class="controls">
            <button id="startStopBtn" class="primary" onclick="toggleConversation()">🎤 開始對話</button>
            <button id="recordBtn" class="primary" disabled onclick="toggleRecording()">🎤 開始錄音</button>
            <button id="commitBtn" disabled>✋ 強制送出</button>
            <button id="cancelBtn" disabled>⏹ 取消</button>
        </div>

        <div class="conversation" id="conversation">
            <div class="placeholder">開始說話吧...</div>
        </div>

        <!-- Thinking indicator (hidden by default) -->
        <div id="thinkingIndicator" style="display: none; margin-bottom: 16px;">
            <div class="thinking">
                <span>AI 思考中</span>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
        </div>

        <div class="debug-toggle" onclick="toggleDebug()">📋 Debug Panel (點擊展開)</div>
        <div id="debug" class="debug">
            <div id="debugContent"></div>
        </div>
        <div class="keyboard-hint">
            快捷鍵: <kbd>Space</kbd> 說話/停止 &nbsp;|&nbsp; <kbd>Esc</kbd> 取消 &nbsp;|&nbsp; <kbd>Ctrl+K</kbd> 清除對話
        </div>
    </div>

    <script>
    // Global error handler to catch uncaught errors
    window.onerror = function(msg, url, line, col, error) {
        log('GLOBAL ERROR: ' + msg + ' at line ' + line + ' col ' + col);
        return false;
    };
    // Version for debugging
    window.UI_VERSION = '2026-04-01-v26-emotion-fix';
    console.log('UI Version: ' + window.UI_VERSION);
    document.getElementById('uiVersion').textContent = '[v26]';

    const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/asr';

    let ws = null;
    let audioContext = null;
    let scriptProcessor = null;
    let isRecording = false;
    let isStartingRecording = false;  // Guard against re-entrant start
    let utteranceId = null;
    let ttsText = '';
    let ttsEmotion = null;
    let recordingStream = null;
    let accumulatedChunks = [];  // Accumulated PCM ArrayBuffers
    let isThinking = false;  // Track if AI is processing
    let selectedVersionId = null;  // Selected TTS version ID from dropdown

    // AudioWorklet for streaming PCM playback
    let audioWorkletNode = null;
    let workletInitialized = false;  // Track if worklet module is registered
    let workletNodeCreated = false;  // Track if AudioWorkletNode is created

    // Simple AudioWorklet implementation
    async function initAudioWorklet() {
        if (workletInitialized) {
            log('Worklet already initialized');
            return true;
        }

        log('initAudioWorklet: starting');

        // Create AudioContext if needed
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
            log('Created AudioContext');
        }

        // Resume if suspended
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
            log('Resumed AudioContext');
        }

        // Check if AudioWorklet is supported
        if (!audioContext.audioWorklet) {
            log('AudioWorklet not supported');
            return false;
        }

        try {
            const workletCode = `
                class PCMPlayer extends AudioWorkletProcessor {
                    constructor() {
                        super();
                        // Ring buffer for incoming PCM data - 20 seconds
                        this.ringBuffer = new Float32Array(24000 * 20);
                        this.writePos = 0;
                        this.readPos = 0;
                        this.samplesInBuffer = 0;
                        this.isPlaying = true;
                        this.pendingFlush = false;  // Don't interrupt mid-sentence

                        this.port.onmessage = (e) => {
                            if (e.data.type === 'pcm') {
                                // Convert incoming Int16 to Float32 and add to ring buffer
                                const int16Data = new Int16Array(e.data.buffer);
                                let dropped = 0;
                                for (let i = 0; i < int16Data.length; i++) {
                                    this.ringBuffer[this.writePos] = int16Data[i] / 32768.0;
                                    this.writePos = (this.writePos + 1) % this.ringBuffer.length;
                                    this.samplesInBuffer++;
                                    // Prevent overflow - drop oldest samples if needed
                                    if (this.samplesInBuffer > this.ringBuffer.length) {
                                        this.readPos = (this.readPos + 1) % this.ringBuffer.length;
                                        this.samplesInBuffer--;
                                        dropped++;
                                    }
                                }
                                // Log if we dropped samples
                                if (dropped > 0) {
                                    this.port.postMessage({ type: 'log', msg: 'BUF DROPPED ' + dropped + ' samples, buf=' + this.samplesInBuffer });
                                }
                            } else if (e.data.type === 'flush') {
                                // Only flush if buffer is nearly empty (end of sentence)
                                // This prevents cutting off mid-sentence audio
                                if (this.samplesInBuffer < 24000) {  // < 1 second
                                    this.writePos = 0;
                                    this.readPos = 0;
                                    this.samplesInBuffer = 0;
                                }
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
                            // Output silence if not playing
                            for (let i = 0; i < out.length; i++) {
                                out[i] = 0;
                            }
                            return true;
                        }

                        // Fill output buffer from ring buffer
                        for (let i = 0; i < out.length; i++) {
                            if (this.samplesInBuffer > 0) {
                                out[i] = this.ringBuffer[this.readPos];
                                this.readPos = (this.readPos + 1) % this.ringBuffer.length;
                                this.samplesInBuffer--;
                            } else {
                                out[i] = 0;
                            }
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
            // Ensure audio context is running
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
            // Handle messages from AudioWorklet
            audioWorkletNode.port.onmessage = (e) => {
                if (e.data.type === 'log') {
                    log('Worklet: ' + e.data.msg);
                }
            };
            workletNodeCreated = true;
            log('AudioWorkletNode created, state=' + audioContext.state);

            // CRITICAL: Ensure AudioContext is running before sending audio
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
                log('Resumed after node creation, state=' + audioContext.state);
            }
        }
        return ok;
    }

    const statusEl = document.getElementById('status');
    const startStopBtn = document.getElementById('startStopBtn');
    const recordBtn = document.getElementById('recordBtn');
    const commitBtn = document.getElementById('commitBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const convEl = document.getElementById('conversation');
    const debugEl = document.getElementById('debug');
    const debugContent = document.getElementById('debugContent');
    const listenerEl = document.getElementById('listener');
    const personaEl = document.getElementById('persona');

    function setStatus(s) {
        statusEl.className = 'status ' + s;
        statusEl.textContent = { disconnected: 'Disconnected', connecting: 'Connecting...', connected: 'Connected', recording: 'Recording...' }[s] || s;
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
            e.textContent = '情緒: ' + emotion;
            div.appendChild(e);
        }
        convEl.appendChild(div);
        convEl.scrollTop = convEl.scrollHeight;
    }

    function log(msg, level = 'info') {
        const d = document.createElement('div');
        d.className = 'debug-entry' + (level === 'error' ? ' error' : '');
        d.textContent = new Date().toISOString().substr(11,8) + ' [' + level.toUpperCase() + '] ' + msg;
        debugContent.prepend(d);
    }

    function toggleDebug() {
        debugEl.classList.toggle('show');
    }

    function clearConversation() {
        convEl.innerHTML = '<div class="placeholder">開始說話吧...</div>';
        isThinking = false;
        document.getElementById('thinkingIndicator').style.display = 'none';
        log('Conversation cleared');
    }

    function connect() {
        if (ws) ws.close();
        setStatus('connecting');
        ws = new WebSocket(WS_URL);
        ws.binaryType = 'arraybuffer';

        ws.onopen = async () => {
            setStatus('connected');
            if (startStopBtn) startStopBtn.textContent = '🔴 停止對話';
            if (recordBtn) recordBtn.disabled = false;
            log('WS connected');
            // Initialize AudioWorklet for TTS playback
            await ensureWorklet();
            // Send config
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
            log('Config sent with tts_model=' + (ttsModelEl ? ttsModelEl.value : '1.7B'));
        };

        ws.onmessage = async (e) => {
            if (typeof e.data === 'string') {
                const msg = JSON.parse(e.data);
                handleMessage(msg);
            } else if (e.data instanceof ArrayBuffer) {
                // Binary PCM chunk streamed directly from TTS — play immediately
                const buf = new Int16Array(e.data);
                if (buf.length > 0 && audioWorkletNode) {
                    const now = performance.now();
                    const gap = ws._lastBinaryTime ? Math.round(now - ws._lastBinaryTime) : 0;
                    ws._lastBinaryTime = now;
                    log('WS binary: ' + buf.length + ' samples, gap=' + gap + 'ms');
                    // Ensure AudioContext is running before posting
                    if (audioContext.state === 'suspended') {
                        await audioContext.resume();
                    }
                    // Send raw PCM to AudioWorklet for immediate playback
                    audioWorkletNode.port.postMessage({ type: 'pcm', buffer: buf.buffer }, [buf.buffer]);
                } else if (buf.length === 0) {
                    log('WS binary: empty chunk');
                } else if (!audioWorkletNode) {
                    log('WS binary: no audioWorkletNode');
                }
            }
        };

        ws.onclose = () => {
            setStatus('disconnected');
            if (startStopBtn) {
                startStopBtn.textContent = '🎤 開始對話';
                startStopBtn.disabled = false;
            }
            if (recordBtn) {
                recordBtn.disabled = true;
                recordBtn.textContent = '🎤 開始錄音';
            }
            commitBtn.disabled = true;
            cancelBtn.disabled = true;
            if (isRecording) {
                if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
                if (recordingStream) { recordingStream.getTracks().forEach(t => t.stop()); recordingStream = null; }
                isRecording = false;
                accumulatedChunks = [];
            }
            if (audioWorkletNode) {
                try { audioWorkletNode.port.postMessage({ type: 'flush' }); } catch(e) {}
            }
            log('WS disconnected');
        };

        ws.onerror = (e) => log('WS error: ' + JSON.stringify(e));
    }

    // Load all versions for selected persona into dropdown
    async function loadVersions() {
        const versionSelect = document.getElementById('versionSelect');
        const versionInfo = document.getElementById('versionInfo');
        try {
            const personaId = document.getElementById('persona').value;
            const response = await fetch(`/api/training/versions?persona_id=${personaId}`);
            const data = await response.json();
            const versions = data.versions || [];

            // Get active version
            const activeRes = await fetch(`/api/training/active?persona_id=${personaId}`);
            const activeData = await activeRes.json();
            const activeVersionId = activeData.version?.version_id;

            versionSelect.innerHTML = '<option value="">系統預設</option>';
            versions.forEach(v => {
                if (v.status === 'ready') {
                    const label = v.nickname ? `${v.version_id}: ${v.nickname}` : v.version_id;
                    const selected = v.version_id === activeVersionId ? 'selected' : '';
                    versionSelect.innerHTML += `<option value="${v.version_id}" ${selected}>${label}</option>`;
                }
            });

            // Set selected from active
            if (activeVersionId) {
                versionSelect.value = activeVersionId;
                selectedVersionId = activeVersionId;
            } else {
                versionSelect.value = '';
                selectedVersionId = null;
            }

            // Show info for selected version
            updateVersionInfo(activeData.version);

        } catch (e) {
            log('Failed to load versions: ' + e.message);
        }
    }

    // Update version info tooltip
    function updateVersionInfo(version) {
        const versionInfo = document.getElementById('versionInfo');
        if (!version) {
            versionInfo.textContent = '';
            return;
        }
        const loss = version.final_loss ? ` loss: ${version.final_loss.toFixed(4)}` : '';
        const date = version.completed_at ? new Date(version.completed_at).toLocaleDateString('zh-TW') : '';
        versionInfo.textContent = `${loss} ${date}`;
    }

    // On version dropdown change → activate version
    async function onVersionChange() {
        const versionSelect = document.getElementById('versionSelect');
        const versionId = versionSelect.value;
        selectedVersionId = versionId || null;

        if (versionId) {
            try {
                await fetch(`/api/training/versions/${versionId}/activate`, { method: 'POST' });
                log(`Version ${versionId} activated`);
                showToast(`已切換至: ${versionSelect.options[versionSelect.selectedIndex].text}`, 'success');
                loadVersions();  // Refresh to update "active" badge
            } catch (e) {
                log(`Failed to activate version: ${e.message}`, 'error');
                showToast('切換版本失敗', 'error');
            }
        }
    }

    // Show toast notification
    function showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer') || document.body;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        // Force reflow so transition fires
        toast.offsetHeight;
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => { if (toast.parentNode) toast.remove(); }, 300);
        }, 3000);
    }

    // Toggle conversation start/stop
    function toggleConversation() {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            connect();
        } else {
            ws.close();
            ws = null;
        }
    }

    // Toggle recording start/stop
    function toggleRecording() {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            log('WS not open, cannot toggle recording');
            return;
        }
        if (isRecording) {
            stopRecordingAndSend();
        } else {
            startRecording();
        }
    }

    function onTtsModelChange() {
        const hint = document.getElementById('ttsModelHint');
        const modelValue = document.getElementById('tts_model').value;
        const modelLabel = modelValue === '0.6B' ? '0.6B (快速)' : '1.7B (高品質)';
        document.getElementById('ttsModelHintValue').textContent = modelLabel;
        hint.style.display = 'block';
        hint.style.color = '#00ccff';
        // Hide after 3 seconds
        setTimeout(() => { hint.style.display = 'none'; }, 3000);
        log('TTS model preference set to: ' + modelValue);
    }

    function onVadChange() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const vadValue = document.getElementById('vad').value;
        ws.send(JSON.stringify({
            type: 'config',
            audio: { sample_rate: 24000, channels: 1, format: 'pcm' },
            persona_id: personaEl.value,
            listener_id: listenerEl.value,
            model: 'gpt-4o-mini',
            vad: vadValue
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
            // Server VAD detected end of speech — audio was already sent in real-time
            // Just stop recording and send commit_utterance (don't resend audio)
            if (isRecording) {
                // Signal onaudioprocess to stop FIRST
                if (scriptProcessor && scriptProcessor._localIsRecording) {
                    scriptProcessor._localIsRecording.value = false;
                }
                // Stop audio processing
                if (scriptProcessor) {
                    scriptProcessor.disconnect();
                    scriptProcessor = null;
                }
                if (recordingStream) {
                    recordingStream.getTracks().forEach(t => t.stop());
                    recordingStream = null;
                }
                isRecording = false;
                if (recordBtn) recordBtn.textContent = '🎤 開始錄音';
                commitBtn.disabled = true;
                cancelBtn.disabled = true;
                // Audio was already sent in real-time - just send commit_utterance
                ws.send(JSON.stringify({ type: 'control', action: 'commit_utterance' }));
                accumulatedChunks = [];
                log('VAD commit: sent commit_utterance (audio already streamed)');
            }
        }

        if (msg.type === 'llm_start') {
            log('LLM started: utterance_id=' + msg.utterance_id);
            // Flush AudioWorklet for new utterance (barge-in)
            if (audioWorkletNode) {
                try {
                    audioWorkletNode.port.postMessage({ type: 'flush' });
                } catch(e) {}
            }
            ttsText = '';
            // Show thinking indicator
            isThinking = true;
            document.getElementById('thinkingIndicator').style.display = 'block';
            log('Thinking indicator shown');
        }

        if (msg.type === 'llm_token') {
            // Hide thinking indicator on first real token
            if (isThinking) {
                isThinking = false;
                document.getElementById('thinkingIndicator').style.display = 'none';
                log('Thinking indicator hidden (first token)');
            }
            let content = msg.content || '';
            // Before accumulating, strip if this content looks like emotion tag fragment
            // These fragments arrive with emotion=null: "默", " 幽", ":", "感", "情", "["
            if (content === '[' || content === '情' || content === '感' || content === ':' ||
                content === '默' || content === ' 幽' || content.match(/^\[情感/)) {
                // Skip this fragment, don't accumulate
            } else {
                ttsText += content;
            }
            // Strip complete emotion tags
            let displayText = ttsText.replace(/\[情感[:：][^\]]*\]/g, '');
            // Remove any partial bracket at start
            if (displayText.startsWith('[')) {
                displayText = displayText.replace(/^\[情感[:：][^\]]*/, '');
            }
            // Only create a new box if there is NO existing AI message box
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
                e.textContent = '情緒: ' + msg.emotion;
            }
            convEl.scrollTop = convEl.scrollHeight;
        }

        if (msg.type === 'tts_start') {
            // New sentence TTS starting over WebSocket — prepare AudioWorklet for immediate playback
            // Flush any pending streaming audio so new chunks take priority
            if (audioWorkletNode) {
                audioWorkletNode.port.postMessage({ type: 'flush' });
            }
            // Ensure AudioContext is running
            if (audioContext && audioContext.state === 'suspended') {
                audioContext.resume();
            }
            log('TTS stream started: sentence_idx=' + msg.sentence_idx);
        }

        if (msg.type === 'tts_done') {
            log('TTS stream done: sentence_idx=' + msg.sentence_idx);
        }

        if (msg.type === 'llm_done') {
            log('LLM done: text=' + (msg.text || '').substring(0, 80) + ' total_tokens=' + (msg.total_tokens || '?'));
            ttsText = '';
            ttsEmotion = '';
            isThinking = false;
            document.getElementById('thinkingIndicator').style.display = 'none';
        }

        if (msg.type === 'llm_cancelled') {
            // Flush AudioWorklet (WS binary streaming - cancel ongoing TTS)
            if (audioWorkletNode) {
                try {
                    audioWorkletNode.port.postMessage({ type: 'flush' });
                } catch(e) {}
            }
            ttsText = '';
            ttsEmotion = '';
            isThinking = false;
            document.getElementById('thinkingIndicator').style.display = 'none';
            log('LLM cancelled: partial=' + (msg.partial_text || '').substring(0, 50));
        }

        if (msg.type === 'llm_error') {
            log('LLM ERROR: ' + (msg.error || 'unknown'));
            isThinking = false;
            document.getElementById('thinkingIndicator').style.display = 'none';
        }

        if (msg.type === 'asr_error') {
            log('ASR ERROR: ' + (msg.error || 'unknown'), 'error');
        }

        if (msg.type === 'vad_error') {
            log('VAD ERROR: ' + (msg.error || 'unknown'), 'error');
        }

        if (msg.type === 'tts_error') {
            log('TTS stream error: sentence_idx=' + (msg.sentence_idx || '?') + ' error=' + (msg.error || 'unknown'), 'error');
        }
    }

    async function startRecording() {
        // Guard against re-entrant calls
        if (isStartingRecording) return;
        isStartingRecording = true;

        // Stop any existing recording first
        if (isRecording) {
            stopRecordingAndSend();
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 24000,
                    channelCount: 1,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                }
            });
            recordingStream = stream;

            // Resume AudioContext if suspended
            if (!audioContext || audioContext.state === 'closed') {
                audioContext = new AudioContext({ sampleRate: 24000 });
            } else if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }

            // --- Web Audio preprocessing chain ---
            // Source → HighPass(80Hz) → Compressor → ScriptProcessor → Destination
            // High-pass removes low-frequency rumble (fan, HVAC, electrical hum)
            // Compressor normalizes volume spikes, making voice more consistent

            // High-pass filter: remove frequencies below 80Hz
            const highPass = audioContext.createBiquadFilter();
            highPass.type = 'highpass';
            highPass.frequency.value = 80;   // Hz - removes rumble below human speech range
            highPass.Q.value = 0.7;
            log('High-pass filter: 80Hz (removes low-frequency noise)');

            // Dynamics compressor: normalize voice volume, reduce clipping
            const compressor = audioContext.createDynamicsCompressor();
            compressor.threshold.value = -24;  // dB - start compressing here
            compressor.knee.value = 12;         // dB - soft knee
            compressor.ratio.value = 4;         // 4:1 compression above threshold
            compressor.attack.value = 0.003;    // 3ms attack - fast enough for speech
            compressor.release.value = 0.1;     // 100ms release - smooth
            log('Dynamics compressor: -24dB threshold, 4:1 ratio');

            // Noise gate state: track running energy to suppress ambient noise
            // Calibrate on first 10 chunks to establish noise floor
            const noiseGate = {
                calibrated: false,
                noiseFloor: 0.0,
                sampleCount: 0,
                calibrationSamples: 10,
                threshold: 0.015,  // RMS threshold - chunks below this are suppressed
                // Adaptive: raise threshold slightly above measured noise floor
                getThreshold: function() {
                    return Math.max(this.threshold, this.noiseFloor * 2.5);
                }
            };

            // Create and configure script processor
            if (scriptProcessor) {
                scriptProcessor.disconnect();
            }
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
            log('Using onaudioprocess for raw PCM at ' + audioContext.sampleRate + ' Hz');

            // Debug: check first few audio samples
            let debugSampleCount = 0;
            let debugSum = 0;

            const localIsRecording = { value: true };

            scriptProcessor.onaudioprocess = (e) => {
                if (!localIsRecording.value) return;
                const inputData = e.inputBuffer.getChannelData(0); // Float32 mono

                // --- Noise gate: calibrate on first N chunks, then suppress quiet ambient ---
                let chunkRMS = 0;
                for (let i = 0; i < inputData.length; i++) {
                    chunkRMS += inputData[i] * inputData[i];
                }
                chunkRMS = Math.sqrt(chunkRMS / inputData.length);

                // Calibrate noise floor from first few chunks
                if (!noiseGate.calibrated) {
                    noiseGate.noiseFloor = Math.max(noiseGate.noiseFloor, chunkRMS);
                    noiseGate.sampleCount++;
                    if (noiseGate.sampleCount >= noiseGate.calibrationSamples) {
                        noiseGate.calibrated = true;
                        log('Noise gate calibrated: floor=' + noiseGate.noiseFloor.toFixed(4) +
                            ', threshold=' + noiseGate.getThreshold().toFixed(4));
                    }
                }

                // Suppress chunks below adaptive threshold (gating, not muting)
                // Mute entirely if not yet calibrated
                if (!noiseGate.calibrated || chunkRMS >= noiseGate.getThreshold()) {
                    // Debug: check if audio data is non-zero
                    let sum = 0;
                    for (let i = 0; i < inputData.length; i++) {
                        sum += Math.abs(inputData[i]);
                    }
                    debugSum += sum;
                    debugSampleCount++;
                    if (debugSampleCount <= 3) {
                        log('audio chunk ' + debugSampleCount + ': avg_abs=' + (sum/inputData.length).toFixed(4));
                    }

                    // Convert Float32 [-1,1] -> Int16 [−32768, 32767]
                    const int16 = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        const s = Math.max(-1, Math.min(1, inputData[i]));
                        int16[i] = s < 0 ? s * 32768 : s * 32767;
                    }
                    // Send chunk to server immediately for VAD processing
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(int16.buffer);
                    }
                    // Also accumulate for later sending on commit
                    accumulatedChunks.push(int16.buffer);
                }
                // else: chunk suppressed by noise gate - do NOT send or accumulate
            };

            // Connect the nodes (needed to start processing)
            // Chain: source → highpass → compressor → scriptProcessor → destination
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(highPass);
            highPass.connect(compressor);
            compressor.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            accumulatedChunks = []; // Reset accumulation buffer
            ttsText = ''; // Clear previous LLM response text
            isRecording = true;
            // Store the closure for stop function to access
            scriptProcessor._localIsRecording = localIsRecording;

            // Notify server to cancel any ongoing LLM and reset VAD (barge-in)
            ws.send(JSON.stringify({ type: 'control', action: 'start_speech' }));
            log('start_speech sent');

            setStatus('recording');
            if (recordBtn) recordBtn.textContent = '🔴 錄音中...';
            commitBtn.disabled = false;
            cancelBtn.disabled = false;
            log('Recording started');
        } catch (e) {
            log('Mic error: ' + e.message);
            alert('無法訪問麥克風: ' + e.message);
        } finally {
            isStartingRecording = false;
        }
    }

    function stopRecordingAndSend() {
        if (!isRecording) return;

        // Signal onaudioprocess to stop FIRST (before disconnecting)
        if (scriptProcessor && scriptProcessor._localIsRecording) {
            scriptProcessor._localIsRecording.value = false;
        }

        // Stop audio processing
        if (scriptProcessor) {
            scriptProcessor.disconnect();
            scriptProcessor = null;
        }
        if (recordingStream) {
            recordingStream.getTracks().forEach(t => t.stop());
            recordingStream = null;
        }
        isRecording = false;
        if (recordBtn) recordBtn.textContent = '🎤 開始錄音';
        commitBtn.disabled = true;
        cancelBtn.disabled = true;

        // Audio is now sent in real-time via onaudioprocess
        // Just send commit_utterance (don't resend audio to avoid duplicates)
        ws.send(JSON.stringify({ type: 'control', action: 'commit_utterance' }));
        accumulatedChunks = [];

        setStatus(ws && ws.readyState === WebSocket.OPEN ? 'connected' : 'disconnected');
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
                tts_model: ttsModelEl ? ttsModelEl.value : '1.7B'
            }));
        }
        // Reload versions when listener changes (persona might be same but different active version)
        loadVersions();
    });

    personaEl.addEventListener('change', () => {
        // Reload versions when persona changes
        loadVersions();
    });

    // Commit button: force send current audio even if VAD hasn't triggered
    commitBtn.addEventListener('click', () => {
        log('COMMIT BTN: isRecording=' + isRecording + ', chunks=' + accumulatedChunks.length);
        if (isRecording) {
            stopRecordingAndSend();
        } else if (accumulatedChunks.length > 0) {
            // Send pending audio even without recording
            let totalSamples = 0;
            for (const chunk of accumulatedChunks) totalSamples += chunk.byteLength / 2;
            const combined = new Int16Array(totalSamples);
            let offset = 0;
            for (const chunk of accumulatedChunks) { combined.set(new Int16Array(chunk), offset); offset += new Int16Array(chunk).length; }
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(combined.buffer);
                ws.send(JSON.stringify({ type: 'control', action: 'commit_utterance' }));
                accumulatedChunks = [];
                log('COMMIT: sent pending audio + commit_utterance');
            }
        } else {
            // No audio accumulated — just send commit_utterance
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'control', action: 'commit_utterance' }));
                log('COMMIT: sent commit_utterance (no audio)');
            }
        }
        commitBtn.disabled = true;
    });

    // Cancel button: cancel current LLM request
    cancelBtn.addEventListener('click', () => {
        log('CANCEL BTN');
        if (isRecording) {
            if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
            if (recordingStream) { recordingStream.getTracks().forEach(t => t.stop()); recordingStream = null; }
            isRecording = false;
            accumulatedChunks = [];
        }
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'control', action: 'cancel' }));
            log('CANCEL: cancel SENT');
        }
        cancelBtn.disabled = true;
    });

    // Keyboard shortcuts
    // Space: Push-to-talk (when not recording, start; when recording, stop+send)
    // Esc: Cancel current request
    // Ctrl+K: Clear conversation
    document.addEventListener('keydown', (e) => {
        // Ignore if user is typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

        if (e.code === 'Space' && !e.repeat) {
            e.preventDefault();
            if (!recordBtn.disabled) {
                if (isRecording) {
                    stopRecordingAndSend();
                } else {
                    startRecording();
                }
            }
        }

        if (e.code === 'Escape') {
            e.preventDefault();
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'control', action: 'cancel' }));
                log('ESC: cancel SENT');
            }
            if (isRecording) {
                stopRecordingAndSend();
            }
        }

        if (e.code === 'KeyK' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            // Clear conversation
            convEl.innerHTML = '<div class="placeholder">開始說話吧...</div>';
            log('Ctrl+K: conversation cleared');
        }
    });

    // Auto-connect
    connect();
    // Load versions for selected persona
    loadVersions();
    </script>
</body>
</html>
"""


@router.get("/ui")
async def serve_ui():
    """Serve the standalone voice AI UI."""
    return HTMLResponse(UI_HTML)
