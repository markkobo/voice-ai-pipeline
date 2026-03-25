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
        .placeholder { color: #aaa; text-align: center; padding: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice AI — 小S</h1>
        <p class="subtitle">選擇對象，開始對話</p>
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
                <select id="vad">
                    <option value="low">低</option>
                    <option value="medium" selected>中</option>
                    <option value="high">高</option>
                </select>
            </div>
            <div class="config-item">
                <label>TTS 模型</label>
                <select id="tts_model">
                    <option value="0.6B">0.6B (快速)</option>
                    <option value="1.7B" selected>1.7B (高品質)</option>
                </select>
            </div>
        </div>

        <div class="controls">
            <button id="recordBtn" class="primary" disabled>🎤 按住說話</button>
            <button id="commitBtn" disabled onclick="window.__commitBtnClicked()">✋ 強制送出</button>
            <button id="cancelBtn" disabled>⏹ 取消</button>
        </div>

        <div class="conversation" id="conversation">
            <div class="placeholder">開始說話吧...</div>
        </div>

        <div class="debug-toggle" onclick="toggleDebug()">📋 Debug Panel (點擊展開)</div>
        <div id="debug" class="debug">
            <div id="debugContent"></div>
        </div>
    </div>

    <script>
    const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/asr';
    const TTS_BASE = location.origin + '/api/tts/stream';

    let ws = null;
    let audioContext = null;
    let scriptProcessor = null;
    let isRecording = false;
    let isStartingRecording = false;  // Guard against re-entrant start
    let utteranceId = null;
    let ttsText = '';
    let ttsEmotion = null;
    let ttsAbortController = null;
    let recordingStream = null;
    let accumulatedChunks = [];  // Accumulated PCM ArrayBuffers

    const statusEl = document.getElementById('status');
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

    function log(msg) {
        const d = document.createElement('div');
        d.className = 'debug-entry';
        d.textContent = new Date().toISOString().substr(11,8) + ' ' + msg;
        debugContent.prepend(d);
    }

    function toggleDebug() {
        debugEl.classList.toggle('show');
    }

    function connect() {
        if (ws) ws.close();
        setStatus('connecting');
        ws = new WebSocket(WS_URL);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            setStatus('connected');
            recordBtn.disabled = false;
            commitBtn.disabled = false;
            cancelBtn.disabled = false;
            log('WS connected');
            // Send config
            ws.send(JSON.stringify({
                type: 'config',
                audio: { sample_rate: 24000, channels: 1, format: 'pcm' },
                persona_id: personaEl.value,
                listener_id: listenerEl.value,
                model: 'gpt-4o-mini'
            }));
            log('Config sent');
        };

        ws.onmessage = async (e) => {
            if (typeof e.data === 'string') {
                const msg = JSON.parse(e.data);
                handleMessage(msg);
            }
        };

        ws.onclose = () => {
            setStatus('disconnected');
            recordBtn.disabled = true;
            commitBtn.disabled = true;
            cancelBtn.disabled = true;
            if (isRecording) {
                // Clean up audio nodes without sending
                if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
                if (recordingStream) { recordingStream.getTracks().forEach(t => t.stop()); recordingStream = null; }
                isRecording = false;
                accumulatedChunks = [];
            }
            log('WS disconnected');
        };

        ws.onerror = (e) => log('WS error: ' + JSON.stringify(e));
    }

    function handleMessage(msg) {
        log('MSG: ' + JSON.stringify(msg));

        if (msg.type === 'asr_result') {
            log('ASR result: is_final=' + msg.is_final + ' text=' + (msg.text || '(empty)'));
            addMessage('user', msg.text || '(no text)');
            commitBtn.disabled = false;
        }

        if (msg.type === 'vad_commit') {
            log('VAD commit: energy=' + (msg.energy || '?'));
        }

        if (msg.type === 'llm_start') {
            log('LLM started: utterance_id=' + msg.utterance_id);
            ttsText = '';
        }

        if (msg.type === 'llm_token') {
            // Strip emotion tag from content if present (defensive)
            let content = msg.content || '';
            if (msg.emotion) {
                content = content.replace(/\[情感[:：]\s*.*?\]/g, '');
            }
            ttsText += content;
            log('LLM token: content=' + JSON.stringify(msg.content) + ' emotion=' + (msg.emotion || 'none') + ' ttsText so far=' + ttsText.substring(0, 50));
            // Update AI message
            let aiMsg = convEl.querySelector('.message.ai:last-child');
            if (!aiMsg || aiMsg.querySelector('.emotion')) {
                aiMsg = document.createElement('div');
                aiMsg.className = 'message ai';
                convEl.appendChild(aiMsg);
            }
            aiMsg.textContent = ttsText;
            if (msg.emotion) {
                let e = aiMsg.querySelector('.emotion');
                if (!e) { e = document.createElement('div'); e.className = 'emotion'; aiMsg.appendChild(e); }
                e.textContent = '情緒: ' + msg.emotion;
            }
            convEl.scrollTop = convEl.scrollHeight;
        }

        if (msg.type === 'tts_ready') {
            // tts_ready fires TWICE: once with text="" when emotion is first detected,
            // then again with accumulated text after each llm_token.
            // Only fetch when we have actual text to avoid empty-URL / re-fetch chaos.
            if (!ttsText) {
                // First tts_ready (empty) — just record the emotion and stream_url for later
                ttsEmotion = msg.emotion;
                log('TTS ready (waiting for text): emotion=' + msg.emotion);
                return;
            }
            // Second+ tts_ready with actual text — fetch once
            if (ttsAbortController) ttsAbortController.abort();
            ttsAbortController = new AbortController();
            const url = msg.stream_url + (msg.stream_url.includes('?') ? '&text=' : '?text=') + encodeURIComponent(ttsText);
            log('TTS fetch: emotion=' + msg.emotion + ' text_len=' + ttsText.length + ' url=' + url.substring(0, 80));
            playTTS(url, ttsAbortController.signal);
        }

        if (msg.type === 'llm_done') {
            log('LLM done: text=' + (msg.text || '').substring(0, 80) + ' total_tokens=' + (msg.total_tokens || '?'));
            // Don't addMessage — streaming tokens already built the display
            ttsText = '';
        }

        if (msg.type === 'llm_cancelled') {
            if (ttsAbortController) ttsAbortController.abort();
            ttsText = '';
            log('LLM cancelled: partial=' + (msg.partial_text || '').substring(0, 50));
        }

        if (msg.type === 'llm_error') {
            log('LLM ERROR: ' + (msg.error || 'unknown'));
        }
    }

    async function playTTS(url, signal) {
        try {
            log('Fetching TTS: ' + url.substring(0, 80) + '...');
            const response = await fetch(url, { signal });
            if (!response.ok) throw new Error('TTS fetch failed: ' + response.status);
            const reader = response.body.getReader();
            audioChunks = [];
            if (!audioContext) audioContext = new AudioContext({ sampleRate: 24000 });

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                audioChunks.push(value);
                log('TTS chunk: ' + value.byteLength + ' bytes');
            }
            log('TTS done, playing...');

            // Play all chunks
            for (const chunk of audioChunks) {
                const buf = await audioContext.decodeAudioData(chunk);
                const source = audioContext.createBufferSource();
                source.buffer = buf;
                source.connect(audioContext.destination);
                source.start();
                await new Promise(r => source.onended = r);
            }
            log('Playback done');
        } catch (e) {
            if (e.name !== 'AbortError') log('TTS error: ' + e.message);
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
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            recordingStream = stream;

            // Resume AudioContext if suspended
            if (!audioContext || audioContext.state === 'closed') {
                audioContext = new AudioContext({ sampleRate: 24000 });
            } else if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }

            // Create and configure script processor
            if (scriptProcessor) {
                scriptProcessor.disconnect();
            }
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
            log('Using onaudioprocess for raw PCM at ' + audioContext.sampleRate + ' Hz');

            const localIsRecording = { value: true };

            scriptProcessor.onaudioprocess = (e) => {
                if (!localIsRecording.value) return;
                const inputData = e.inputBuffer.getChannelData(0); // Float32 mono
                // Convert Float32 [-1,1] -> Int16 [−32768, 32767]
                const int16 = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    int16[i] = s < 0 ? s * 32768 : s * 32767;
                }
                // Accumulate for later sending
                accumulatedChunks.push(int16.buffer);
            };

            // Connect the nodes (needed to start processing)
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(scriptProcessor);
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
            recordBtn.textContent = '🔴 錄音中...';
            commitBtn.disabled = false;
            cancelBtn.disabled = false;
            log('Recording started (raw PCM via onaudioprocess)');
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
        recordBtn.textContent = '🎤 按住說話';

        // Send accumulated audio as one combined ArrayBuffer
        if (accumulatedChunks.length > 0) {
            log('Sending accumulated audio: ' + accumulatedChunks.length + ' chunks');

            // Combine all Int16Array chunks into one
            let totalSamples = 0;
            for (const chunk of accumulatedChunks) {
                totalSamples += chunk.byteLength / 2;
            }
            const combined = new Int16Array(totalSamples);
            let offset = 0;
            for (const chunk of accumulatedChunks) {
                const arr = new Int16Array(chunk);
                combined.set(arr, offset);
                offset += arr.length;
            }
            log('Combined PCM: ' + combined.length + ' samples, ' + combined.byteLength + ' bytes');
            ws.send(combined.buffer);
            accumulatedChunks = [];
        } else {
            log('No audio accumulated');
        }

        setStatus(ws && ws.readyState === WebSocket.OPEN ? 'connected' : 'disconnected');
    }

    recordBtn.addEventListener('mousedown', startRecording);
    recordBtn.addEventListener('mouseup', stopRecordingAndSend);
    recordBtn.addEventListener('mouseleave', stopRecordingAndSend);
    recordBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
    recordBtn.addEventListener('touchend', stopRecordingAndSend);

    window.__commitBtnClicked = function() {
        log('COMMIT BTN CLICKED: isRecording=' + isRecording + ', ws.readyState=' + (ws ? ws.readyState : 'null') + ', chunks=' + accumulatedChunks.length);
        if (isRecording) {
            log('COMMIT: stopping recording');
            stopRecordingAndSend();
        } else if (accumulatedChunks.length > 0) {
            log('COMMIT: sending pending audio...');
            let totalSamples = 0;
            for (const chunk of accumulatedChunks) totalSamples += chunk.byteLength / 2;
            const combined = new Int16Array(totalSamples);
            let offset = 0;
            for (const chunk of accumulatedChunks) { combined.set(new Int16Array(chunk), offset); offset += new Int16Array(chunk).length; }
            ws.send(combined.buffer);
            accumulatedChunks = [];
        }
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'control', action: 'commit_utterance' }));
            log('commit_utterance SENT');
        } else {
            log('WS NOT OPEN, cannot send commit');
        }
        commitBtn.disabled = true;
    };

    commitBtn.addEventListener('click', window.__commitBtnClicked);

    window.__cancelBtnClicked = function() {
        log('CANCEL BTN CLICKED');
        if (isRecording) stopRecordingAndSend();
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'control', action: 'cancel' }));
            log('cancel SENT');
        }
        cancelBtn.disabled = true;
    };
    cancelBtn.addEventListener('click', window.__cancelBtnClicked);

    listenerEl.addEventListener('change', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'config',
                audio: { sample_rate: 24000, channels: 1, format: 'pcm' },
                persona_id: personaEl.value,
                listener_id: listenerEl.value,
                model: 'gpt-4o-mini'
            }));
        }
    });

    // Auto-connect
    connect();
    </script>
</body>
</html>
"""


@router.get("/ui")
async def serve_ui():
    """Serve the standalone voice AI UI."""
    return HTMLResponse(UI_HTML)
