"""
Gradio UI for Voice AI Streaming Chat.

Provides a browser-based UI for real-time voice conversation with:
- Listener/persona selection
- VAD sensitivity control
- Continuous streaming with VAD auto-commit
- Debug mode with full pipeline visibility
- Human-readable log viewer
"""
import json
from pathlib import Path
from typing import Optional

import gradio as gr

from app.logging_config import get_logger

log = get_logger(__name__, component="gradio")

# Listeners and personas available
LISTENERS = ["child", "mom", "reporter", "friend", "stranger", "default"]
PERSONAS = ["xiao_s"]
VAD_SENSITIVITIES = ["low", "medium", "high"]
LLM_MODELS = ["gpt-4o-mini", "gpt-4o"]
TTS_MODELS = ["0.6B", "1.7B"]


# JavaScript for WebSocket communication and audio handling
# Handles: WS connection, WebRTC mic recording, audio playback
WEBSOCKET_JS = """
<script>
(function() {
    // Globals
    let ws = null;
    let mediaRecorder = null;
    let audioContext = null;
    let audioChunks = [];
    let isRecording = false;
    let currentStreamUrl = null;
    let activeAbortController = null;
    let ttsAudioElement = null;

    // Connect to WebSocket
    function connectWS() {
        if (ws && ws.readyState === WebSocket.OPEN) return;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = protocol + '//' + window.location.host + '/ws/asr';

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('WS connected');
            updateStatus('Connected', 'green');

            // Send config
            const listenerId = getSelectedValue('listener-select');
            const personaId = getSelectedValue('persona-select');
            const llmModel = getSelectedValue('llm-model-select');

            ws.send(JSON.stringify({
                type: 'config',
                audio: {sample_rate: 24000, channels: 1, format: 'pcm'},
                persona_id: personaId,
                listener_id: listenerId,
                model: llmModel
            }));
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleServerMessage(data);
            } catch(e) {
                console.error('WS parse error:', e);
            }
        };

        ws.onclose = () => {
            console.log('WS closed');
            updateStatus('Disconnected', 'gray');
            setTimeout(connectWS, 2000);
        };

        ws.onerror = (e) => {
            console.error('WS error:', e);
            updateStatus('Error', 'red');
        };
    }

    function getSelectedValue(id) {
        const el = document.getElementById(id);
        return el ? el.value : null;
    }

    // Handle incoming server messages
    function handleServerMessage(data) {
        switch(data.type) {
            case 'asr_result':
                appendDebug('ASR', data.text || '(empty)', data.telemetry || {});
                if (data.is_final && data.text) {
                    appendConversation('user', data.text);
                }
                break;

            case 'llm_start':
                appendDebug('LLM', '[START] Generating response...', {});
                break;

            case 'llm_token':
                if (data.emotion) {
                    appendDebug('EMOTION', data.emotion, {instruct: data.instruct || ''});
                }
                // LLM text appears in debug only, not in main conversation
                break;

            case 'tts_ready':
                // Trigger TTS HTTP fetch
                if (data.stream_url) {
                    fetchTTSAudio(data.stream_url, data.emotion);
                    appendDebug('TTS', `[READY] emotion=${data.emotion}`, {
                        instruct: data.instruct || '',
                        text: data.text || ''
                    });
                }
                break;

            case 'llm_done':
                appendDebug('LLM DONE', data.text, data.telemetry || {});
                appendConversation('ai', data.text);
                break;

            case 'llm_cancelled':
                appendDebug('LLM', '[CANCELLED] Interrupted', {});
                stopTTSAudio();
                break;

            case 'vad_commit':
                appendDebug('VAD', 'Utterance committed', data.telemetry || {});
                break;

            case 'llm_error':
                appendDebug('LLM ERROR', data.error || 'Unknown error', {});
                break;
        }
    }

    // Fetch TTS audio from HTTP endpoint and play
    function fetchTTSAudio(streamUrl, emotion) {
        // Cancel any existing fetch
        if (activeAbortController) {
            activeAbortController.abort();
        }
        stopTTSAudio();

        activeAbortController = new AbortController();
        const signal = activeAbortController.signal;

        // Convert relative URL to absolute
        const baseUrl = window.location.origin;
        const fullUrl = streamUrl.startsWith('http') ? streamUrl : baseUrl + streamUrl;

        console.log('Fetching TTS from:', fullUrl);

        fetch(fullUrl, {signal})
            .then(response => {
                if (!response.ok) {
                    throw new Error('TTS fetch failed: ' + response.status);
                }
                return response.body;
            })
            .then(body => {
                if (!body) throw new Error('No response body');

                const reader = body.getReader();
                const chunks = [];

                function push() {
                    reader.read().then(({done, value}) => {
                        if (done) {
                            // All chunks received
                            if (chunks.length > 0) {
                                playPCMChunks(chunks);
                            }
                            return;
                        }
                        chunks.push(value);
                        // Play chunks as they arrive (progressive playback)
                        playPCMChunks([value]);
                        push();
                    }).catch(err => {
                        if (err.name !== 'AbortError') {
                            console.error('TTS read error:', err);
                        }
                    });
                }

                push();
            })
            .catch(err => {
                if (err.name !== 'AbortError') {
                    console.error('TTS fetch error:', err);
                    appendDebug('TTS ERROR', err.message, {});
                }
            });
    }

    // Play raw PCM chunks
    function playPCMChunks(chunks) {
        audioContext = audioContext || new AudioContext({sampleRate: 24000});

        for (const chunk of chunks) {
            const pcmData = new Int16Array(chunk.buffer, chunk.byteOffset, chunk.byteLength / 2);
            const floatData = new Float32Array(pcmData.length);
            for (let i = 0; i < pcmData.length; i++) {
                floatData[i] = pcmData[i] / 32768.0;
            }

            const buffer = audioContext.createBuffer(1, floatData.length, 24000);
            buffer.copyToChannel(floatData, 0);

            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            source.start();
        }
    }

    function stopTTSAudio() {
        if (activeAbortController) {
            activeAbortController.abort();
            activeAbortController = null;
        }
    }

    function appendDebug(component, message, telemetry) {
        const panel = document.getElementById('debug-content');
        if (!panel) return;

        const entry = {
            component,
            message,
            telemetry,
            time: new Date().toISOString()
        };

        const div = document.createElement('div');
        div.className = 'debug-entry';
        const ttyJson = JSON.stringify(telemetry || {});
        div.innerHTML = `<strong style="color:#0066cc">${component}</strong>: ${escapeHtml(message)} <span style="color:#888;font-size:11px">${escapeHtml(ttyJson)}</span>`;
        panel.appendChild(div);
        panel.scrollTop = panel.scrollHeight;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function appendConversation(role, text) {
        const box = document.getElementById('conversation-display');
        if (!box) return;

        const msgClass = role === 'user' ? 'user-msg' : 'ai-msg';
        const label = role === 'user' ? 'You' : '小S';

        const div = document.createElement('div');
        div.innerHTML = `<span class="${msgClass}"><strong>${label}:</strong> ${escapeHtml(text)}</span>`;
        box.appendChild(div);
        box.scrollTop = box.scrollHeight;
    }

    function updateStatus(text, color) {
        const el = document.getElementById('status-indicator');
        if (el) {
            el.textContent = text;
            el.style.color = color;
        }
    }

    // Recording
    async function startRecording() {
        if (isRecording) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 24000,
                    channels: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            // Use webm/opus for recording (browser-native, good compression)
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            mediaRecorder = new MediaRecorder(stream, {mimeType});
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(audioChunks, {type: mimeType});
                blob.arrayBuffer().then(buffer => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(buffer);
                    }
                });
                stream.getTracks().forEach(t => t.stop());
                audioChunks = [];
            };

            // Collect chunks every 100ms for lower latency
            mediaRecorder.start(100);
            isRecording = true;
            updateStatus('Recording...', 'orange');

            // Update recording indicator
            const indicator = document.getElementById('recording-indicator');
            if (indicator) indicator.classList.add('active');

        } catch(e) {
            console.error('Recording error:', e);
            alert('Could not start recording: ' + e.message);
            updateStatus('Error', 'red');
        }
    }

    function stopRecording() {
        if (!isRecording || !mediaRecorder) return;
        mediaRecorder.stop();
        isRecording = false;
        updateStatus('Processing...', 'yellow');

        const indicator = document.getElementById('recording-indicator');
        if (indicator) indicator.classList.remove('active');
    }

    function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    function commitUtterance() {
        stopRecording();
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({type: 'control', action: 'commit_utterance'}));
            updateStatus('Committing...', 'yellow');
        }
    }

    function cancelOp() {
        stopRecording();
        stopTTSAudio();
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({type: 'control', action: 'cancel'}));
            updateStatus('Cancelled', 'orange');
        }
    }

    // Init on load
    window.addEventListener('DOMContentLoaded', () => {
        connectWS();

        const recordBtn = document.getElementById('record-btn');
        if (recordBtn) recordBtn.addEventListener('click', toggleRecording);

        const commitBtn = document.getElementById('commit-btn');
        if (commitBtn) commitBtn.addEventListener('click', commitUtterance);

        const cancelBtn = document.getElementById('cancel-btn');
        if (cancelBtn) cancelBtn.addEventListener('click', cancelOp);

        // Reconnect on config change
        ['listener-select', 'persona-select', 'llm-model-select'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', () => {
                if (ws) ws.close();
                connectWS();
            });
        });
    });
})();
"""


CSS = """
#gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.debug-entry {
    padding: 4px 8px;
    border-bottom: 1px solid #eee;
    font-family: monospace;
    font-size: 13px;
}
#status-indicator {
    font-weight: bold;
    font-size: 16px;
}
.conversation-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    min-height: 200px;
    max-height: 400px;
    overflow-y: auto;
}
.user-msg { color: #0066cc; font-weight: bold; margin: 8px 0; }
.ai-msg { color: #cc0000; font-weight: bold; margin: 8px 0; }
.controls {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}
.recording-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #ccc;
    display: inline-block;
}
.recording-indicator.active {
    background: #ff0000;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
"""


def build_ui() -> gr.Blocks:
    """Build the Gradio UI blocks."""

    with gr.Blocks(title="Voice AI — Streaming Chat", css=CSS) as ui:
        gr.Markdown("## 🎙️ Voice AI — Streaming Chat")
        gr.Markdown("*小S 語音助理 — 選擇你的對象，開始對話吧！*")

        # Status
        status = gr.HTML(
            '<span id="status-indicator" style="color:gray">Disconnected</span>'
        )

        with gr.Row():
            with gr.Column(scale=1):
                listener = gr.Dropdown(
                    choices=LISTENERS,
                    value="default",
                    label="對象 (Listener)",
                    info="選擇 AI 說話的對象"
                )
            with gr.Column(scale=1):
                persona = gr.Dropdown(
                    choices=PERSONAS,
                    value="xiao_s",
                    label="人格 (Persona)",
                    info="選擇 AI 的人格"
                )
            with gr.Column(scale=1):
                vad_sensitivity = gr.Dropdown(
                    choices=VAD_SENSITIVITIES,
                    value="medium",
                    label="VAD 靈敏度",
                    info="low=安靜環境, high=吵雜環境"
                )

        with gr.Row():
            with gr.Column(scale=1):
                llm_model = gr.Dropdown(
                    choices=LLM_MODELS,
                    value="gpt-4o-mini",
                    label="LLM Model"
                )
            with gr.Column(scale=1):
                tts_model = gr.Dropdown(
                    choices=TTS_MODELS,
                    value="0.6B",
                    label="TTS Model",
                    info="0.6B=快速, 1.7B=高品質"
                )
            with gr.Column(scale=1):
                debug_mode = gr.Checkbox(
                    value=False,
                    label="Debug Mode",
                    info="顯示完整 pipeline 資訊"
                )

        # Recording controls
        with gr.Group():
            gr.HTML("""
            <div class="controls">
                <span class="recording-indicator" id="recording-indicator"></span>
                <button id="record-btn" class="gr-button gr-button-primary">🎤 按住說話</button>
                <button id="commit-btn" class="gr-button">✋ 強制結束</button>
                <button id="cancel-btn" class="gr-button gr-button-secondary">⏹ 取消</button>
            </div>
            """)
        gr.HTML('<small style="color:#888">按下麥克風說話，VAD 自動偵測停頓並送出。<br>'
                '也可以手動按「強制結束」立刻送出。</small>')

        # Conversation display
        with gr.Group():
            gr.Markdown("### 💬 對話")
            conversation = gr.HTML(
                '<div class="conversation-box" id="conversation-box">'
                '<div id="conversation-display" style="padding:8px">'
                '<div style="color:#888">開始說話吧...</div>'
                '</div>'
                '</div>'
            )

        # Debug panel
        with gr.Group(visible=False) as debug_section:
            gr.Markdown("### 🔍 Debug Panel")
            gr.HTML(
                '<div id="debug-panel" style="background:#f0f0f0;border-radius:8px;padding:8px;'
                'max-height:300px;overflow-y:auto;font-family:monospace;font-size:13px;">'
                '<div id="debug-content"></div>'
                '</div>'
            )

        # Log viewer
        with gr.Group(visible=False) as log_section:
            gr.Markdown("### 📋 Log Viewer")
            gr.HTML(
                '<div id="log-viewer" style="background:#1e1e1e;color:#ddd;border-radius:8px;padding:8px;'
                'max-height:300px;overflow-y:auto;font-family:monospace;font-size:12px;">'
                '<div style="color:#888">Logs appear here...</div>'
                '</div>',
                elem_id="log-viewer"
            )

        # Debug mode toggle
        debug_mode.change(
            fn=lambda show: gr.update(visible=show),
            inputs=[debug_mode],
            outputs=[debug_section],
        )

        # Inject WebSocket JS
        gr.HTML(f'<script>{WEBSOCKET_JS}</script>')

    return ui
