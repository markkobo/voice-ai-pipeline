"""
Standalone Recordings UI page.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])


@router.get("/ui/recordings")
async def recordings_page():
    """Recordings management page."""
    html = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>錄音管理 - Voice AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }

        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }
        .header h1 { font-size: 1.5rem; color: #fff; }
        .back-btn {
            background: #444;
            color: #fff;
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            text-decoration: none;
        }
        .back-btn:hover { background: #555; }

        /* Version selector */
        .version-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
        }
        .version-selector select {
            background: #333;
            color: #fff;
            border: 1px solid #555;
            padding: 6px 12px;
            border-radius: 6px;
        }

        /* Recording Card */
        .recording-section {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 1.1rem;
            margin-bottom: 15px;
            color: #fff;
        }

        /* WebRTC Recording */
        .recorder {
            background: #0f3460;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
        }
        .db-meter {
            height: 40px;
            background: #1a1a2e;
            border-radius: 6px;
            margin-bottom: 10px;
            overflow: hidden;
            position: relative;
        }
        .db-meter-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #00ccff, #ffcc00, #ff4444);
            width: 0%;
            transition: width 0.1s;
        }
        .db-info {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: #aaa;
            margin-bottom: 10px;
        }
        .recorder-controls {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .rec-btn {
            background: #e94560;
            color: #fff;
            border: none;
            padding: 12px 24px;
            border-radius: 50px;
            font-size: 1rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .rec-btn.recording {
            background: #ff4444;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .stop-btn {
            background: #444;
            color: #fff;
            border: none;
            padding: 12px 24px;
            border-radius: 50px;
            font-size: 1rem;
            cursor: pointer;
            display: none;
        }
        .stop-btn.visible { display: block; }
        .duration {
            font-size: 1.2rem;
            font-family: monospace;
            color: #fff;
        }

        /* Metadata selectors */
        .metadata-selectors {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }
        .selector-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .selector-group label {
            font-size: 0.8rem;
            color: #888;
        }
        .selector-group select {
            background: #1a1a2e;
            color: #fff;
            border: 1px solid #333;
            padding: 8px 12px;
            border-radius: 6px;
            min-width: 120px;
        }

        /* Upload area */
        .upload-area {
            border: 2px dashed #333;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            margin-bottom: 15px;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        .upload-area:hover { border-color: #00ccff; }
        .upload-area input { display: none; }
        .upload-icon { font-size: 2rem; margin-bottom: 10px; }
        .upload-text { color: #888; }
        .upload-text strong { color: #00ccff; }

        /* Debug Panel */
        .debug-panel {
            background: #0d1b2a;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            font-family: monospace;
            font-size: 0.8rem;
            max-height: 200px;
            overflow-y: auto;
        }
        .debug-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #333;
        }
        .debug-toggle {
            background: #333;
            color: #fff;
            border: none;
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .log-entry {
            padding: 2px 0;
            color: #aaa;
        }
        .log-entry.info { color: #00ccff; }
        .log-entry.warning { color: #ffcc00; }
        .log-entry.error { color: #ff4444; }
        .log-entry.debug { color: #888; }
        .log-time { color: #666; margin-right: 8px; }
        .log-component { color: #00ff88; margin-right: 8px; }

        /* Recordings List */
        .recordings-list {
            margin-top: 20px;
        }
        .recording-card {
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .recording-info {
            flex: 1;
        }
        .recording-title {
            font-weight: 500;
            margin-bottom: 5px;
        }
        .recording-meta {
            font-size: 0.85rem;
            color: #888;
            display: flex;
            gap: 15px;
        }
        .recording-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            background: #333;
        }
        .recording-badge.ok { background: #00ff8833; color: #00ff88; }
        .recording-badge.warning { background: #ffcc0033; color: #ffcc00; }
        .recording-badge.processing { background: #00ccff33; color: #00ccff; }
        .recording-badge.failed { background: #ff444433; color: #ff4444; }
        .recording-actions {
            display: flex;
            gap: 8px;
        }
        .action-btn {
            background: #333;
            color: #fff;
            border: none;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .action-btn:hover { background: #444; }
        .action-btn.play { background: #00ccff; }
        .action-btn.play:hover { background: #00aadd; }
        .action-btn.delete { background: #ff4444; }
        .action-btn.delete:hover { background: #dd3333; }

        /* Quality indicator */
        .quality-info {
            font-size: 0.8rem;
            color: #888;
            margin-top: 5px;
        }
        .quality-good { color: #00ff88; }
        .quality-bad { color: #ff4444; }

        /* Inline edit */
        .edit-btn {
            background: #333;
            color: #fff;
            border: none;
            padding: 2px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
            margin-left: 5px;
        }
        .edit-btn:hover { background: #555; }
        .inline-edit {
            background: #1a1a2e;
            color: #fff;
            border: 1px solid #00ccff;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            min-width: 100px;
        }
        .save-btn {
            background: #00ccff;
            color: #000;
            border: none;
            padding: 2px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
            margin-left: 4px;
        }
        .cancel-btn {
            background: #666;
            color: #fff;
            border: none;
            padding: 2px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
            margin-left: 4px;
        }

        /* Speaker labeling */
        .speaker-section {
            margin-top: 10px;
            padding: 10px;
            background: #0d1b2a;
            border-radius: 6px;
        }
        .speaker-section-title {
            font-size: 0.85rem;
            color: #00ccff;
            margin-bottom: 8px;
        }
        .speaker-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
            font-size: 0.85rem;
        }
        .speaker-name {
            font-family: monospace;
            color: #aaa;
            min-width: 100px;
        }
        .speaker-select {
            background: #1a1a2e;
            color: #fff;
            border: 1px solid #333;
            padding: 4px 8px;
            border-radius: 4px;
            min-width: 120px;
        }
        .speaker-audio-btn {
            background: #333;
            color: #fff;
            border: none;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .speaker-audio-btn:hover { background: #444; }

        /* Transcription */
        .transcription {
            margin-top: 10px;
            padding: 10px;
            background: #0d1b2a;
            border-radius: 6px;
            font-size: 0.9rem;
            color: #ccc;
            max-height: 100px;
            overflow-y: auto;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #666;
        }

        /* Loading spinner */
        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #333;
            border-top-color: #00ccff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Gray out during training */
        .disabled-overlay {
            opacity: 0.5;
            pointer-events: none;
        }
        .training-indicator {
            background: #ffcc00;
            color: #000;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            display: none;
        }
        .training-indicator.visible { display: inline-block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="/ui" class="back-btn">← 返回對話</a>
            <h1>錄音管理</h1>
            <div class="version-selector">
                <span>版本:</span>
                <select id="versionSelect">
                    <option value="default">系統預設</option>
                </select>
            </div>
        </div>

        <!-- WebRTC Recording Section -->
        <div class="recording-section">
            <div class="section-title">🎤 錄音 (WebRTC)</div>

            <div class="recorder" id="recorder">
                <!-- dB Meter -->
                <div class="db-meter">
                    <div class="db-meter-fill" id="dbMeterFill"></div>
                </div>
                <div class="db-info">
                    <span id="dbLevel">-∞ dB</span>
                    <span id="qualityIndicator">等待錄音...</span>
                </div>

                <!-- Metadata selectors -->
                <div class="metadata-selectors">
                    <div class="selector-group">
                        <label>👤 聆聽者</label>
                        <select id="listenerSelect">
                            <option value="child">小孩</option>
                            <option value="mom">媽媽</option>
                            <option value="dad">爸爸</option>
                            <option value="friend">朋友</option>
                            <option value="reporter">記者</option>
                            <option value="elder">長輩</option>
                            <option value="default" selected>預設</option>
                        </select>
                    </div>
                    <div class="selector-group">
                        <label>🎭 人格</label>
                        <select id="personaSelect">
                            <option value="xiao_s" selected>小S</option>
                            <option value="caregiver">照護者</option>
                            <option value="elder_gentle">長輩-溫柔</option>
                            <option value="elder_playful">長輩-活潑</option>
                        </select>
                    </div>
                </div>

                <!-- Controls -->
                <div class="recorder-controls">
                    <button class="rec-btn" id="recBtn">
                        <span id="recIcon">●</span>
                        <span id="recText">開始錄音</span>
                    </button>
                    <button class="stop-btn" id="stopBtn">■ 停止</button>
                    <span class="duration" id="duration">00:00</span>
                    <span class="training-indicator" id="trainingIndicator">Training 中...</span>
                </div>
            </div>

            <!-- File Upload -->
            <div class="upload-area" id="uploadArea">
                <input type="file" id="fileInput" accept=".wav,.mp3,.m4a,.webm" multiple>
                <div class="upload-icon">📁</div>
                <div class="upload-text">
                    拖放檔案到此處 或 <strong>選擇檔案</strong><br>
                    <small>支援: WAV, MP3, M4A, WebM (最大 50MB)</small>
                </div>
            </div>
        </div>

        <!-- Debug Panel -->
        <div class="debug-panel" id="debugPanel">
            <div class="debug-panel-header">
                <span>📋 Debug Panel</span>
                <button class="debug-toggle" id="debugToggle">折疊</button>
            </div>
            <div id="debugLogs"></div>
        </div>

        <!-- Recordings List -->
        <div class="recording-section">
            <div class="section-title">📚 錄音列表</div>
            <div id="recordingsList">
                <div class="empty-state">尚無錄音</div>
            </div>
        </div>
    </div>

    <script>
        // State
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let startTime = 0;
        let durationInterval = null;
        let audioContext = null;
        let analyser = null;
        let currentStream = null;

        // Elements
        const recBtn = document.getElementById('recBtn');
        const stopBtn = document.getElementById('stopBtn');
        const duration = document.getElementById('duration');
        const dbMeterFill = document.getElementById('dbMeterFill');
        const dbLevel = document.getElementById('dbLevel');
        const qualityIndicator = document.getElementById('qualityIndicator');
        const debugLogs = document.getElementById('debugLogs');
        const recordingsList = document.getElementById('recordingsList');
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const listenerSelect = document.getElementById('listenerSelect');
        const personaSelect = document.getElementById('personaSelect');

        // Logging
        function log(message, level = 'info', component = 'UI') {
            const time = new Date().toLocaleTimeString('zh-TW', { hour12: false });
            const entry = document.createElement('div');
            entry.className = `log-entry ${level}`;
            entry.innerHTML = `<span class="log-time">${time}</span><span class="log-component">[${component}]</span> ${message}`;
            debugLogs.appendChild(entry);
            debugLogs.scrollTop = debugLogs.debugLogsHeight;
            console.log(`[${level.toUpperCase()}] [${component}] ${message}`);
        }

        // Toggle debug panel
        document.getElementById('debugToggle').addEventListener('click', () => {
            const panel = document.getElementById('debugPanel');
            const logs = document.getElementById('debugLogs');
            if (logs.style.display === 'none') {
                logs.style.display = 'block';
                panel.style.maxHeight = '200px';
            } else {
                logs.style.display = 'none';
                panel.style.maxHeight = '40px';
            }
        });

        // Load recordings list
        async function loadRecordings() {
            try {
                const response = await fetch('/api/recordings/');
                const data = await response.json();
                // Handle both paginated {recordings: [...]} and legacy array responses
                const recordings = Array.isArray(data) ? data : (data.recordings || []);
                renderRecordings(recordings);
                applySpeakerLabelDefaults();
                log(`Loaded ${recordings.length} recordings`, 'info', 'UI');
            } catch (e) {
                log(`Failed to load recordings: ${e.message}`, 'error', 'UI');
            }
        }

        // Render recordings list
        function renderRecordings(recordings) {
            if (recordings.length === 0) {
                recordingsList.innerHTML = '<div class="empty-state">尚無錄音</div>';
                return;
            }

            recordingsList.innerHTML = recordings.map(r => {
                const statusClass = r.status === 'processed' ? 'ok' : r.status === 'failed' ? 'failed' : r.status === 'processing' ? 'processing' : '';
                const quality = r.quality_metrics;
                const qualityHtml = quality && quality.training_ready !== null
                    ? `<span class="${quality.training_ready ? 'quality-good' : 'quality-bad'}">${quality.training_ready ? '✓ 可用於訓練' : '⚠ 品質不足'}</span>`
                    : '';

                const speakers = r.speaker_segments || [];
                const uniqueSpeakers = [...new Set(speakers.map(s => s.speaker_id))].sort();
                const speakerLabels = r.speaker_labels || {};
                const speakerSectionHtml = uniqueSpeakers.length > 0 ? renderSpeakerSection(r.recording_id, uniqueSpeakers, speakerLabels) : '';

                return `
                    <div class="recording-card" data-id="${r.recording_id}">
                        <div class="recording-info">
                            <div class="recording-title">
                                ${r.title || r.folder_name || r.recording_id}
                                <span class="recording-badge ${statusClass}">${r.status}</span>
                            </div>
                            <div class="recording-meta" id="meta-${r.recording_id}">
                                <span>👤 <span id="listener-display-${r.recording_id}">${r.listener_id || 'unknown'}</span>
                                    <button class="edit-btn" onclick="showEditField('${r.recording_id}', 'listener')">✎</button>
                                </span>
                                <span>🎭 <span id="persona-display-${r.recording_id}">${r.persona_id || 'unknown'}</span>
                                    <button class="edit-btn" onclick="showEditField('${r.recording_id}', 'persona')">✎</button>
                                </span>
                                <span>⏱ ${r.duration_seconds ? r.duration_seconds.toFixed(1) + 's' : '-'}</span>
                                <span>📅 ${r.created_at ? new Date(r.created_at).toLocaleDateString('zh-TW') : '-'}</span>
                                ${uniqueSpeakers.length > 0 ? `<span>🔊 ${uniqueSpeakers.length} speakers</span>` : ''}
                            </div>
                            ${qualityHtml ? `<div class="quality-info">${qualityHtml}</div>` : ''}
                            ${r.transcription && r.transcription.text ? `<div class="transcription">${r.transcription.text.substring(0, 100)}...</div>` : ''}
                            ${speakerSectionHtml}
                        </div>
                        <div class="recording-actions">
                            ${r.status === 'processed' || r.status === 'raw' ? `<button class="action-btn play" onclick="playRecording('${r.recording_id}')">▶</button>` : ''}
                            <button class="action-btn delete" onclick="deleteRecording('${r.recording_id}')">✕</button>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Render speaker labeling section
        function renderSpeakerSection(recordingId, speakers, labels) {
            const personaOptions = `
                <option value="xiao_s">小S</option>
                <option value="caregiver">照護者</option>
                <option value="elder_gentle">長輩-溫柔</option>
                <option value="elder_playful">長輩-活潑</option>
            `;

            const speakerRows = speakers.map(s => {
                const currentLabel = labels[s] || '';
                return `
                    <div class="speaker-row">
                        <span class="speaker-name">${s}</span>
                        <select class="speaker-select" id="speaker-${s}-${recordingId}" data-initial-value="${currentLabel}" onchange="updateSpeakerLabel('${recordingId}', '${s}', this.value)">
                            <option value="">-- 未標記 --</option>
                            ${personaOptions}
                        </select>
                        <button class="speaker-audio-btn" onclick="playSpeakerAudio('${recordingId}', '${s}')">▶ 播放</button>
                    </div>
                `;
            }).join('');

            return `
                <div class="speaker-section">
                    <div class="speaker-section-title">🔊 說話者標記</div>
                    ${speakerRows}
                </div>
            `;
        }

        // Apply initial speaker label values after rendering
        function applySpeakerLabelDefaults() {
            document.querySelectorAll('.speaker-select[data-initial-value]').forEach(select => {
                const initialValue = select.getAttribute('data-initial-value');
                if (initialValue) {
                    select.value = initialValue;
                }
            });
        }

        // Show inline edit field
        function showEditField(recordingId, field) {
            const displayEl = document.getElementById(`${field}-display-${recordingId}`);
            if (!displayEl) return;

            const currentValue = displayEl.textContent.trim();
            const options = field === 'listener'
                ? `<option value="child">child</option><option value="mom">mom</option><option value="dad">dad</option><option value="friend">friend</option><option value="reporter">reporter</option><option value="elder">elder</option><option value="default">default</option>`
                : `<option value="xiao_s">xiao_s</option><option value="caregiver">caregiver</option><option value="elder_gentle">elder_gentle</option><option value="elder_playful">elder_playful</option>`;

            displayEl.innerHTML = `
                <select class="inline-edit" id="edit-${field}-${recordingId}">
                    ${options}
                </select>
                <button class="save-btn" onclick="saveEdit('${recordingId}', '${field}')">儲存</button>
                <button class="cancel-btn" onclick="cancelEdit('${recordingId}', '${field}', '${currentValue}')">取消</button>
            `;
            document.getElementById(`edit-${field}-${recordingId}`).value = currentValue;
        }

        // Save edited field
        async function saveEdit(recordingId, field) {
            const newValue = document.getElementById(`edit-${field}-${recordingId}`).value;
            try {
                const response = await fetch(`/api/recordings/${recordingId}`, {
                    method: 'PATCH',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({[field === 'listener' ? 'listener_id' : 'persona_id']: newValue})
                });
                if (!response.ok) throw new Error('Update failed');
                loadRecordings();
                log(`Updated ${field} to ${newValue}`, 'info', 'UI');
            } catch (e) {
                log(`Update failed: ${e.message}`, 'error', 'UI');
            }
        }

        // Cancel edit
        function cancelEdit(recordingId, field, originalValue) {
            const displayEl = document.getElementById(`${field}-display-${recordingId}`);
            if (displayEl) {
                displayEl.textContent = originalValue;
            }
        }

        // Update speaker label
        async function updateSpeakerLabel(recordingId, speakerId, personaId) {
            try {
                // Get current labels
                const response = await fetch(`/api/recordings/${recordingId}/speakers`);
                const info = await response.json();
                const labels = {...info.speaker_labels};
                labels[speakerId] = personaId || undefined;

                // Remove undefined keys
                Object.keys(labels).forEach(k => labels[k] === undefined && delete labels[k]);

                // Update
                const updateResponse = await fetch(`/api/recordings/${recordingId}/speakers`, {
                    method: 'PATCH',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({speaker_labels: labels})
                });
                if (!updateResponse.ok) throw new Error('Update failed');
                log(`Labeled ${speakerId} as ${personaId || '(unlabeled)'}`, 'info', 'UI');
            } catch (e) {
                log(`Speaker label update failed: ${e.message}`, 'error', 'UI');
            }
        }

        // Play speaker audio
        async function playSpeakerAudio(recordingId, speakerId) {
            log(`Playing speaker audio: ${speakerId}`, 'info', 'PLAYBACK');
            try {
                const response = await fetch(`/api/recordings/${recordingId}/speaker/${speakerId}/audio`);
                if (!response.ok) throw new Error('Speaker audio not available');
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                audio.play();
            } catch (e) {
                log(`Speaker playback failed: ${e.message}`, 'error', 'PLAYBACK');
            }
        }

        // Play recording
        async function playRecording(recordingId) {
            log(`Playing recording: ${recordingId}`, 'info', 'PLAYBACK');
            try {
                const response = await fetch(`/api/recordings/${recordingId}/stream?stage=enhanced`);
                if (!response.ok) throw new Error('Stream not available');
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                audio.play();
                log(`Playing: ${url}`, 'debug', 'PLAYBACK');
            } catch (e) {
                log(`Playback failed: ${e.message}`, 'error', 'PLAYBACK');
            }
        }

        // Delete recording
        async function deleteRecording(recordingId) {
            if (!confirm('確定要刪除這個錄音嗎？')) return;
            log(`Deleting recording: ${recordingId}`, 'info', 'UI');
            try {
                const response = await fetch(`/api/recordings/${recordingId}`, { method: 'DELETE' });
                if (!response.ok) throw new Error('Delete failed');
                loadRecordings();
                log(`Deleted: ${recordingId}`, 'info', 'UI');
            } catch (e) {
                log(`Delete failed: ${e.message}`, 'error', 'UI');
            }
        }

        // Start recording
        async function startRecording() {
            log('Starting WebRTC recording...', 'info', 'RECORDING');
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                currentStream = stream;

                // Setup audio context for dB meter
                audioContext = new AudioContext();
                analyser = audioContext.createAnalyser();
                const source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);
                analyser.fftSize = 256;

                // Setup MediaRecorder
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                audioChunks = [];

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        audioChunks.push(e.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    log('Recording stopped, processing...', 'info', 'RECORDING');
                    const blob = new Blob(audioChunks, { type: 'audio/webm' });
                    await uploadBlob(blob);
                };

                mediaRecorder.start(100); // Collect data every 100ms
                isRecording = true;
                startTime = Date.now();

                // Update UI
                recBtn.classList.add('recording');
                recBtn.querySelector('#recText').textContent = '錄音中...';
                stopBtn.classList.add('visible');
                qualityIndicator.textContent = '錄音中...';

                // Start duration timer
                durationInterval = setInterval(() => {
                    const elapsed = Math.floor((Date.now() - startTime) / 1000);
                    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
                    const secs = (elapsed % 60).toString().padStart(2, '0');
                    duration.textContent = `${mins}:${secs}`;

                    // Max 5 minutes
                    if (elapsed >= 300) {
                        stopRecording();
                    }
                }, 1000);

                // Start dB meter
                updateDbMeter();

                log('Recording started', 'info', 'RECORDING');
            } catch (e) {
                log(`Failed to start recording: ${e.message}`, 'error', 'RECORDING');
                alert('無法訪問麥克風: ' + e.message);
            }
        }

        // Update dB meter
        function updateDbMeter() {
            if (!isRecording || !analyser) return;

            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(dataArray);

            // Calculate RMS
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i] * dataArray[i];
            }
            const rms = Math.sqrt(sum / dataArray.length);
            const db = 20 * Math.log10(rms / 255);

            // Update UI
            const percent = Math.min(100, Math.max(0, (db + 60) * 100 / 60));
            dbMeterFill.style.width = percent + '%';
            dbLevel.textContent = db > -60 ? `${db.toFixed(1)} dB` : '-∞ dB';

            // Quality indicator
            if (db > -12 && db < 0) {
                qualityIndicator.textContent = '✓ 音量良好';
                qualityIndicator.style.color = '#00ff88';
            } else if (db >= 0) {
                qualityIndicator.textContent = '⚠ 音量太大';
                qualityIndicator.style.color = '#ff4444';
            } else if (db < -30) {
                qualityIndicator.textContent = '⚠ 音量太小';
                qualityIndicator.style.color = '#ffcc00';
            } else {
                qualityIndicator.textContent = '錄音中...';
                qualityIndicator.style.color = '#888';
            }

            requestAnimationFrame(updateDbMeter);
        }

        // Stop recording
        function stopRecording() {
            if (!isRecording) return;

            log('Stopping recording...', 'info', 'RECORDING');
            isRecording = false;

            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }

            // Stop all tracks
            if (currentStream) {
                currentStream.getTracks().forEach(track => track.stop());
            }

            // Cleanup
            if (durationInterval) {
                clearInterval(durationInterval);
                durationInterval = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }

            // Reset UI
            recBtn.classList.remove('recording');
            recBtn.querySelector('#recText').textContent = '開始錄音';
            stopBtn.classList.remove('visible');
            duration.textContent = '00:00';
            dbMeterFill.style.width = '0%';
            dbLevel.textContent = '-∞ dB';
            qualityIndicator.textContent = '等待錄音...';
            qualityIndicator.style.color = '#888';
        }

        // Upload blob as file
        async function uploadBlob(blob) {
            const listenerId = listenerSelect.value;
            const personaId = personaSelect.value;
            const durationSec = (Date.now() - startTime) / 1000;

            log(`Uploading: listener=${listenerId}, persona=${personaId}, duration=${durationSec.toFixed(1)}s`, 'info', 'UPLOAD');

            const formData = new FormData();
            formData.append('file', blob, 'recording.webm');
            formData.append('listener_id', listenerId);
            formData.append('persona_id', personaId);

            try {
                const response = await fetch('/api/recordings/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Upload failed');
                }

                const result = await response.json();
                log(`Upload complete: recording_id=${result.recording_id}`, 'info', 'UPLOAD');
                loadRecordings();

                // Trigger processing
                await triggerProcessing(result.recording_id);

            } catch (e) {
                log(`Upload failed: ${e.message}`, 'error', 'UPLOAD');
                alert('上傳失敗: ' + e.message);
            }
        }

        // Trigger processing
        async function triggerProcessing(recordingId) {
            log(`Triggering processing: ${recordingId}`, 'info', 'PIPELINE');
            try {
                const response = await fetch(`/api/recordings/${recordingId}/process`, {
                    method: 'POST'
                });
                const result = await response.json();
                log(`Processing started: ${result.status}`, 'info', 'PIPELINE');
            } catch (e) {
                log(`Processing trigger failed: ${e.message}`, 'error', 'PIPELINE');
            }
        }

        // File upload handlers
        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#00ccff';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#333';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#333';
            handleFiles(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', () => {
            handleFiles(fileInput.files);
        });

        async function handleFiles(files) {
            for (const file of files) {
                log(`Uploading file: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`, 'info', 'UPLOAD');
                const formData = new FormData();
                formData.append('file', file);
                formData.append('listener_id', listenerSelect.value);
                formData.append('persona_id', personaSelect.value);

                try {
                    const response = await fetch('/api/recordings/upload', {
                        method: 'POST',
                        body: formData
                    });
                    if (!response.ok) throw new Error('Upload failed');
                    const result = await response.json();
                    log(`File uploaded: ${result.recording_id}`, 'info', 'UPLOAD');
                    await triggerProcessing(result.recording_id);
                } catch (e) {
                    log(`File upload failed: ${e.message}`, 'error', 'UPLOAD');
                }
            }
            loadRecordings();
        }

        // Event listeners
        recBtn.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });

        stopBtn.addEventListener('click', stopRecording);

        // Load data on page load
        loadRecordings();
        log('Recordings page loaded', 'info', 'UI');

        // Auto-refresh recordings list every 5 seconds when page is visible
        // This ensures processing status updates automatically
        setInterval(() => {
            if (!document.hidden) {
                loadRecordings();
            }
        }, 5000);

        // Load personas/listeners for potential future use
        fetch('/api/personas').then(r => r.json()).then(data => {
            log(`Loaded ${data.length} personas`, 'debug', 'UI');
        });
        fetch('/api/listeners').then(r => r.json()).then(data => {
            log(`Loaded ${data.length} listeners`, 'debug', 'UI');
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)
