"""
Training UI page.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])


@router.get("/ui/training")
async def training_page():
    """Training management page."""
    html = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>訓練管理 - Voice AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1100px; margin: 0 auto; }

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

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            background: #333;
            border: none;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            font-size: 1rem;
        }
        .tab.active {
            background: #00ccff;
            color: #000;
        }

        .section {
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

        .form-row {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .form-group label {
            font-size: 0.85rem;
            color: #888;
        }
        .form-group select, .form-group input {
            background: #1a1a2e;
            color: #fff;
            border: 1px solid #333;
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 1rem;
        }

        .recording-card {
            background: #0d1b2a;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            cursor: pointer;
            border: 2px solid transparent;
        }
        .recording-card.selected {
            border-color: #00ccff;
        }
        .recording-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .recording-title {
            font-weight: 500;
        }
        .recording-meta {
            font-size: 0.85rem;
            color: #888;
            margin-top: 5px;
        }
        .speaker-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 8px;
            font-size: 0.9rem;
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
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85rem;
        }
        .speaker-checkbox {
            margin-right: 5px;
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

        .training-settings {
            background: #0d1b2a;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .training-settings h3 {
            font-size: 1rem;
            margin-bottom: 10px;
            color: #00ccff;
        }
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }

        .training-preview {
            background: #0d1b2a;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .preview-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #333;
        }
        .preview-item:last-child { border-bottom: none; }
        .preview-total {
            display: flex;
            justify-content: space-between;
            font-weight: 500;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 2px solid #00ccff;
        }

        .btn-primary {
            background: #00ccff;
            color: #000;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            width: 100%;
        }
        .btn-primary:hover { background: #00aadd; }
        .btn-primary:disabled {
            background: #555;
            color: #888;
            cursor: not-allowed;
        }

        .progress-section {
            background: #0d1b2a;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            display: none;
        }
        .progress-section.visible { display: block; }
        .progress-bar {
            height: 30px;
            background: #1a1a2e;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ccff, #00ff88);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            color: #000;
        }
        .progress-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            text-align: center;
        }
        .stat-item {
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
        }
        .stat-label {
            font-size: 0.8rem;
            color: #888;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 1.2rem;
            font-weight: 500;
        }

        .version-card {
            background: #0d1b2a;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .version-card.active {
            border: 2px solid #00ff88;
        }
        .version-info {
            flex: 1;
        }
        .version-name {
            font-weight: 500;
            margin-bottom: 5px;
        }
        .version-meta {
            font-size: 0.85rem;
            color: #888;
        }
        .version-actions {
            display: flex;
            gap: 10px;
        }
        .btn-activate {
            background: #00ff88;
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
        }
        .btn-activate:hover { background: #00dd77; }
        .btn-delete {
            background: #ff4444;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
        }
        .btn-delete:hover { background: #dd3333; }

        .empty-state {
            text-align: center;
            padding: 40px;
            color: #666;
        }

        .debug-panel {
            background: #0d1b2a;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            font-family: monospace;
            font-size: 0.8rem;
            max-height: 150px;
            overflow-y: auto;
        }
        .log-entry { padding: 2px 0; }
        .log-entry.info { color: #00ccff; }
        .log-entry.warning { color: #ffcc00; }
        .log-entry.error { color: #ff4444; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="/ui" class="back-btn">← 返回對話</a>
            <h1>訓練管理</h1>
            <a href="/ui/recordings" class="back-btn">錄音管理</a>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('training')">訓練</button>
            <button class="tab" onclick="switchTab('versions')">版本</button>
        </div>

        <!-- Training Tab -->
        <div id="training-tab">
            <!-- Training Section -->
            <div class="section">
                <div class="section-title">🎯 選擇訓練目標</div>

                <div class="form-row">
                    <div class="form-group">
                        <label>人格 (Persona)</label>
                        <select id="personaSelect">
                            <option value="xiao_s">小S</option>
                            <option value="caregiver">照護者</option>
                            <option value="elder_gentle">長輩-溫柔</option>
                            <option value="elder_playful">長輩-活潑</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Recordings Selection -->
            <div class="section">
                <div class="section-title">📁 選擇錄音</div>
                <div id="recordingsList">載入中...</div>
            </div>

            <!-- Training Settings -->
            <div class="section">
                <div class="section-title">⚙️ 訓練設定</div>
                <div class="training-settings">
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>Epochs</label>
                            <select id="epochsSelect">
                                <option value="5">5</option>
                                <option value="10" selected>10</option>
                                <option value="20">20</option>
                                <option value="30">30</option>
                                <option value="50">50</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>LoRA Rank</label>
                            <select id="rankSelect">
                                <option value="4">4</option>
                                <option value="8">8</option>
                                <option value="16" selected>16</option>
                                <option value="32">32</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Batch Size</label>
                            <select id="batchSizeSelect">
                                <option value="1">1</option>
                                <option value="2">2</option>
                                <option value="4" selected>4</option>
                                <option value="8">8</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Training Preview -->
                <div class="training-preview" id="trainingPreview">
                    <h3>📊 訓練預覽</h3>
                    <div id="previewList">
                        <div class="empty-state">選擇錄音以查看預覽</div>
                    </div>
                </div>

                <!-- Progress -->
                <div class="progress-section" id="progressSection">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%">0%</div>
                    </div>
                    <div class="progress-stats">
                        <div class="stat-item">
                            <div class="stat-label">Epoch</div>
                            <div class="stat-value" id="statEpoch">0/0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Loss</div>
                            <div class="stat-value" id="statLoss">-</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">預計剩餘</div>
                            <div class="stat-value" id="statETA">-</div>
                        </div>
                    </div>
                </div>

                <button class="btn-primary" id="startTrainingBtn" onclick="startTraining()">
                    開始訓練
                </button>
            </div>
        </div>

        <!-- Versions Tab -->
        <div id="versions-tab" style="display: none;">
            <div class="section">
                <div class="section-title">📦 版本列表</div>
                <div id="versionsList">載入中...</div>
            </div>
        </div>

        <!-- Debug Panel -->
        <div class="debug-panel" id="debugPanel">
            <div id="debugLogs"></div>
        </div>
    </div>

    <script>
        // State
        let recordings = [];
        let selectedRecordings = {};
        let speakerSelections = {};
        let currentTraining = null;
        let eventSource = null;

        // Elements
        const personaSelect = document.getElementById('personaSelect');
        const recordingsList = document.getElementById('recordingsList');
        const versionsList = document.getElementById('versionsList');
        const progressSection = document.getElementById('progressSection');
        const startBtn = document.getElementById('startTrainingBtn');

        // Logging
        function log(msg, level = 'info') {
            const logs = document.getElementById('debugLogs');
            const entry = document.createElement('div');
            entry.className = `log-entry ${level}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            logs.appendChild(entry);
            logs.scrollTop = logs.scrollHeight;
            console.log(`[${level.toUpperCase()}] ${msg}`);
        }

        // Switch tabs
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('training-tab').style.display = tab === 'training' ? 'block' : 'none';
            document.getElementById('versions-tab').style.display = tab === 'versions' ? 'block' : 'none';
            if (tab === 'versions') loadVersions();
        }

        // Load recordings
        async function loadRecordings() {
            try {
                const personaId = personaSelect.value;
                const response = await fetch('/api/recordings/');
                const data = await response.json();
                recordings = Array.isArray(data) ? data : (data.recordings || []);

                // Filter by persona
                const filtered = recordings.filter(r => r.persona_id === personaId && r.status === 'processed');

                if (filtered.length === 0) {
                    recordingsList.innerHTML = '<div class="empty-state">沒有可用於訓練的錄音</div>';
                    return;
                }

                recordingsList.innerHTML = filtered.map(r => {
                    const speakers = r.speaker_segments || [];
                    const uniqueSpeakers = [...new Set(speakers.map(s => s.speaker_id))].sort();
                    const labels = r.speaker_labels || {};
                    const isSelected = selectedRecordings[r.recording_id];
                    const recId = r.recording_id;
                    const totalDuration = uniqueSpeakers.length > 0
                        ? speakers.filter(s => labels[s.speaker_id] === personaId).reduce((acc, s) => acc + (s.end_time - s.start_time), 0)
                        : (r.duration_seconds || 0);

                    let speakersHtml = '';
                    if (uniqueSpeakers.length > 0) {
                        speakersHtml = '<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;">';
                        speakersHtml += '<div style="font-size: 0.85rem; color: #888; margin-bottom: 8px;">說話者:</div>';
                        uniqueSpeakers.forEach(sp => {
                            const label = labels[sp] || '';
                            const isLabelMatch = label === personaId;
                            const sel = speakerSelections[recId + '_' + sp] !== undefined
                                ? speakerSelections[recId + '_' + sp]
                                : isLabelMatch;
                            speakersHtml += `
                                <div class="speaker-row">
                                    <input type="checkbox" class="speaker-checkbox" id="chk_${recId}_${sp}"
                                        ${sel ? 'checked' : ''}
                                        onchange="toggleSpeaker('${recId}', '${sp}', this.checked)">
                                    <span class="speaker-name">${sp}</span>
                                    <select class="speaker-select" id="sel_${recId}_${sp}"
                                        onchange="setSpeakerLabel('${recId}', '${sp}', this.value)">
                                        <option value="">--</option>
                                        <option value="xiao_s" ${label === 'xiao_s' ? 'selected' : ''}>小S</option>
                                        <option value="caregiver" ${label === 'caregiver' ? 'selected' : ''}>照護者</option>
                                        <option value="elder_gentle" ${label === 'elder_gentle' ? 'selected' : ''}>長輩-溫柔</option>
                                        <option value="elder_playful" ${label === 'elder_playful' ? 'selected' : ''}>長輩-活潑</option>
                                    </select>
                                    <button class="speaker-audio-btn" onclick="playSpeakerAudio('${recId}', '${sp}')">▶</button>
                                </div>
                            `;
                        });
                        speakersHtml += '</div>';
                    }

                    return `
                        <div class="recording-card ${isSelected ? 'selected' : ''}" onclick="toggleRecording('${r.recording_id}')">
                            <div class="recording-card-header">
                                <div>
                                    <div class="recording-title">${r.folder_name || r.recording_id}</div>
                                    <div class="recording-meta">
                                        ${r.duration_seconds ? r.duration_seconds.toFixed(1) + 's' : '-'} |
                                        ${uniqueSpeakers.length > 0 ? uniqueSpeakers.length + ' speakers' : '1 speaker'}
                                        ${labels && Object.keys(labels).length > 0 ? ' | 已標記' : ''}
                                    </div>
                                </div>
                                <div style="font-size: 1.5rem; color: ${isSelected ? '#00ccff' : '#555'};">
                                    ${isSelected ? '✓' : ''}
                                </div>
                            </div>
                            ${speakersHtml}
                        </div>
                    `;
                }).join('');

                updatePreview();
                log(`Loaded ${filtered.length} recordings for ${personaId}`);
            } catch (e) {
                log(`Failed to load recordings: ${e.message}`, 'error');
            }
        }

        // Toggle recording selection
        function toggleRecording(recId) {
            if (selectedRecordings[recId]) {
                delete selectedRecordings[recId];
            } else {
                selectedRecordings[recId] = true;
            }
            loadRecordings();
        }

        // Toggle speaker
        function toggleSpeaker(recId, speakerId, checked) {
            const key = recId + '_' + speakerId;
            if (checked) {
                speakerSelections[key] = speakerId;
                // Auto-set label to current persona
                document.getElementById('sel_' + recId + '_' + speakerId).value = personaSelect.value;
            } else {
                delete speakerSelections[key];
            }
            updatePreview();
        }

        // Set speaker label
        function setSpeakerLabel(recId, speakerId, label) {
            const key = recId + '_' + speakerId;
            speakerSelections[key] = label || undefined;
            updatePreview();
        }

        // Play speaker audio
        async function playSpeakerAudio(recId, speakerId) {
            try {
                const response = await fetch(`/api/recordings/${recId}/speaker/${speakerId}/audio`);
                if (!response.ok) throw new Error('Not available');
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                audio.play();
            } catch (e) {
                log(`Speaker audio failed: ${e.message}`, 'error');
            }
        }

        // Update training preview
        function updatePreview() {
            const previewList = document.getElementById('previewList');
            const selectedRecs = recordings.filter(r => selectedRecordings[r.recording_id]);

            if (selectedRecs.length === 0) {
                previewList.innerHTML = '<div class="empty-state">選擇錄音以查看預覽</div>';
                startBtn.disabled = true;
                return;
            }

            let totalDuration = 0;
            const items = selectedRecs.map(r => {
                const speakers = r.speaker_segments || [];
                const uniqueSpeakers = [...new Set(speakers.map(s => s.speaker_id))];
                let duration = r.duration_seconds || 30;

                // Calculate duration based on selected speakers
                const recId = r.recording_id;
                let usedSpeakers = 0;
                uniqueSpeakers.forEach(sp => {
                    const key = recId + '_' + sp;
                    if (speakerSelections[key]) {
                        const spSegments = speakers.filter(s => s.speaker_id === sp);
                        const spDuration = spSegments.reduce((acc, s) => acc + (s.end_time - s.start_time), 0);
                        duration = Math.max(duration, spDuration);
                        usedSpeakers++;
                    }
                });

                totalDuration += usedSpeakers > 0 ? duration : (r.duration_seconds || 30);
                return { name: r.folder_name || r.recording_id, duration };
            });

            const personaId = personaSelect.value;
            const epochs = parseInt(document.getElementById('epochsSelect').value);
            const estimatedTime = Math.round(totalDuration * epochs * 0.5 * 1.3);
            const minutes = Math.floor(estimatedTime / 60);
            const seconds = estimatedTime % 60;

            previewList.innerHTML = `
                <div style="font-size: 0.9rem; margin-bottom: 10px;">訓練人格: <strong>${personaId}</strong></div>
                ${items.map(item => `
                    <div class="preview-item">
                        <span>${item.name}</span>
                        <span>${item.duration.toFixed(1)}s</span>
                    </div>
                `).join('')}
                <div class="preview-total">
                    <span>總計: ${totalDuration.toFixed(1)}s</span>
                    <span>預計: ${minutes}:${seconds.toString().padStart(2, '0')}</span>
                </div>
                ${totalDuration < 10 ? '<div style="color: #ffcc00; margin-top: 10px;">⚠️ 音頻總時長不足 10 秒</div>' : ''}
            `;

            startBtn.disabled = totalDuration < 10;
        }

        // Start training
        async function startTraining() {
            const personaId = personaSelect.value;
            const epochs = parseInt(document.getElementById('epochsSelect').value);
            const rank = parseInt(document.getElementById('rankSelect').value);
            const batchSize = parseInt(document.getElementById('batchSizeSelect').value);

            // Build speaker selections
            const speakerSel = {};
            Object.entries(speakerSelections).forEach(([key, value]) => {
                if (value) {
                    const [recId, speakerId] = key.split('_');
                    if (!speakerSel[recId]) speakerSel[recId] = {};
                    speakerSel[recId] = speakerId;
                }
            });

            const recordingIds = Object.keys(selectedRecordings);
            if (recordingIds.length === 0) {
                alert('請選擇至少一個錄音');
                return;
            }

            startBtn.disabled = true;
            startBtn.textContent = '訓練中...';
            progressSection.classList.add('visible');

            log(`Starting training: persona=${personaId}, recordings=${recordingIds.length}`);

            try {
                const response = await fetch('/api/training/versions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        persona_id: personaId,
                        recording_ids: recordingIds,
                        rank,
                        num_epochs: epochs,
                        batch_size: batchSize,
                        speaker_selections: speakerSel,
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Training failed');
                }

                const result = await response.json();
                currentTraining = result.version_id;
                log(`Training started: ${currentTraining}`);

                // Connect to SSE for progress
                connectProgress(result.version_id);

            } catch (e) {
                log(`Training failed: ${e.message}`, 'error');
                alert('訓練失敗: ' + e.message);
                startBtn.disabled = false;
                startBtn.textContent = '開始訓練';
                progressSection.classList.remove('visible');
            }
        }

        // Connect to progress SSE
        function connectProgress(versionId) {
            if (eventSource) {
                eventSource.close();
            }

            eventSource = new EventSource(`/api/training/versions/${versionId}/progress`);

            eventSource.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.event === 'progress') {
                    updateProgress(data);
                } else if (data.event === 'complete') {
                    log('Training complete!');
                    progressSection.classList.remove('visible');
                    startBtn.disabled = false;
                    startBtn.textContent = '開始訓練';
                    eventSource.close();
                    // Refresh versions
                    loadVersions();
                } else if (data.event === 'error') {
                    log(`Training error: ${data.error}`, 'error');
                    progressSection.classList.remove('visible');
                    startBtn.disabled = false;
                    startBtn.textContent = '開始訓練';
                    eventSource.close();
                }
            };

            eventSource.onerror = () => {
                log('SSE connection lost, will retry...', 'warning');
            };
        }

        // Update progress UI
        function updateProgress(data) {
            const pct = data.progress_pct || 0;
            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressFill').textContent = pct + '%';
            document.getElementById('statEpoch').textContent = `${data.current_epoch || 0}/${data.total_epochs || 0}`;
            document.getElementById('statLoss').textContent = data.current_loss ? data.current_loss.toFixed(4) : '-';

            const eta = data.eta_seconds || 0;
            const mins = Math.floor(eta / 60);
            const secs = eta % 60;
            document.getElementById('statETA').textContent = mins > 0 ? `${mins}:${secs.toString().padStart(2, '0')}` : '-';
        }

        // Load versions
        async function loadVersions() {
            try {
                const response = await fetch('/api/training/versions');
                const data = await response.json();
                const versions = data.versions || [];

                if (versions.length === 0) {
                    versionsList.innerHTML = '<div class="empty-state">尚無訓練版本</div>';
                    return;
                }

                versionsList.innerHTML = versions.map(v => {
                    const isActive = v.completed_at && !v.status;
                    const lossStr = v.final_loss ? v.final_loss.toFixed(4) : '-';
                    const timeStr = v.training_time_seconds
                        ? Math.floor(v.training_time_seconds / 60) + 'm'
                        : (v.status === 'training' ? '訓練中' : '-');

                    return `
                        <div class="version-card ${v.status === 'ready' && !isActive ? '' : ''}">
                            <div class="version-info">
                                <div class="version-name">
                                    ${v.version_id}
                                    <span style="color: #888; font-size: 0.85rem;">
                                        (${v.persona_id})
                                    </span>
                                </div>
                                <div class="version-meta">
                                    ${v.status === 'training' ? '🔄 訓練中' : v.status === 'ready' ? '✓ 就緒' : '✕ 失敗'} |
                                    Loss: ${lossStr} |
                                    時間: ${timeStr} |
                                    ${v.num_recordings_used || v.recording_ids_used?.length || 0} 個錄音
                                </div>
                            </div>
                            <div class="version-actions">
                                ${v.status === 'ready' ? `<button class="btn-activate" onclick="activateVersion('${v.version_id}')">啟用</button>` : ''}
                                ${v.status !== 'training' ? `<button class="btn-delete" onclick="deleteVersion('${v.version_id}')">刪除</button>` : ''}
                            </div>
                        </div>
                    `;
                }).join('');

                log(`Loaded ${versions.length} versions`);
            } catch (e) {
                log(`Failed to load versions: ${e.message}`, 'error');
            }
        }

        // Activate version
        async function activateVersion(versionId) {
            try {
                const response = await fetch(`/api/training/versions/${versionId}/activate`, { method: 'POST' });
                if (!response.ok) throw new Error('Activation failed');
                log(`Activated: ${versionId}`);
                loadVersions();
            } catch (e) {
                log(`Activation failed: ${e.message}`, 'error');
            }
        }

        // Delete version
        async function deleteVersion(versionId) {
            if (!confirm('確定要刪除這個版本嗎？')) return;
            try {
                const response = await fetch(`/api/training/versions/${versionId}`, { method: 'DELETE' });
                if (!response.ok) throw new Error('Delete failed');
                log(`Deleted: ${versionId}`);
                loadVersions();
            } catch (e) {
                log(`Delete failed: ${e.message}`, 'error');
            }
        }

        // Event listeners
        personaSelect.addEventListener('change', () => {
            selectedRecordings = {};
            speakerSelections = {};
            loadRecordings();
        });

        ['epochsSelect', 'rankSelect', 'batchSizeSelect'].forEach(id => {
            document.getElementById(id).addEventListener('change', updatePreview);
        });

        // Load data on page load
        loadRecordings();
        log('Training page loaded');
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)
