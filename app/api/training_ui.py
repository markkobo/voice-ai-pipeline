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
            flex-wrap: wrap;
            gap: 10px;
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
            display: inline-flex;
            align-items: center;
            gap: 6px;
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
            min-width: 140px;
        }

        /* ==================== TREE VIEW STYLES ==================== */
        .filter-bar {
            display: flex;
            gap: 12px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .filter-bar select {
            background: #1a1a2e;
            color: #fff;
            border: 1px solid #333;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 14px;
            min-width: 140px;
        }
        .recordings-count {
            font-size: 0.85rem;
            color: #888;
            margin-bottom: 12px;
        }

        /* Tree recording folder */
        .tree-recording {
            cursor: pointer;
            padding: 12px;
            background: #0d1b2a;
            border-radius: 8px;
            margin-bottom: 8px;
            border: 2px solid transparent;
            transition: background 0.15s, border-color 0.15s;
        }
        .tree-recording:hover { background: #16213e; }
        .tree-recording.has-selected { border-color: #00ccff44; }
        .tree-recording .folder-icon {
            display: inline-block;
            width: 20px;
            margin-right: 8px;
        }
        .tree-recording.expanded .folder-icon::before { content: '📂'; }
        .tree-recording:not(.expanded) .folder-icon::before { content: '📁'; }

        /* Recording header row */
        .recording-header-row {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .recording-title-text {
            font-weight: 500;
            color: #fff;
        }
        .recording-badges {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }
        .recording-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            background: #333;
            color: #aaa;
        }
        .recording-badge.listener { background: #00ccff22; color: #00ccff; }
        .recording-badge.persona { background: #00ff8822; color: #00ff88; }
        .recording-duration {
            font-size: 0.85rem;
            color: #888;
            margin-left: auto;
        }
        .tree-segments {
            margin-left: 24px;
            margin-top: 8px;
            display: none;
        }
        .tree-recording.expanded .tree-segments { display: block; }

        /* Segment row */
        .tree-segment {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: #16213e;
            border-radius: 6px;
            margin-bottom: 4px;
            flex-wrap: wrap;
        }
        .segment-checkbox {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        .segment-info {
            flex: 1;
            min-width: 200px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .segment-speaker {
            font-family: monospace;
            font-size: 0.85rem;
            color: #00ccff;
        }
        .segment-meta {
            font-size: 0.75rem;
            color: #888;
        }
        .segment-transcript {
            font-size: 0.8rem;
            color: #666;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 300px;
        }
        .segment-duration {
            font-size: 0.85rem;
            color: #888;
            font-family: monospace;
            min-width: 50px;
            text-align: center;
        }
        .quality-badge {
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 4px;
            min-width: 60px;
            text-align: center;
        }
        .quality-good { background: #00ff8833; color: #00ff88; }
        .quality-bad { background: #ff444433; color: #ff4444; }

        /* Segment audio controls */
        .segment-audio {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .audio-btn {
            background: #333;
            color: #fff;
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .audio-btn:hover { background: #444; }
        .audio-btn.play { background: #00ccff; color: #000; }
        .audio-btn.play:hover { background: #00aadd; }
        .audio-btn.pause { background: #ffcc00; color: #000; }
        .segment-progress {
            flex: 1;
            min-width: 100px;
            height: 8px;
            background: #333;
            border-radius: 4px;
            cursor: pointer;
            overflow: hidden;
            position: relative;
        }
        .segment-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ccff, #00ff88);
            width: 0%;
            transition: width 0.1s;
            border-radius: 4px;
        }
        .progress-time {
            font-size: 0.7rem;
            color: #888;
            font-family: monospace;
            min-width: 70px;
        }

        /* Training settings */
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
        @media (max-width: 600px) {
            .settings-grid { grid-template-columns: 1fr; }
        }

        /* Training preview */
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
            font-size: 0.9rem;
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

        /* Progress section */
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

        /* Version card */
        .version-card {
            background: #0d1b2a;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 15px;
            flex-wrap: wrap;
        }
        .version-card.active {
            border: 2px solid #00ff88;
        }
        .version-card.new-version {
            border: 2px solid #00ccff;
            animation: newVersionPulse 2s ease-in-out 3;
        }
        @keyframes newVersionPulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(0, 204, 255, 0.4); }
            50% { box-shadow: 0 0 20px 5px rgba(0, 204, 255, 0.4); }
        }
        .version-info { flex: 1; min-width: 250px; }
        .version-name-row {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 8px;
        }
        .version-name {
            font-weight: 500;
            color: #fff;
        }
        .nickname-display {
            cursor: pointer;
            color: #00ccff;
            font-size: 0.9rem;
        }
        .nickname-display:hover { text-decoration: underline; }
        .nickname-edit {
            background: #1a1a2e;
            color: #fff;
            border: 1px solid #00ccff;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: inherit;
        }
        .version-badges {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }
        .version-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            background: #333;
            color: #aaa;
        }
        .version-badge.active-tag { background: #00ff88; color: #000; }
        .version-badge.new-tag { background: #00ccff; color: #000; }
        .version-meta {
            font-size: 0.85rem;
            color: #888;
            margin-top: 5px;
        }
        .version-stats {
            font-size: 0.85rem;
            color: #aaa;
            margin-top: 5px;
        }
        .version-actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn-activate {
            background: #00ff88;
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            min-height: 36px;
        }
        .btn-activate:hover { background: #00dd77; }
        .btn-preview {
            background: #00ccff;
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            min-height: 36px;
        }
        .btn-preview:hover { background: #00aadd; }
        .btn-delete {
            background: #ff4444;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            min-height: 36px;
        }
        .btn-delete:hover { background: #dd3333; }
        .delete-confirm {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
        }
        .delete-confirm a { color: #ff4444; }
        .delete-confirm a:hover { text-decoration: underline; }
        .delete-confirm .cancel-link { color: #888; }

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

        /* Toast notifications */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .toast {
            background: #16213e;
            color: #fff;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.9rem;
            animation: toastIn 0.3s ease;
            max-width: 320px;
        }
        .toast.success { border-left: 4px solid #00ff88; }
        .toast.error { border-left: 4px solid #ff4444; }
        .toast.info { border-left: 4px solid #00ccff; }
        .toast.fadeOut { animation: toastOut 0.3s ease forwards; }
        @keyframes toastIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes toastOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }

        /* Mobile responsive */
        @media (max-width: 768px) {
            .tree-segment { flex-direction: column; align-items: flex-start; }
            .segment-audio { width: 100%; }
            .segment-progress { width: 100%; }
            .version-card { flex-direction: column; }
            .version-actions { width: 100%; }
            .form-group select, .form-group input { min-width: 100%; }
        }
    </style>
</head>
<body>
    <!-- Toast container -->
    <div class="toast-container" id="toastContainer"></div>

    <div class="container">
        <div class="header">
            <a href="/ui" class="back-btn">跳轉到對話</a>
            <h1>訓練管理</h1>
            <a href="/ui/recordings" class="back-btn">錄音管理</a>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('training')">訓練</button>
            <button class="tab" onclick="switchTab('versions')">版本</button>
        </div>

        <!-- Training Tab -->
        <div id="training-tab">
            <!-- Persona Selection -->
            <div class="section">
                <div class="section-title">選擇人格</div>
                <div class="form-row">
                    <div class="form-group">
                        <label>人格 (Persona)</label>
                        <select id="personaSelect"></select>
                    </div>
                </div>
            </div>

            <!-- Recordings Selection with Tree View -->
            <div class="section">
                <div class="section-title">選擇片段</div>
                <div class="filter-bar">
                    <select id="listenerFilter">
                        <option value="">全部聆聽者</option>
                    </select>
                    <span id="recordingsCount" class="recordings-count"></span>
                </div>
                <div id="treeView">
                    <div class="empty-state">載入中...</div>
                </div>
            </div>

            <!-- Training Settings -->
            <div class="section">
                <div class="section-title">訓練設定</div>
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
                    <h3 style="font-size: 1rem; margin-bottom: 10px; color: #00ccff;">訓練預覽</h3>
                    <div id="previewList">
                        <div class="empty-state">選擇片段以查看預覽</div>
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
                <div class="section-title">版本列表</div>
                <div style="margin-bottom: 12px;">
                    <label style="font-size: 0.85rem; color: #888; display: block; margin-bottom: 6px;">測試文字 (預覽時朗讀)</label>
                    <input type="text" id="previewText" value="你好，這是我的聲音測試。"
                        style="width: 100%; padding: 10px 14px; background: #1a1a2e; color: #fff; border: 1px solid #333; border-radius: 8px; font-size: 14px;">
                </div>
                <div id="versionsList">載入中...</div>
            </div>
        </div>

        <!-- Debug Panel -->
        <div class="debug-panel" id="debugPanel">
            <div id="debugLogs"></div>
        </div>
    </div>

    <script>
        // ==================== STATE ====================
        let allRecordings = [];
        let personas = [];
        let listeners = [];
        let selectedSegments = new Set();  // Set of "{recording_id}_{speaker_id}"
        let expandedRecordings = new Set();
        let activeAudio = null;
        let currentPlayingSegment = null;
        let currentTraining = null;
        let eventSource = null;

        // ==================== ELEMENTS ====================
        const personaSelect = document.getElementById('personaSelect');
        const listenerFilter = document.getElementById('listenerFilter');
        const treeView = document.getElementById('treeView');
        const versionsList = document.getElementById('versionsList');
        const progressSection = document.getElementById('progressSection');
        const startBtn = document.getElementById('startTrainingBtn');

        // ==================== LOGGING ====================
        function log(msg, level = 'info') {
            const logs = document.getElementById('debugLogs');
            const entry = document.createElement('div');
            entry.className = `log-entry ${level}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            logs.appendChild(entry);
            logs.scrollTop = logs.scrollHeight;
            console.log(`[${level.toUpperCase()}] ${msg}`);
        }

        // ==================== TOAST ====================
        function showToast(message, type = 'info', duration = 4000) {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : type === 'warning' ? '⚠' : 'ℹ';
            toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
            container.appendChild(toast);
            setTimeout(() => {
                toast.classList.add('fadeOut');
                setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 300);
            }, duration);
        }

        // ==================== INITIALIZATION ====================
        async function init() {
            await Promise.all([loadPersonas(), loadListeners(), loadRecordings()]);
            buildPersonaDropdown();
            buildListenerFilter();
            log('Training page loaded');
        }

        async function loadPersonas() {
            try {
                const res = await fetch('/api/personas/');
                const data = await res.json();
                personas = data.personas || [];
                log(`Loaded ${personas.length} personas`);
            } catch (e) {
                log(`Failed to load personas: ${e.message}`, 'error');
                personas = [];
            }
        }

        async function loadListeners() {
            try {
                const res = await fetch('/api/listeners/');
                const data = await res.json();
                listeners = data.listeners || [];
                log(`Loaded ${listeners.length} listeners`);
            } catch (e) {
                log(`Failed to load listeners: ${e.message}`, 'error');
                listeners = [];
            }
        }

        function buildPersonaDropdown() {
            const current = personaSelect.value;
            personaSelect.innerHTML = '';
            personas.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.persona_id;
                opt.textContent = p.name || p.persona_id;
                personaSelect.appendChild(opt);
            });
            if (current && personaSelect.querySelector(`option[value="${current}"]`)) {
                personaSelect.value = current;
            }
        }

        function buildListenerFilter() {
            const current = listenerFilter.value;
            listenerFilter.innerHTML = '<option value="">全部聆聽者</option>';
            listeners.forEach(l => {
                const opt = document.createElement('option');
                opt.value = l.listener_id;
                opt.textContent = l.name || l.listener_id;
                listenerFilter.appendChild(opt);
            });
            listenerFilter.value = current;
        }

        function getPersonaName(personaId) {
            if (!personaId) return '未設定';
            const p = personas.find(x => x.persona_id === personaId);
            return p ? p.name : personaId;
        }

        function getListenerName(listenerId) {
            if (!listenerId) return '未設定';
            const l = listeners.find(x => x.listener_id === listenerId);
            return l ? l.name : listenerId;
        }

        // ==================== RECORDINGS LOADING ====================
        async function loadRecordings() {
            try {
                const response = await fetch('/api/recordings/');
                const data = await response.json();
                const recordings = Array.isArray(data) ? data : (data.recordings || []);
                allRecordings = recordings;
                renderTree();
                log(`Loaded ${recordings.length} recordings`);
            } catch (e) {
                log(`Failed to load recordings: ${e.message}`, 'error');
                treeView.innerHTML = '<div class="empty-state">載入失敗</div>';
            }
        }

        // ==================== TREE VIEW RENDERING ====================
        function renderTree() {
            const listenerId = listenerFilter.value;

            let filtered = allRecordings;
            if (listenerId) {
                filtered = filtered.filter(r => r.listener_id === listenerId);
            }

            // Only show processed recordings
            filtered = filtered.filter(r => r.status === 'processed');

            if (filtered.length === 0) {
                treeView.innerHTML = '<div class="empty-state">沒有處理的錄音</div>';
                document.getElementById('recordingsCount').textContent = '';
                return;
            }

            document.getElementById('recordingsCount').textContent =
                `共 ${filtered.length} 個錄音${listenerId ? ' (聆聽者: ' + getListenerName(listenerId) + ')' : ''}`;

            treeView.innerHTML = filtered.map(r => renderRecordingFolder(r)).join('');

            // Restore expanded state
            expandedRecordings.forEach(id => {
                const el = document.getElementById(`rec-${id}`);
                if (el) el.classList.add('expanded');
            });
        }

        function renderRecordingFolder(r) {
            const speakers = r.speaker_segments || [];
            const hasSelected = speakers.some(s => selectedSegments.has(`${r.recording_id}_${s.speaker_id}`));
            const listenerName = getListenerName(r.listener_id);
            const personaName = getPersonaName(r.persona_id);
            const durationText = r.duration_seconds ? r.duration_seconds.toFixed(1) + 's' : '-';

            return `
                <div class="tree-recording ${hasSelected ? 'has-selected' : ''}" id="rec-${r.recording_id}">
                    <div class="recording-header-row" onclick="toggleRecording('${r.recording_id}')">
                        <span class="folder-icon"></span>
                        <span class="recording-title-text">${r.folder_name || r.recording_id}</span>
                        <div class="recording-badges">
                            <span class="recording-badge listener">[${listenerName}]</span>
                            <span class="recording-badge persona">[${personaName}]</span>
                        </div>
                        <span class="recording-duration">${durationText}</span>
                    </div>
                    <div class="tree-segments" id="segments-${r.recording_id}">
                        ${speakers.length === 0 ? '<div style="padding:8px;color:#666;font-size:0.85rem;">無片段</div>' : ''}
                        ${speakers.map(s => renderSegmentRow(r, s)).join('')}
                    </div>
                </div>
            `;
        }

        function renderSegmentRow(r, segment) {
            const segId = `${r.recording_id}_${segment.speaker_id}`;
            const isSelected = selectedSegments.has(segId);
            const duration = segment.duration_seconds || 0;
            const transcript = segment.transcription?.text || '';
            const quality = segment.quality_score;
            const personaId = segment.persona_id || '';
            const listenerId = segment.listener_id || r.listener_id || '';
            const qualityClass = quality !== null && quality !== undefined ? (quality >= 0.6 ? 'quality-good' : 'quality-bad') : '';
            const qualityIcon = quality !== null && quality !== undefined ? (quality >= 0.6 ? '✓' : '⚠') : '';
            const truncated = transcript.length > 50 ? transcript.substring(0, 50) + '...' : transcript;

            return `
                <div class="tree-segment" id="seg-${segId.replace(/[^a-zA-Z0-9]/g, '_')}">
                    <input type="checkbox" class="segment-checkbox"
                        id="chk-${segId.replace(/[^a-zA-Z0-9]/g, '_')}"
                        ${isSelected ? 'checked' : ''}
                        onchange="toggleSegment('${segId}', this.checked)"
                        onclick="event.stopPropagation()">
                    <div class="segment-info">
                        <span class="segment-speaker">${segment.speaker_id}</span>
                        <div class="segment-meta">
                            人格: ${getPersonaName(personaId)} | 對: ${getListenerName(listenerId)}
                        </div>
                        ${transcript ? `<div class="segment-transcript">${truncated}</div>` : ''}
                    </div>
                    <span class="segment-duration">${duration.toFixed(1)}s</span>
                    ${quality !== null && quality !== undefined ? `<span class="quality-badge ${qualityClass}">${qualityIcon} 品質</span>` : ''}
                    <div class="segment-audio" onclick="event.stopPropagation()">
                        <button class="audio-btn play" id="play-${segId.replace(/[^a-zA-Z0-9]/g, '_')}"
                            onclick="playSegment('${r.recording_id}', '${segment.speaker_id}')">▶</button>
                        <button class="audio-btn pause" id="pause-${segId.replace(/[^a-zA-Z0-9]/g, '_')}"
                            onclick="pauseSegment()" style="display:none">⏸</button>
                        <button class="audio-btn" onclick="stopSegment()" style="background:#666">⏹</button>
                        <div class="segment-progress" id="progress-${segId.replace(/[^a-zA-Z0-9]/g, '_')}">
                            <div class="segment-progress-fill" id="progress-fill-${segId.replace(/[^a-zA-Z0-9]/g, '_')}"></div>
                        </div>
                        <span class="progress-time" id="time-${segId.replace(/[^a-zA-Z0-9]/g, '_')}">0:00 / ${formatTime(duration)}</span>
                    </div>
                </div>
            `;
        }

        function formatTime(seconds) {
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return `${m}:${s.toString().padStart(2, '0')}`;
        }

        // ==================== TOGGLE EXPAND/COLLAPSE ====================
        function toggleRecording(recordingId) {
            const el = document.getElementById(`rec-${recordingId}`);
            if (el.classList.contains('expanded')) {
                el.classList.remove('expanded');
                expandedRecordings.delete(recordingId);
            } else {
                el.classList.add('expanded');
                expandedRecordings.add(recordingId);
            }
        }

        // ==================== SEGMENT SELECTION ====================
        function toggleSegment(segId, checked) {
            if (checked) {
                selectedSegments.add(segId);
            } else {
                selectedSegments.delete(segId);
            }
            // Update recording folder style
            const recId = segId.split('_')[0];
            const el = document.getElementById(`rec-${recId}`);
            if (el) {
                const hasSelected = Array.from(selectedSegments).some(s => s.startsWith(recId + '_'));
                el.classList.toggle('has-selected', hasSelected);
            }
            updatePreview();
        }

        // ==================== SEGMENT PLAYBACK ====================
        function playSegment(recordingId, speakerId) {
            stopSegment();

            const segId = `${recordingId}_${speakerId}`;
            const safeId = segId.replace(/[^a-zA-Z0-9]/g, '_');
            const playBtn = document.getElementById(`play-${safeId}`);
            const pauseBtn = document.getElementById(`pause-${safeId}`);

            log(`Playing segment: ${speakerId}`);

            fetch(`/api/recordings/${recordingId}/speaker/${speakerId}/audio`)
                .then(res => {
                    if (!res.ok) throw new Error('Audio not available');
                    return res.blob();
                })
                .then(blob => {
                    const url = URL.createObjectURL(blob);
                    activeAudio = new Audio(url);
                    currentPlayingSegment = segId;

                    activeAudio.addEventListener('timeupdate', () => updateProgressUI(segId));
                    activeAudio.addEventListener('ended', () => resetPlaybackUI(segId));
                    activeAudio.addEventListener('error', (e) => {
                        log(`Audio error: ${e.message}`, 'error');
                        resetPlaybackUI(segId);
                    });

                    playBtn.style.display = 'none';
                    pauseBtn.style.display = 'flex';
                    activeAudio.play();
                })
                .catch(e => {
                    log(`Failed to play segment: ${e.message}`, 'error');
                    showToast('播放失敗: ' + e.message, 'error');
                });
        }

        function pauseSegment() {
            if (!activeAudio || !currentPlayingSegment) return;
            const safeId = currentPlayingSegment.replace(/[^a-zA-Z0-9]/g, '_');
            const playBtn = document.getElementById(`play-${safeId}`);
            const pauseBtn = document.getElementById(`pause-${safeId}`);
            activeAudio.pause();
            if (playBtn) playBtn.style.display = 'flex';
            if (pauseBtn) pauseBtn.style.display = 'none';
        }

        function stopSegment() {
            if (activeAudio) {
                activeAudio.pause();
                activeAudio.currentTime = 0;
                if (currentPlayingSegment) {
                    resetPlaybackUI(currentPlayingSegment);
                }
                activeAudio = null;
                currentPlayingSegment = null;
            }
        }

        function resetPlaybackUI(segId) {
            const safeId = segId.replace(/[^a-zA-Z0-9]/g, '_');
            const playBtn = document.getElementById(`play-${safeId}`);
            const pauseBtn = document.getElementById(`pause-${safeId}`);
            const progressFill = document.getElementById(`progress-fill-${safeId}`);
            if (playBtn) playBtn.style.display = 'flex';
            if (pauseBtn) pauseBtn.style.display = 'none';
            if (progressFill) progressFill.style.width = '0%';
        }

        function updateProgressUI(segId) {
            if (!activeAudio) return;
            const safeId = segId.replace(/[^a-zA-Z0-9]/g, '_');
            const progressFill = document.getElementById(`progress-fill-${safeId}`);
            const timeDisplay = document.getElementById(`time-${safeId}`);
            if (!progressFill || !timeDisplay) return;

            const percent = (activeAudio.currentTime / activeAudio.duration) * 100;
            progressFill.style.width = `${percent}%`;
            const current = formatTime(activeAudio.currentTime);
            const total = formatTime(activeAudio.duration || 0);
            timeDisplay.textContent = `${current} / ${total}`;
        }

        // ==================== TRAINING PREVIEW ====================
        function updatePreview() {
            const previewList = document.getElementById('previewList');
            const personaId = personaSelect.value;

            if (selectedSegments.size === 0) {
                previewList.innerHTML = '<div class="empty-state">選擇片段以查看預覽</div>';
                startBtn.disabled = true;
                return;
            }

            let totalDuration = 0;
            const items = [];

            selectedSegments.forEach(segId => {
                const [recId, speakerId] = segId.split('_', 1);
                const rec = allRecordings.find(r => r.recording_id === recId);
                if (!rec) return;
                const segment = rec.speaker_segments?.find(s => s.speaker_id === speakerId);
                if (!segment) return;

                const duration = segment.duration_seconds || 0;
                totalDuration += duration;
                items.push({
                    recName: rec.folder_name || recId,
                    speakerId,
                    duration,
                    segId
                });
            });

            const personaName = getPersonaName(personaId);
            const epochs = parseInt(document.getElementById('epochsSelect').value);
            const estimatedTime = Math.round(totalDuration * epochs * 0.5 * 1.3);
            const minutes = Math.floor(estimatedTime / 60);
            const seconds = Math.round(estimatedTime % 60);

            previewList.innerHTML = `
                <div style="font-size: 0.9rem; margin-bottom: 10px;">將訓練「${personaName}」:</div>
                ${items.map(item => `
                    <div class="preview-item">
                        <span>• ${item.recName} - ${item.speakerId} (${item.duration.toFixed(1)}s)</span>
                    </div>
                `).join('')}
                <div style="border-top: 1px solid #333; margin-top: 8px; padding-top: 8px;">
                    <div class="preview-item">
                        <span>總計:</span>
                        <span>~${totalDuration.toFixed(1)}s, ${items.length} 個片段</span>
                    </div>
                    <div class="preview-item">
                        <span>預計時間:</span>
                        <span>約 ${minutes}m ${seconds}s</span>
                    </div>
                </div>
                ${totalDuration < 10 ? '<div style="color: #ffcc00; margin-top: 10px;">⚠️ 建議至少 10s</div>' : ''}
            `;

            startBtn.disabled = totalDuration < 10;
        }

        // ==================== START TRAINING ====================
        async function startTraining() {
            const personaId = personaSelect.value;
            const epochs = parseInt(document.getElementById('epochsSelect').value);
            const rank = parseInt(document.getElementById('rankSelect').value);
            const batchSize = parseInt(document.getElementById('batchSizeSelect').value);

            const segmentIds = Array.from(selectedSegments);
            if (segmentIds.length === 0) {
                showToast('請選擇至少一個片段', 'error');
                return;
            }

            // Calculate total duration
            let totalDuration = 0;
            segmentIds.forEach(segId => {
                const [recId, speakerId] = segId.split('_', 1);
                const rec = allRecordings.find(r => r.recording_id === recId);
                const segment = rec?.speaker_segments?.find(s => s.speaker_id === speakerId);
                if (segment) totalDuration += segment.duration_seconds || 0;
            });

            if (totalDuration < 10) {
                showToast('音頻總時長不足 10 秒', 'error');
                return;
            }

            startBtn.disabled = true;
            startBtn.textContent = '訓練中...';
            progressSection.classList.add('visible');

            log(`Starting training: persona=${personaId}, segments=${segmentIds.length}`);

            try {
                const response = await fetch('/api/training/versions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        persona_id: personaId,
                        segment_ids: segmentIds,
                        rank,
                        num_epochs: epochs,
                        batch_size: batchSize,
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Training failed');
                }

                const result = await response.json();
                currentTraining = result.version_id;
                log(`Training started: ${currentTraining}`);

                connectProgress(result.version_id);

            } catch (e) {
                log(`Training failed: ${e.message}`, 'error');
                showToast('訓練失敗: ' + e.message, 'error');
                startBtn.disabled = false;
                startBtn.textContent = '開始訓練';
                progressSection.classList.remove('visible');
            }
        }

        // ==================== PROGRESS SSE ====================
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
                    showNotification('訓練完成！', `版本 ${data.version_id || versionId} 訓練成功`);
                    switchTab('versions');
                    loadVersions(data.version_id || versionId);
                } else if (data.event === 'error') {
                    log(`Training error: ${data.error}`, 'error');
                    progressSection.classList.remove('visible');
                    startBtn.disabled = false;
                    startBtn.textContent = '開始訓練';
                    eventSource.close();
                    showNotification('訓練失敗', data.error || '未知錯誤');
                }
            };

            eventSource.onerror = () => {
                log('SSE connection lost, will retry...', 'warning');
            };
        }

        function updateProgress(data) {
            const pct = data.progress_pct || 0;
            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressFill').textContent = pct + '%';
            document.getElementById('statEpoch').textContent = `${data.current_epoch || 0}/${data.total_epochs || 0}`;
            document.getElementById('statLoss').textContent = data.current_loss ? data.current_loss.toFixed(4) : '-';

            const eta = data.eta_seconds || 0;
            const mins = Math.floor(eta / 60);
            const secs = Math.round(eta % 60);
            document.getElementById('statETA').textContent = mins > 0 ? `${mins}:${secs.toString().padStart(2, '0')}` : '-';
        }

        // ==================== NOTIFICATIONS ====================
        function requestNotificationPermission() {
            if ('Notification' in window && Notification.permission === 'default') {
                Notification.requestPermission();
            }
        }

        function showNotification(title, body) {
            if ('Notification' in window && Notification.permission === 'granted') {
                new Notification(title, { body, icon: '🎓' });
            }
        }

        requestNotificationPermission();

        // ==================== VERSIONS TAB ====================
        async function loadVersions(newVersionId = null) {
            try {
                const [versionsRes, activeRes, recordingsRes] = await Promise.all([
                    fetch('/api/training/versions'),
                    fetch('/api/training/active?persona_id=' + personaSelect.value),
                    fetch('/api/recordings/')
                ]);

                const versionsData = await versionsRes.json();
                const activeData = await activeRes.json();
                const recordingsData = await recordingsRes.json();
                const versions = versionsData.versions || [];
                const activeVersionId = activeData.version?.version_id;
                const recordingsMap = {};
                (recordingsData.recordings || []).forEach(r => recordingsMap[r.recording_id] = r);

                if (versions.length === 0) {
                    versionsList.innerHTML = '<div class="empty-state">尚無訓練版本</div>';
                    return;
                }

                // Sort: active first, then by created_at desc
                versions.sort((a, b) => {
                    if (a.version_id === activeVersionId) return -1;
                    if (b.version_id === activeVersionId) return 1;
                    return new Date(b.created_at) - new Date(a.created_at);
                });

                versionsList.innerHTML = versions.map(v => {
                    const isActive = v.version_id === activeVersionId;
                    const isNew = v.version_id === newVersionId;
                    const lossStr = v.final_loss ? v.final_loss.toFixed(4) : '-';
                    const completedDate = v.completed_at
                        ? new Date(v.completed_at).toLocaleString('zh-TW', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })
                        : (v.created_at ? new Date(v.created_at).toLocaleString('zh-TW', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '-');

                    // Get segment info from manifest
                    const manifest = v.manifest || {};
                    const segmentIds = manifest.segment_ids || v.segment_ids_used || [];
                    const segmentCount = segmentIds.length;
                    const recordingCount = v.recording_ids_used?.length || manifest.recordings?.length || 0;

                    const displayName = v.nickname ? `${v.version_id}: ${v.nickname}` : v.version_id;

                    return `
                        <div class="version-card ${isActive ? 'active' : ''} ${isNew ? 'new-version' : ''}" id="version-${v.version_id}">
                            <div class="version-info">
                                <div class="version-name-row">
                                    <span class="nickname-display" onclick="editNickname('${v.version_id}', '${v.nickname || ''}')">${displayName}</span>
                                    <div class="version-badges">
                                        ${isActive ? '<span class="version-badge active-tag">當前啟用</span>' : ''}
                                        ${isNew ? '<span class="version-badge new-tag">剛完成</span>' : ''}
                                        <span class="version-badge">(${v.persona_id})</span>
                                    </div>
                                </div>
                                <div class="version-meta">
                                    ${v.status === 'training' ? '🔄 訓練中' : v.status === 'ready' ? '✓ 就緒' : '✕ 失敗'} |
                                    完成: ${completedDate}
                                </div>
                                <div class="version-stats">
                                    Loss: ${lossStr} | Epochs: ${v.num_epochs} | Rank: ${v.rank}
                                </div>
                                <div style="font-size: 0.8rem; margin-top: 5px; color: #666;">
                                    片段: ${recordingCount} 個錄音, ${segmentCount} 個片段
                                </div>
                            </div>
                            <div class="version-actions">
                                ${v.status === 'ready' && !isActive ? `<button class="btn-activate" onclick="activateVersion('${v.version_id}')">啟用</button>` : ''}
                                ${v.status === 'ready' ? `<button class="btn-preview" onclick="previewVersion('${v.version_id}')">▶ 預覽</button>` : ''}
                                ${v.status !== 'training' ? `
                                    <button class="btn-delete" id="delbtn-${v.version_id}" onclick="confirmDelete('${v.version_id}')">✕</button>
                                    <span id="delcfm-${v.version_id}" class="delete-confirm" style="display: none;">
                                        確定？<a href="#" onclick="doDelete('${v.version_id}'); return false;">刪除</a>
                                        <a href="#" class="cancel-link" onclick="cancelDelete('${v.version_id}'); return false;">取消</a>
                                    </span>
                                ` : ''}
                            </div>
                        </div>
                    `;
                }).join('');

                // Scroll to and highlight new version
                if (newVersionId) {
                    const el = document.getElementById('version-' + newVersionId);
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        setTimeout(() => { el.style.boxShadow = ''; }, 3000);
                    }
                }

                log(`Loaded ${versions.length} versions`);
            } catch (e) {
                log(`Failed to load versions: ${e.message}`, 'error');
            }
        }

        // ==================== NICKNAME EDITING ====================
        function editNickname(versionId, currentNickname) {
            const displayEl = document.querySelector(`#version-${versionId} .nickname-display`);
            if (!displayEl) return;

            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'nickname-edit';
            input.value = currentNickname || '';
            input.placeholder = '輸入暱稱...';
            input.maxLength = 50;

            const save = async () => {
                const newNickname = input.value.trim();
                try {
                    const res = await fetch(`/api/training/versions/${versionId}`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ nickname: newNickname || null })
                    });
                    if (!res.ok) throw new Error('Update failed');
                    showToast('暱稱已更新', 'success');
                    loadVersions();
                } catch (e) {
                    log(`Failed to update nickname: ${e.message}`, 'error');
                    showToast('更新失敗: ' + e.message, 'error');
                }
            };

            input.addEventListener('blur', save);
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    input.blur();
                } else if (e.key === 'Escape') {
                    loadVersions();
                }
            });

            displayEl.replaceWith(input);
            input.focus();
            input.select();
        }

        // ==================== VERSION ACTIONS ====================
        async function activateVersion(versionId) {
            try {
                const response = await fetch(`/api/training/versions/${versionId}/activate`, { method: 'POST' });
                if (!response.ok) throw new Error('Activation failed');
                log(`Activated: ${versionId}`);
                showToast(`版本 ${versionId} 已啟用`, 'success');
                loadVersions();
            } catch (e) {
                log(`Activation failed: ${e.message}`, 'error');
                showToast('啟用失敗: ' + e.message, 'error');
            }
        }

        function confirmDelete(versionId) {
            document.getElementById('delbtn-' + versionId).style.display = 'none';
            document.getElementById('delcfm-' + versionId).style.display = 'inline';
        }

        function cancelDelete(versionId) {
            document.getElementById('delbtn-' + versionId).style.display = '';
            document.getElementById('delcfm-' + versionId).style.display = 'none';
        }

        async function doDelete(versionId) {
            try {
                const response = await fetch(`/api/training/versions/${versionId}`, { method: 'DELETE' });
                if (!response.ok) throw new Error('Delete failed');
                log(`Deleted: ${versionId}`);
                showToast(`版本已刪除`, 'success');
                cancelDelete(versionId);
                loadVersions();
            } catch (e) {
                log(`Delete failed: ${e.message}`, 'error');
                showToast('刪除失敗: ' + e.message, 'error');
                cancelDelete(versionId);
            }
        }

        async function previewVersion(versionId) {
            const previewText = document.getElementById('previewText').value.trim() || '你好，這是我的聲音測試。';
            log(`Generating preview for ${versionId}: "${previewText}"...`);
            try {
                const response = await fetch(`/api/training/versions/${versionId}/preview`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: previewText })
                });
                if (!response.ok) throw new Error('Preview failed');
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                audio.play();
                log(`Preview playing for ${versionId}`);
            } catch (e) {
                log(`Preview failed: ${e.message}`, 'error');
                showToast('預覽失敗: ' + e.message, 'error');
            }
        }

        // ==================== TAB SWITCHING ====================
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('training-tab').style.display = tab === 'training' ? 'block' : 'none';
            document.getElementById('versions-tab').style.display = tab === 'versions' ? 'block' : 'none';
            if (tab === 'versions') loadVersions();
        }

        // ==================== EVENT LISTENERS ====================
        personaSelect.addEventListener('change', () => {
            selectedSegments.clear();
            renderTree();
            updatePreview();
        });

        listenerFilter.addEventListener('change', () => {
            renderTree();
        });

        ['epochsSelect', 'rankSelect', 'batchSizeSelect'].forEach(id => {
            document.getElementById(id).addEventListener('change', updatePreview);
        });

        // ==================== INITIAL LOAD ====================
        init();
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)
