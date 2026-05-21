// System status poller + gateIfTraining moved to _status_bar.js
        // (RFC_M6 Phase 0-pre review #28). Register a SYS_ON_UPDATE hook
        // for training-page-specific gating (disable startTrainingBtn).
        window.SYS_ON_UPDATE = window.SYS_ON_UPDATE || [];
        window.SYS_ON_UPDATE.push(function () {
            const startBtn = document.getElementById('startTrainingBtn');
            if (!startBtn) return;
            if (window.SYS.trainingActive) {
                startBtn.disabled = true;
                startBtn.classList.add('gated');
            } else {
                startBtn.disabled = false;
                startBtn.classList.remove('gated');
            }
        });
        // ----------------------------------------

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
            await resumeInFlightTraining();
        }

        // If a training is already running when the page loads (user
        // refreshed mid-training, or opened from another tab), reattach
        // the progress SSE so the widget shows live state instead of
        // leaving "Epoch 0/0, Loss —, ETA —" as if nothing was happening.
        // Without this, the only way to see progress was to be on the page
        // continuously from before the start button was pressed.
        async function resumeInFlightTraining() {
            try {
                const res = await fetch('/api/training/versions');
                if (!res.ok) return;
                const data = await res.json();
                const active = (data.versions || []).find(v => v.status === 'training');
                if (!active) return;
                log(`Resuming progress for in-flight training: ${active.version_id}`);
                currentTraining = active.version_id;
                progressSection.classList.add('visible');
                startBtn.disabled = true;
                startBtn.textContent = '訓練中...';
                connectProgress(active.version_id);
            } catch (e) {
                log(`Failed to resume in-flight training: ${e.message}`, 'warning');
            }
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
            const personaId = personaSelect && personaSelect.value;  // top-level "選擇人格"

            let filtered = allRecordings;
            // Persona filter (user-reported 2026-05-21): the top-level
            // 人格 selector picks WHICH persona to train; the recordings
            // list must show only recordings tagged with that persona so
            // the user can't accidentally mix personas in a training run.
            if (personaId) {
                filtered = filtered.filter(r => r.persona_id === personaId);
            }
            if (listenerId) {
                filtered = filtered.filter(r => r.listener_id === listenerId);
            }

            // Only show processed recordings
            filtered = filtered.filter(r => r.status === 'processed');

            if (filtered.length === 0) {
                const conds = [];
                if (personaId) conds.push('人格: ' + getPersonaName(personaId));
                if (listenerId) conds.push('聆聽者: ' + getListenerName(listenerId));
                const suffix = conds.length ? `（${conds.join('、')}）` : '';
                treeView.innerHTML = `<div class="empty-state">沒有符合條件的錄音${suffix}</div>`;
                document.getElementById('recordingsCount').textContent = '';
                return;
            }

            const labelBits = [];
            if (personaId) labelBits.push('人格: ' + getPersonaName(personaId));
            if (listenerId) labelBits.push('聆聽者: ' + getListenerName(listenerId));
            document.getElementById('recordingsCount').textContent =
                `共 ${filtered.length} 個錄音${labelBits.length ? ' (' + labelBits.join('、') + ')' : ''}`;

            treeView.innerHTML = filtered.map(r => renderRecordingFolder(r)).join('');

            // Restore expanded state
            expandedRecordings.forEach(id => {
                const el = document.getElementById(`rec-${id}`);
                if (el) el.classList.add('expanded');
            });
        }

        // Group the flat per-snippet speaker_segments[] into one row per
        // unique speaker_id. Selection is already keyed by speaker_id, so the
        // checkbox behavior is unchanged — this only collapses the visual
        // list (1 podcast can have 50+ alternating snippets).
        function groupSegmentsBySpeaker(segments) {
            const groups = new Map();
            for (const s of segments) {
                let g = groups.get(s.speaker_id);
                if (!g) {
                    g = {
                        speaker_id: s.speaker_id,
                        segments: [],
                        total_duration: 0,
                        quality_scores: [],
                        transcript_preview: '',
                        quality_flags: { has_overlap: false, low_energy: false, high_noise: false, too_short: false },
                        persona_id: s.persona_id,
                        listener_id: s.listener_id,
                        longest_segment: null,
                    };
                    groups.set(s.speaker_id, g);
                }
                g.segments.push(s);
                const d = s.duration_seconds || 0;
                g.total_duration += d;
                if (s.quality_score !== null && s.quality_score !== undefined) {
                    g.quality_scores.push(s.quality_score);
                }
                if (!g.transcript_preview && s.transcription?.text) {
                    g.transcript_preview = s.transcription.text;
                }
                const qf = s.quality_flags || {};
                if (qf.has_overlap) g.quality_flags.has_overlap = true;
                if (qf.low_energy) g.quality_flags.low_energy = true;
                if (qf.high_noise) g.quality_flags.high_noise = true;
                if (qf.too_short) g.quality_flags.too_short = true;
                if (!g.longest_segment || d > (g.longest_segment.duration_seconds || 0)) {
                    g.longest_segment = s;
                }
            }
            const out = Array.from(groups.values());
            out.forEach(g => {
                g.avg_quality = g.quality_scores.length
                    ? g.quality_scores.reduce((a, b) => a + b, 0) / g.quality_scores.length
                    : null;
            });
            return out.sort((a, b) => a.speaker_id.localeCompare(b.speaker_id));
        }

        function renderRecordingFolder(r) {
            const segments = r.speaker_segments || [];
            const speakerGroups = groupSegmentsBySpeaker(segments);
            const hasSelected = speakerGroups.some(g => selectedSegments.has(`${r.recording_id}_${g.speaker_id}`));
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
                        ${speakerGroups.length === 0 ? '<div style="padding:8px;color:#666;font-size:0.85rem;">無片段</div>' : ''}
                        ${speakerGroups.map(g => renderSpeakerGroupRow(r, g)).join('')}
                    </div>
                </div>
            `;
        }

        function renderSpeakerGroupRow(r, group) {
            const selectionKey = `${r.recording_id}_${group.speaker_id}`;
            const segId = `${r.recording_id}_${group.speaker_id}`;
            const safeSegId = segId.replace(/[^a-zA-Z0-9]/g, '_');
            const isSelected = selectedSegments.has(selectionKey);
            const duration = group.total_duration || 0;
            const segCount = group.segments.length;
            const transcript = group.transcript_preview || '';
            const quality = group.avg_quality;
            const qualityFlags = group.quality_flags || {};
            const personaId = group.persona_id || '';
            const listenerId = group.listener_id || r.listener_id || '';
            // Play the longest snippet — its quality is most representative.
            const playSeg = group.longest_segment || group.segments[0] || {};
            const playStart = playSeg.start_time;
            const playEnd = playSeg.end_time;

            // Quality badge with new classes
            const qualityBadgeClass = quality !== null && quality !== undefined
                ? (quality >= 0.8 ? 'quality-excellent' : quality >= 0.6 ? 'quality-good' : quality >= 0.4 ? 'quality-fair' : 'quality-poor')
                : '';
            const qualityLabel = quality !== null && quality !== undefined
                ? (quality >= 0.8 ? '優秀' : quality >= 0.6 ? '良好' : quality >= 0.4 ? '一般' : '惡劣')
                : '';
            const qualityBadge = quality !== null && quality !== undefined
                ? `<span class="quality-badge ${qualityBadgeClass}">${qualityLabel} ${(quality * 100).toFixed(0)}%</span>`
                : '';

            // Quality flags display
            const flagsHtml = [];
            if (qualityFlags.has_overlap) flagsHtml.push('<span class="quality-flag active">重疊</span>');
            if (qualityFlags.low_energy) flagsHtml.push('<span class="quality-flag active">低能量</span>');
            if (qualityFlags.high_noise) flagsHtml.push('<span class="quality-flag active">高噪音</span>');
            if (qualityFlags.too_short) flagsHtml.push('<span class="quality-flag active">太短</span>');
            const flagsDisplay = flagsHtml.length > 0 ? `<div class="quality-flags">${flagsHtml.join('')}</div>` : '';

            const truncated = transcript.length > 50 ? transcript.substring(0, 50) + '...' : transcript;
            const countBadge = segCount > 1
                ? `<span class="quality-badge" style="background:#37475a;color:#bcd;">${segCount} 段</span>`
                : '';

            return `
                <div class="tree-segment" id="seg-${safeSegId}">
                    <input type="checkbox" class="segment-checkbox"
                        id="chk-${safeSegId}"
                        ${isSelected ? 'checked' : ''}
                        onchange="toggleSegment('${selectionKey}', this.checked)"
                        onclick="event.stopPropagation()">
                    <div class="segment-info">
                        <span class="segment-speaker">${group.speaker_id}</span>
                        <div class="segment-meta">
                            人格: ${getPersonaName(personaId)} | 對: ${getListenerName(listenerId)}
                        </div>
                        ${transcript ? `<div class="segment-transcript">${truncated}</div>` : ''}
                        ${flagsDisplay}
                    </div>
                    <span class="segment-duration">${duration.toFixed(1)}s</span>
                    ${countBadge}
                    ${qualityBadge}
                    <div class="segment-audio" onclick="event.stopPropagation()">
                        <button class="audio-btn play" id="play-${safeSegId}"
                            onclick="playSegment('${r.recording_id}', '${group.speaker_id}', ${playStart}, ${playEnd}, '${safeSegId}')">▶</button>
                        <button class="audio-btn pause" id="pause-${safeSegId}"
                            onclick="pauseSegment()" style="display:none">⏸</button>
                        <button class="audio-btn" onclick="stopSegment()" style="background:#666">⏹</button>
                        <div class="segment-progress" id="progress-${safeSegId}">
                            <div class="segment-progress-fill" id="progress-fill-${safeSegId}"></div>
                        </div>
                        <span class="progress-time" id="time-${safeSegId}">0:00 / ${formatTime(playSeg.duration_seconds || 0)}</span>
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
            // segId format is {recording_id}_{speaker_id} where speaker_id is
            // SPEAKER_NN — split on '_SPEAKER_' to recover recording_id.
            const speakerIndex = segId.indexOf('_SPEAKER_');
            const recId = speakerIndex !== -1 ? segId.substring(0, speakerIndex) : segId;
            const el = document.getElementById(`rec-${recId}`);
            if (el) {
                const hasSelected = Array.from(selectedSegments).some(s => s.startsWith(recId + '_'));
                el.classList.toggle('has-selected', hasSelected);
            }
            updatePreview();
        }

        // ==================== SEGMENT PLAYBACK ====================
        function playSegment(recordingId, speakerId, startTime, endTime, safeId) {
            stopSegment();

            const playBtn = document.getElementById(`play-${safeId}`);
            const pauseBtn = document.getElementById(`pause-${safeId}`);

            log(`Playing segment: ${speakerId} [${startTime?.toFixed(2)}s - ${endTime?.toFixed(2)}s]`);

            let url = `/api/recordings/${recordingId}/speaker/${speakerId}/audio`;
            const params = [];
            if (startTime !== null && startTime !== undefined) params.push(`start=${startTime}`);
            if (endTime !== null && endTime !== undefined) params.push(`end=${endTime}`);
            if (params.length > 0) url += '?' + params.join('&');

            fetch(url)
                .then(res => {
                    if (!res.ok) throw new Error('Audio not available');
                    return res.blob();
                })
                .then(blob => {
                    const audioUrl = URL.createObjectURL(blob);
                    activeAudio = new Audio(audioUrl);
                    currentPlayingSegment = safeId;

                    activeAudio.addEventListener('timeupdate', () => updateProgressUI(safeId));
                    activeAudio.addEventListener('ended', () => resetPlaybackUI(safeId));
                    activeAudio.addEventListener('error', (e) => {
                        log(`Audio error: ${e.message}`, 'error');
                        resetPlaybackUI(safeId);
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
                // segId format: {recording_id}_{speaker_id}
                // speaker_id is SPEAKER_XX, so find the _SPEAKER_ pattern from the end
                const speakerIndex = segId.indexOf('_SPEAKER_');
                if (speakerIndex === -1) {
                    log(`Invalid segId format (no _SPEAKER_): ${segId}`, 'warning', 'TRAINING');
                    return;
                }
                const recId = segId.substring(0, speakerIndex);
                const actualSpeakerId = segId.substring(speakerIndex + 1);  // Include the leading underscore
                const rec = allRecordings.find(r => r.recording_id === recId);
                if (!rec) {
                    log(`Recording not found: ${recId}`, 'warning', 'TRAINING');
                    return;
                }
                // Sum all segments for this speaker (recordings often have many
                // alternating segments per speaker — collapse to one preview row
                // matching the grouped tree view).
                const matching = (rec.speaker_segments || []).filter(s => s.speaker_id === actualSpeakerId);
                if (matching.length === 0) {
                    log(`No segments for ${actualSpeakerId} in ${recId}`, 'warning', 'TRAINING');
                    return;
                }
                const duration = matching.reduce((acc, s) => acc + (s.duration_seconds || 0), 0);
                totalDuration += duration;
                items.push({
                    recName: rec.folder_name || recId,
                    speakerId: actualSpeakerId,
                    duration,
                    segCount: matching.length,
                    segId
                });
                log(`Preview: ${recId}/${actualSpeakerId} = ${duration.toFixed(1)}s (${matching.length} 段)`, 'info', 'TRAINING');
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
                        <span>• ${item.recName} - ${item.speakerId} (${item.duration.toFixed(1)}s${item.segCount > 1 ? `, ${item.segCount} 段` : ''})</span>
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

        // Hide/show LoRA-rank field based on selected training type.
        function onTrainingTypeChange() {
            const ttype = document.getElementById('trainingTypeSelect').value;
            const rankGroup = document.getElementById('rankGroup');
            if (rankGroup) rankGroup.style.display = (ttype === 'lora') ? '' : 'none';
        }

        // ==================== START TRAINING ====================
        async function startTraining() {
            if (gateIfTraining('開始訓練')) return;
            const personaId = personaSelect.value;
            const trainingType = document.getElementById('trainingTypeSelect').value;
            const epochs = parseInt(document.getElementById('epochsSelect').value);
            const rank = parseInt(document.getElementById('rankSelect').value);
            const batchSize = parseInt(document.getElementById('batchSizeSelect').value);

            const segmentIds = Array.from(selectedSegments);
            if (segmentIds.length === 0) {
                showToast('請選擇至少一個片段', 'error');
                return;
            }

            // Calculate total duration — sum ALL segments for each selected
            // (recording, speaker) pair, since one speaker often has many
            // alternating snippets in a podcast.
            let totalDuration = 0;
            segmentIds.forEach(segId => {
                // segId format: {recording_id}_{speaker_id}
                const speakerIndex = segId.indexOf('_SPEAKER_');
                if (speakerIndex === -1) return;
                const recId = segId.substring(0, speakerIndex);
                const actualSpeakerId = segId.substring(speakerIndex + 1);
                const rec = allRecordings.find(r => r.recording_id === recId);
                const matching = (rec?.speaker_segments || []).filter(s => s.speaker_id === actualSpeakerId);
                totalDuration += matching.reduce((acc, s) => acc + (s.duration_seconds || 0), 0);
            });

            if (totalDuration < 10) {
                showToast('音頻總時長不足 10 秒', 'error');
                return;
            }

            startBtn.disabled = true;
            startBtn.textContent = '訓練中...';
            progressSection.classList.add('visible');

            log(`Starting training: persona=${personaId}, segments=${segmentIds.length}`);
            console.log('DEBUG segmentIds:', JSON.stringify(segmentIds));

            try {
                const payload = {
                    persona_id: personaId,
                    segment_ids: segmentIds,
                    rank,
                    num_epochs: epochs,
                    batch_size: batchSize,
                    training_type: trainingType,
                };
                console.log('DEBUG payload:', JSON.stringify(payload));

                const response = await fetch('/api/training/versions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const err = await response.json();
                    console.log('DEBUG error response:', JSON.stringify(err));
                    throw new Error(err.detail || `Training failed (${response.status})`);
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
        async function loadVersions(newVersionId = null, personaOverride = null) {
            const queryPersona = personaOverride || personaSelect.value;
            try {
                const [versionsRes, activeRes, recordingsRes] = await Promise.all([
                    fetch('/api/training/versions'),
                    fetch('/api/training/active?persona_id=' + queryPersona),
                    fetch('/api/recordings/')
                ]);

                if (!versionsRes.ok) {
                    const text = await versionsRes.text();
                    throw new Error(`Failed to load versions (${versionsRes.status}): ${text.substring(0, 100)}`);
                }
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
                    // Prefer the explicit training_type written by the
                    // backend (TrainingVersion.training_type). model_type
                    // is the post-merge runtime descriptor ('custom_voice'
                    // for SFT-converted models) which doesn't distinguish
                    // LoRA from SFT once LoRA is also baked into a
                    // custom_voice config (per the 2026-05-21 merge fix).
                    const tType = v.training_type
                        || v.model_type
                        || (v.rank ? 'lora' : 'sft');
                    const modelTypeBadge =
                        tType === 'lora'
                            ? '<span class="version-badge" style="background: #ff9800; color: #000;">LoRA</span>'
                        : (tType === 'sft' || tType === 'custom_voice')
                            ? '<span class="version-badge" style="background: #9c27b0;">SFT</span>'
                        : tType === 'voicedesign'
                            ? '<span class="version-badge" style="background: #2196f3;">VoiceDesign</span>'
                        : '';
                    const lrStr = v.learning_rate ? (v.learning_rate < 0.001 ? `${v.learning_rate * 1000}μ` : v.learning_rate) : '-';
                    const batchInfo = v.batch_size ? `BS:${v.batch_size}` : '';
                    const trainingSummary = v.training_summary
                        ? `<div class="version-summary">${v.training_summary}</div>`
                        : '';

                    return `
                        <div class="version-card ${isActive ? 'active' : ''} ${isNew ? 'new-version' : ''}" id="version-${v.version_id}">
                            <div class="version-info">
                                <div class="version-name-row">
                                    <span class="nickname-display" onclick="editNickname('${v.version_id}', '${v.nickname || ''}')">${displayName}</span>
                                    <div class="version-badges">
                                        ${isActive ? '<span class="version-badge active-tag">當前啟用</span>' : ''}
                                        ${isNew ? '<span class="version-badge new-tag">剛完成</span>' : ''}
                                        <span class="version-badge">(${v.persona_id})</span>
                                        ${modelTypeBadge}
                                    </div>
                                </div>
                                <div class="version-meta">
                                    ${v.status === 'training' ? '🔄 訓練中' : v.status === 'merging' ? '⚙️ 合併中' : v.status === 'ready' ? '✓ 就緒' : '✕ 失敗'} |
                                    完成: ${completedDate}
                                </div>
                                ${v.status === 'failed' && v.error_message ? `
                                    <div class="version-error" title="${String(v.error_message).replace(/"/g, '&quot;')}">
                                        失敗原因: ${String(v.error_message).slice(0, 140)}${String(v.error_message).length > 140 ? '…' : ''}
                                    </div>
                                ` : ''}
                                <div class="version-stats">
                                    <span title="Loss 數值單獨看沒意義 — 按右側 ▶ 預覽 聽實際音質才是真正的判斷依據">Loss: ${lossStr}</span> | Epochs: ${v.num_epochs} | LR: ${lrStr} ${batchInfo}
                                </div>
                                <div style="font-size: 0.8rem; margin-top: 5px; color: #666;">
                                    片段: ${recordingCount} 個錄音, ${segmentCount} 個片段
                                </div>
                                ${trainingSummary}
                                <div class="version-details" id="details-${v.version_id}" style="display: none;">
                                    ${renderVersionDetails(v, manifest)}
                                </div>
                            </div>
                            <div class="version-actions">
                                ${v.status === 'ready' && !isActive ? `<button class="btn-activate" onclick="activateVersion('${v.version_id}')">啟用</button>` : ''}
                                ${v.status === 'ready' ? `<button class="btn-preview" title="試聽此版本的音色 (Loss 不能反映音質，請以此為準)" onclick="previewVersion('${v.version_id}')">▶ 預覽</button>` : ''}
                                <button class="btn-details" onclick="toggleDetails('${v.version_id}')" title="顯示完整 metadata (路徑、IDs、時間戳)">詳細</button>
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

        // ==================== VERSION DETAILS ====================
        function fmtDate(iso) {
            if (!iso) return '—';
            try { return new Date(iso).toLocaleString('zh-TW'); }
            catch (e) { return iso; }
        }
        function fmtDuration(seconds) {
            if (seconds == null) return '—';
            const s = Math.round(seconds);
            if (s < 60) return `${s}s`;
            const m = Math.floor(s / 60);
            const rem = s % 60;
            if (m < 60) return `${m}m ${rem}s`;
            const h = Math.floor(m / 60);
            return `${h}h ${m % 60}m`;
        }
        function escapeHtml(str) {
            if (str == null) return '';
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        }
        function renderVersionDetails(v, manifest) {
            const rows = [];
            const add = (label, value, mono = false) => {
                if (value === undefined || value === null || value === '' || (Array.isArray(value) && value.length === 0)) return;
                const display = Array.isArray(value) ? value.join(', ') : value;
                rows.push(`<div class="detail-row"><span class="detail-label">${label}</span><span class="detail-value${mono ? ' mono' : ''}">${escapeHtml(display)}</span></div>`);
            };
            add('Version ID', v.version_id, true);
            add('Persona', v.persona_id);
            add('Status', v.status);
            add('Training type', v.training_type);
            add('Model type (runtime)', v.model_type);
            add('Base model', v.base_model, true);
            add('Created', fmtDate(v.created_at));
            add('Completed', fmtDate(v.completed_at));
            add('Training time', fmtDuration(v.training_time_seconds));
            add('Final loss', v.final_loss != null ? v.final_loss.toFixed(6) : null);
            if (v.training_type === 'lora') add('LoRA rank', v.rank);
            add('Learning rate', v.learning_rate);
            add('Epochs', v.num_epochs);
            add('Batch size', v.batch_size);
            add('LoRA path', v.lora_path, true);
            add('Merged path', v.merged_path, true);
            const recIds = v.recording_ids_used || (manifest.recordings || []).map(r => r.recording_id);
            add(`Recording IDs (${recIds.length})`, recIds, true);
            const segIds = v.segment_ids_used || manifest.segment_ids || [];
            add(`Segment IDs (${segIds.length})`, segIds, true);
            if (manifest.total_duration_seconds != null) {
                add('Total audio', fmtDuration(manifest.total_duration_seconds));
            }
            if (v.status === 'failed' && v.error_message) {
                rows.push(`<div class="detail-row"><span class="detail-label">Error</span><span class="detail-value mono detail-error">${escapeHtml(v.error_message)}</span></div>`);
            }
            return rows.join('');
        }
        function toggleDetails(versionId) {
            const el = document.getElementById(`details-${versionId}`);
            if (!el) return;
            el.style.display = el.style.display === 'none' ? 'block' : 'none';
        }

        // ==================== VERSION ACTIONS ====================
        async function activateVersion(versionId) {
            if (gateIfTraining('啟用模型')) return;
            try {
                const response = await fetch(`/api/training/versions/${versionId}/activate`, { method: 'POST' });
                if (!response.ok) throw new Error('Activation failed');
                log(`Activated: ${versionId}`);
                showToast(`版本 ${versionId} 已啟用`, 'success');
                // Find the activated version's persona to query correct active version
                const versRes = await fetch('/api/training/versions');
                const versData = await versRes.json();
                const vers = versData.versions || [];
                const activatedVers = vers.find(v => v.version_id === versionId);
                const personaId = activatedVers?.persona_id || personaSelect.value;
                // Re-fetch with correct persona
                loadVersions(null, personaId);
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
            if (gateIfTraining('預覽語音')) return;
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
            if (tab === 'versions') loadVersions(null, personaSelect.value);
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
