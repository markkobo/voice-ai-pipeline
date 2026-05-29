// System status poller + gateIfTraining moved to _status_bar.js
        // (RFC_M6 Phase 0-pre review #28). Register a SYS_ON_UPDATE hook
        // for training-page-specific gating (disable startTrainingBtn).
        window.SYS_ON_UPDATE = window.SYS_ON_UPDATE || [];
        window.SYS_ON_UPDATE.push(function (status) {
            const startBtn = document.getElementById('startTrainingBtn');
            if (startBtn) {
                if (window.SYS.trainingActive) {
                    startBtn.disabled = true;
                    startBtn.classList.add('gated');
                } else {
                    startBtn.disabled = false;
                    startBtn.classList.remove('gated');
                }
            }
            // If the active TTS version changed elsewhere (e.g. user
            // activated a version from the chat page, or the chat's
            // eager activation on persona change), refresh the versions
            // tab so the "當前啟用" badge moves to the right row instead
            // of waiting until the user navigates away and back. User
            // observation 2026-05-21: "training page doesn't update
            // like the status bar does".
            const newActive = status && status.tts && status.tts.active_version || null;
            if (newActive !== window._lastSeenActiveVersion) {
                window._lastSeenActiveVersion = newActive;
                // Only refresh if we're on the versions tab and have
                // already loaded once — avoids racing with initial load.
                const versionsTab = document.getElementById('versions-tab');
                const onVersionsTab = versionsTab && versionsTab.style.display !== 'none';
                if (typeof loadVersions === 'function' && onVersionsTab) {
                    loadVersions();
                }
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

        // ==================== TRAINING UI STATE ====================
        // Single source of truth for "are we currently training?". Toggles
        // start button + cancel button visibility together so they can't
        // get out of sync. Callers used to set startBtn.disabled and
        // textContent inline in three different places — risky.
        function setTrainingUI(isTraining) {
            const cancelBtn = document.getElementById('cancelTrainingBtn');
            if (isTraining) {
                startBtn.disabled = true;
                startBtn.textContent = 'Training...';
                if (cancelBtn) cancelBtn.style.display = '';
            } else {
                startBtn.disabled = false;
                startBtn.textContent = 'Start Training';
                if (cancelBtn) cancelBtn.style.display = 'none';
                currentTraining = null;
            }
        }

        async function cancelTraining() {
            if (!currentTraining) {
                log('No active training to cancel', 'warning');
                return;
            }
            if (!confirm('Cancel the in-progress training? Epoch progress already written to disk will be lost.')) return;
            const vid = currentTraining;
            log(`Cancelling training: ${vid}`);
            try {
                const r = await fetch(`/api/training/versions/${vid}/cancel`, { method: 'POST' });
                if (!r.ok) {
                    const err = await r.json().catch(() => ({}));
                    throw new Error(err.detail || err.message || `HTTP ${r.status}`);
                }
                log(`Training cancelled: ${vid}`);
                showToast('Training cancelled', 'success');
                stopPolling();
                if (eventSource) { try { eventSource.close(); } catch (_) {} eventSource = null; }
                progressSection.classList.remove('visible');
                setTrainingUI(false);
                loadVersions();
            } catch (e) {
                log(`Cancel failed: ${e.message}`, 'error');
                showToast('Cancel failed: ' + e.message, 'error');
            }
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
                setTrainingUI(true);
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
            // Default option: show all recordings, no persona pre-selected.
            // User reported 2026-05-26: previously the first persona was
            // auto-selected and the recording tree filtered to it, but
            // users expected to see ALL recordings first and then narrow
            // down. Without this option there was no way to view a
            // cross-persona recordings list. Caller must validate persona
            // is non-empty before kicking off training (handled in
            // startTraining()).
            const allOpt = document.createElement('option');
            allOpt.value = '';
            allOpt.textContent = '— All Personas (select the persona to train) —';
            personaSelect.appendChild(allOpt);
            personas.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.persona_id;
                opt.textContent = p.name || p.persona_id;
                personaSelect.appendChild(opt);
            });
            if (current && personaSelect.querySelector(`option[value="${current}"]`)) {
                personaSelect.value = current;
            } else {
                personaSelect.value = '';  // default to all
            }
        }

        function buildListenerFilter() {
            const current = listenerFilter.value;
            listenerFilter.innerHTML = '<option value="">All Listeners</option>';
            listeners.forEach(l => {
                const opt = document.createElement('option');
                opt.value = l.listener_id;
                opt.textContent = l.name || l.listener_id;
                listenerFilter.appendChild(opt);
            });
            listenerFilter.value = current;
        }

        function getPersonaName(personaId) {
            if (!personaId) return 'Unset';
            const p = personas.find(x => x.persona_id === personaId);
            return p ? p.name : personaId;
        }

        function getListenerName(listenerId) {
            if (!listenerId) return 'Unset';
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
                treeView.innerHTML = '<div class="empty-state">Failed to load</div>';
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
                if (personaId) conds.push('Persona: ' + getPersonaName(personaId));
                if (listenerId) conds.push('Listener: ' + getListenerName(listenerId));
                const suffix = conds.length ? ` (${conds.join(', ')})` : '';
                treeView.innerHTML = `<div class="empty-state">No matching recordings${suffix}</div>`;
                document.getElementById('recordingsCount').textContent = '';
                return;
            }

            const labelBits = [];
            if (personaId) labelBits.push('Persona: ' + getPersonaName(personaId));
            if (listenerId) labelBits.push('Listener: ' + getListenerName(listenerId));
            document.getElementById('recordingsCount').textContent =
                `${filtered.length} recording(s)${labelBits.length ? ' (' + labelBits.join(', ') + ')' : ''}`;

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
                        // voice_audit is per-speaker (audit_voice_training_quality
                        // is computed once per speaker_id by the pipeline and
                        // copied onto every segment); first non-null wins.
                        // Same shape as on the recordings page: {level, warnings, metrics}.
                        voice_audit: null,
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
                if (!g.voice_audit && s.voice_audit) {
                    g.voice_audit = s.voice_audit;
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

            // Aggregate voice-audit across all speakers — worst level wins so
            // the user sees the weakest link without expanding the folder.
            let aggLevel = null;
            const sevRank = { bad: 3, marginal: 2, good: 1 };
            for (const g of speakerGroups) {
                const lvl = g.voice_audit?.level;
                if (!lvl) continue;
                if (!aggLevel || (sevRank[lvl] || 0) > (sevRank[aggLevel] || 0)) aggLevel = lvl;
            }
            let folderAuditBadge = '';
            if (aggLevel) {
                const label = aggLevel === 'good' ? '✓ Good Audio' :
                              aggLevel === 'marginal' ? '⚠ Marginal Audio' :
                              aggLevel === 'bad' ? '✕ Not Suitable for Training' : '';
                const cls = aggLevel === 'good' ? 'va-good' :
                            aggLevel === 'marginal' ? 'va-marginal' :
                            aggLevel === 'bad' ? 'va-bad' : '';
                folderAuditBadge = `<span class="voice-audit-badge ${cls}" title="Expand to see detailed audio analysis">${label}</span>`;
            }

            return `
                <div class="tree-recording ${hasSelected ? 'has-selected' : ''}" id="rec-${r.recording_id}">
                    <div class="recording-header-row" onclick="toggleRecording('${r.recording_id}')">
                        <span class="folder-icon"></span>
                        <span class="recording-title-text">${r.folder_name || r.recording_id}</span>
                        <div class="recording-badges">
                            <span class="recording-badge listener">[${listenerName}]</span>
                            <span class="recording-badge persona">[${personaName}]</span>
                            ${folderAuditBadge}
                        </div>
                        <span class="recording-duration">${durationText}</span>
                    </div>
                    <div class="tree-segments" id="segments-${r.recording_id}">
                        ${speakerGroups.length === 0 ? '<div style="padding:8px;color:#666;font-size:0.85rem;">No segments</div>' : ''}
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
                ? (quality >= 0.8 ? 'Excellent' : quality >= 0.6 ? 'Good' : quality >= 0.4 ? 'Fair' : 'Poor')
                : '';
            const qualityBadge = quality !== null && quality !== undefined
                ? `<span class="quality-badge ${qualityBadgeClass}">${qualityLabel} ${(quality * 100).toFixed(0)}%</span>`
                : '';

            // Quality flags display
            const flagsHtml = [];
            if (qualityFlags.has_overlap) flagsHtml.push('<span class="quality-flag active">Overlap</span>');
            if (qualityFlags.low_energy) flagsHtml.push('<span class="quality-flag active">Low energy</span>');
            if (qualityFlags.high_noise) flagsHtml.push('<span class="quality-flag active">Noisy</span>');
            if (qualityFlags.too_short) flagsHtml.push('<span class="quality-flag active">Too short</span>');
            const flagsDisplay = flagsHtml.length > 0 ? `<div class="quality-flags">${flagsHtml.join('')}</div>` : '';

            // Voice-cloning audit badge — surfaces effective_bandwidth /
            // peak / spectral roll-off summary so the user can see at a
            // glance whether a segment is broadband enough for a clean
            // clone. Same shape + tooltip style as on the recordings page
            // (recordings.js renderSpeakerRow). The training UI previously
            // exposed only the segment quality_score, which doesn't catch
            // muffled-source-audio cases (low effective bw + high quality).
            const va = group.voice_audit;
            let voiceAuditBadge = '';
            if (va && va.level) {
                const level = va.level;
                const label = level === 'good' ? '✓ Good Audio' :
                              level === 'marginal' ? '⚠ Marginal Audio' :
                              level === 'bad' ? '✕ Not Suitable for Training' : '?';
                const cls = level === 'good' ? 'va-good' :
                            level === 'marginal' ? 'va-marginal' :
                            level === 'bad' ? 'va-bad' : '';
                const m = va.metrics || {};
                const warns = (va.warnings || []).join('\n');
                const tooltip = (warns ? warns + '\n\n' : '') +
                    `effective_bw: ${(m.effective_bandwidth_hz || 0).toFixed(0)} Hz\n` +
                    `peak_dbfs: ${(m.peak_dbfs || 0).toFixed(1)}\n` +
                    `rms_dbfs: ${(m.rms_dbfs || 0).toFixed(1)}\n` +
                    `silent_pct: ${(m.silent_pct || 0).toFixed(1)}%\n` +
                    `>4kHz: ${(m.energy_4_8khz_pct || 0).toFixed(1)}%, >8kHz: ${(m.energy_8_12khz_pct || 0).toFixed(2)}%`;
                voiceAuditBadge = `<span class="voice-audit-badge ${cls}" title="${escapeHtml(tooltip)}">${label}</span>`;
            }

            const truncated = transcript.length > 50 ? transcript.substring(0, 50) + '...' : transcript;
            const countBadge = segCount > 1
                ? `<span class="quality-badge" style="background:#37475a;color:#bcd;">${segCount} segments</span>`
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
                            Persona: ${getPersonaName(personaId)} | To: ${getListenerName(listenerId)}
                        </div>
                        ${transcript ? `<div class="segment-transcript">${truncated}</div>` : ''}
                        ${flagsDisplay}
                    </div>
                    <span class="segment-duration">${duration.toFixed(1)}s</span>
                    ${countBadge}
                    ${qualityBadge}
                    ${voiceAuditBadge}
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
                    showToast('Playback failed: ' + e.message, 'error');
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
                previewList.innerHTML = '<div class="empty-state">Select segments to preview</div>';
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
                log(`Preview: ${recId}/${actualSpeakerId} = ${duration.toFixed(1)}s (${matching.length} segments)`, 'info', 'TRAINING');
            });

            const personaName = getPersonaName(personaId);
            const epochs = parseInt(document.getElementById('epochsSelect').value);
            const estimatedTime = Math.round(totalDuration * epochs * 0.5 * 1.3);
            const minutes = Math.floor(estimatedTime / 60);
            const seconds = Math.round(estimatedTime % 60);

            previewList.innerHTML = `
                <div style="font-size: 0.9rem; margin-bottom: 10px;">Training "${personaName}":</div>
                ${items.map(item => `
                    <div class="preview-item">
                        <span>• ${item.recName} - ${item.speakerId} (${item.duration.toFixed(1)}s${item.segCount > 1 ? `, ${item.segCount} segments` : ''})</span>
                    </div>
                `).join('')}
                <div style="border-top: 1px solid #333; margin-top: 8px; padding-top: 8px;">
                    <div class="preview-item">
                        <span>Total:</span>
                        <span>~${totalDuration.toFixed(1)}s, ${items.length} segments</span>
                    </div>
                    <div class="preview-item">
                        <span>Estimated time:</span>
                        <span>~${minutes}m ${seconds}s</span>
                    </div>
                </div>
                ${totalDuration < 10 ? '<div style="color: #ffcc00; margin-top: 10px;">⚠️ At least 10s recommended</div>' : ''}
            `;

            startBtn.disabled = totalDuration < 10;
        }

        // Learning-rate options per training_type. Both default to 1e-5
        // (the safer default empirically — SFT at 1e-6 stalled v10, LoRA
        // at 1e-4 caused runaway training). LoRA tops out at 1e-4 (with
        // a "catastrophic" label) because that's the historical breaking
        // point; SFT tops out at 5e-5 which is the "test if 1e-5 stalls"
        // ceiling. See task description for empirical context.
        const LR_OPTIONS = {
            sft: [
                { value: '1e-6', label: '1e-6 (very conservative — may stall)' },
                { value: '5e-6', label: '5e-6 (conservative)' },
                { value: '1e-5', label: '1e-5 (default)' },
                { value: '2e-5', label: '2e-5 (aggressive)' },
                { value: '5e-5', label: '5e-5 (ceiling — test if 1e-5 stalls)' },
            ],
            lora: [
                { value: '1e-6', label: '1e-6 (very conservative)' },
                { value: '5e-6', label: '5e-6 (conservative)' },
                { value: '1e-5', label: '1e-5 (default, safer)' },
                { value: '5e-5', label: '5e-5 (risky)' },
                { value: '1e-4', label: '1e-4 (experimental — has caused runaway)' },
            ],
        };
        const DEFAULT_LR = '1e-5';

        // Preserve user's previously chosen LR across training_type
        // switches when possible (the option exists in the new list);
        // otherwise fall back to DEFAULT_LR.
        function populateLearningRateOptions(trainingType, preserveValue) {
            const sel = document.getElementById('learningRateSelect');
            if (!sel) return;
            const opts = LR_OPTIONS[trainingType] || LR_OPTIONS.sft;
            const valid = opts.some(o => o.value === preserveValue);
            const target = valid ? preserveValue : DEFAULT_LR;
            sel.innerHTML = opts.map(o => {
                const sel = o.value === target ? ' selected' : '';
                return `<option value="${o.value}"${sel}>${o.label}</option>`;
            }).join('');
        }

        // Hide/show LoRA-rank field + experimental warning based on
        // selected training type. The LoRA path is currently unstable
        // (forward_sub_talker_finetune only reaches code_predictor) so
        // we surface a warning when the user picks it. SFT is the
        // recommended default. Deferred work: forward_talker_finetune.
        function onTrainingTypeChange() {
            const ttype = document.getElementById('trainingTypeSelect').value;
            const rankGroup = document.getElementById('rankGroup');
            const warning = document.getElementById('loraWarning');
            const isLora = (ttype === 'lora');
            if (rankGroup) rankGroup.style.display = isLora ? '' : 'none';
            if (warning) warning.style.display = isLora ? '' : 'none';
            // Repopulate LR dropdown for the new training_type, keeping
            // the user's current choice if still valid.
            const lrSel = document.getElementById('learningRateSelect');
            const currentLr = lrSel ? lrSel.value : DEFAULT_LR;
            populateLearningRateOptions(ttype, currentLr);
        }

        // Initial population on page load (SFT default). Must run after
        // the DOM has the <select> — this script is loaded at the bottom
        // of training.html so the element is guaranteed to exist.
        populateLearningRateOptions(
            document.getElementById('trainingTypeSelect')?.value || 'sft',
            DEFAULT_LR
        );

        // ==================== START TRAINING ====================
        async function startTraining() {
            if (gateIfTraining('start training')) return;
            const personaId = personaSelect.value;
            if (!personaId) {
                showToast('Please select a persona to train above', 'error');
                personaSelect.focus();
                return;
            }
            const trainingType = document.getElementById('trainingTypeSelect').value;
            const epochs = parseInt(document.getElementById('epochsSelect').value);
            const rank = parseInt(document.getElementById('rankSelect').value);
            const batchSize = parseInt(document.getElementById('batchSizeSelect').value);
            // Learning rate: parseFloat handles scientific notation
            // ("1e-5" → 0.00001). Backend (CreateTrainingRequest in
            // app/api/training.py) accepts learning_rate: Optional[float].
            const lrRaw = document.getElementById('learningRateSelect').value;
            const learningRate = lrRaw ? parseFloat(lrRaw) : null;
            // Empty string ("不指定") → null → backend writes Python False
            // into spk_is_dialect[persona]. Non-empty → string like
            // "chinese" / "english" that the backend bakes verbatim.
            const languageRaw = document.getElementById('languageTokenSelect').value;
            const languageToken = languageRaw === '' ? null : languageRaw;

            const segmentIds = Array.from(selectedSegments);
            if (segmentIds.length === 0) {
                showToast('Please select at least one segment', 'error');
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
                showToast('Total audio is under 10 seconds', 'error');
                return;
            }

            setTrainingUI(true);
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
                    language_token: languageToken,
                    learning_rate: learningRate,
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
                showToast('Training failed: ' + e.message, 'error');
                setTrainingUI(false);
                progressSection.classList.remove('visible');
            }
        }

        // ==================== PROGRESS SSE + polling fallback ====================
        // cloudflared's free quick-tunnel (trycloudflare.com) aggressively
        // buffers text/event-stream responses regardless of padding /
        // X-Accel-Buffering / keepalive — confirmed 2026-05-25 with
        // local SSE working perfectly and tunnel SSE getting zero bytes.
        // Falls back to /api/training/versions/{id} polling at 2s if the
        // SSE connection doesn't deliver a `connected` event within 5s.
        let pollTimer = null;

        function stopPolling() {
            if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        }

        function pollProgress(versionId) {
            const tick = async () => {
                try {
                    const r = await fetch(`/api/training/versions/${versionId}`, { cache: 'no-store' });
                    if (!r.ok) return;
                    const v = await r.json();
                    if (v.progress) {
                        updateProgress(v.progress);
                    }
                    if (v.status === 'ready') {
                        log('Training complete (via polling)!');
                        stopPolling();
                        progressSection.classList.remove('visible');
                        setTrainingUI(false);
                        showNotification('Training Complete', `Version ${versionId} trained successfully`);
                        switchTab('versions');
                        loadVersions(versionId);
                    } else if (v.status === 'failed') {
                        log(`Training failed (via polling): ${v.error_message || ''}`, 'error');
                        stopPolling();
                        progressSection.classList.remove('visible');
                        setTrainingUI(false);
                        showNotification('Training Failed', v.error_message || 'Unknown error');
                    }
                } catch (e) {
                    log(`Progress poll error: ${e.message}`, 'warning');
                }
            };
            tick();  // fire immediately
            pollTimer = setInterval(tick, 2000);
        }

        function connectProgress(versionId) {
            if (eventSource) {
                eventSource.close();
            }
            stopPolling();

            let sseDeliveredData = false;
            eventSource = new EventSource(`/api/training/versions/${versionId}/progress`);

            // If SSE doesn't deliver any data event within 5s (cloudflared
            // is buffering at the edge), close it and fall back to HTTP
            // polling. Polling works through every proxy.
            const fallbackTimer = setTimeout(() => {
                if (!sseDeliveredData) {
                    log('SSE silent for 5s — falling back to HTTP polling (likely cloudflared buffering)', 'warning');
                    try { eventSource.close(); } catch (_) {}
                    eventSource = null;
                    pollProgress(versionId);
                }
            }, 5000);

            eventSource.onmessage = (e) => {
                sseDeliveredData = true;
                clearTimeout(fallbackTimer);
                const data = JSON.parse(e.data);
                if (data.event === 'progress') {
                    updateProgress(data);
                } else if (data.event === 'complete') {
                    log('Training complete!');
                    progressSection.classList.remove('visible');
                    setTrainingUI(false);
                    eventSource.close();
                    showNotification('Training Complete', `Version ${data.version_id || versionId} trained successfully`);
                    switchTab('versions');
                    loadVersions(data.version_id || versionId);
                } else if (data.event === 'error') {
                    log(`Training error: ${data.error}`, 'error');
                    progressSection.classList.remove('visible');
                    setTrainingUI(false);
                    eventSource.close();
                    showNotification('Training Failed', data.error || 'Unknown error');
                }
                // `connected` and other events: just keep the connection alive.
            };

            eventSource.onerror = () => {
                log('SSE connection lost', 'warning');
                if (!sseDeliveredData) {
                    // Never got any data and now errored — go to polling.
                    clearTimeout(fallbackTimer);
                    try { eventSource.close(); } catch (_) {}
                    eventSource = null;
                    pollProgress(versionId);
                }
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
                    versionsList.innerHTML = '<div class="empty-state">No training versions yet</div>';
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
                        ? new Date(v.completed_at).toLocaleString('en-US', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })
                        : (v.created_at ? new Date(v.created_at).toLocaleString('en-US', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '-');

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
                                        ${isActive ? '<span class="version-badge active-tag">Active</span>' : ''}
                                        ${isNew ? '<span class="version-badge new-tag">Just finished</span>' : ''}
                                        <span class="version-badge">(${v.persona_id})</span>
                                        ${modelTypeBadge}
                                    </div>
                                </div>
                                <div class="version-meta">
                                    ${v.status === 'training' ? '🔄 Training' : v.status === 'merging' ? '⚙️ Merging' : v.status === 'ready' ? '✓ Ready' : '✕ Failed'} |
                                    Completed: ${completedDate}
                                </div>
                                ${v.status === 'failed' && v.error_message ? `
                                    <div class="version-error" title="${String(v.error_message).replace(/"/g, '&quot;')}">
                                        Reason: ${String(v.error_message).slice(0, 140)}${String(v.error_message).length > 140 ? '…' : ''}
                                    </div>
                                ` : ''}
                                <div class="version-stats">
                                    <span title="Loss alone is not meaningful — click ▶ Preview on the right to judge by actual audio quality">Loss: ${lossStr}</span> | Epochs: ${v.num_epochs} | LR: ${lrStr} ${batchInfo}
                                </div>
                                <div style="font-size: 0.8rem; margin-top: 5px; color: #666;">
                                    ${recordingCount} recording(s), ${segmentCount} segment(s)
                                </div>
                                ${trainingSummary}
                                <div class="version-details" id="details-${v.version_id}" style="display: none;">
                                    ${renderVersionDetails(v, manifest)}
                                </div>
                            </div>
                            <div class="version-actions">
                                ${v.status === 'ready' && !isActive ? `<button class="btn-activate" onclick="activateVersion('${v.version_id}')">Activate</button>` : ''}
                                ${v.status === 'ready' ? `<button class="btn-preview" data-preview-btn="${v.version_id}" title="Preview this version's voice (judge by audio, not loss) — click again to stop" onclick="previewVersion('${v.version_id}')">▶ Preview</button>` : ''}
                                ${(v.status === 'failed' || v.status === 'cancelled') && v.progress && v.progress.latest_checkpoint_epoch != null ? `<button class="btn-resume" onclick="resumeVersion('${v.version_id}', ${v.progress.latest_checkpoint_epoch})" title="Continue training from epoch ${v.progress.latest_checkpoint_epoch}">↻ Resume (ep ${v.progress.latest_checkpoint_epoch})</button>` : ''}
                                <button class="btn-details" onclick="toggleDetails('${v.version_id}')" title="Show full metadata (paths, IDs, timestamps)">Details</button>
                                ${v.status !== 'training' ? `
                                    <button class="btn-delete" id="delbtn-${v.version_id}" onclick="confirmDelete('${v.version_id}')">✕</button>
                                    <span id="delcfm-${v.version_id}" class="delete-confirm" style="display: none;">
                                        Sure? <a href="#" onclick="doDelete('${v.version_id}'); return false;">Delete</a>
                                        <a href="#" class="cancel-link" onclick="cancelDelete('${v.version_id}'); return false;">Cancel</a>
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
            input.placeholder = 'Enter nickname...';
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
                    showToast('Nickname updated', 'success');
                    loadVersions();
                } catch (e) {
                    log(`Failed to update nickname: ${e.message}`, 'error');
                    showToast('Update failed: ' + e.message, 'error');
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
            try { return new Date(iso).toLocaleString('en-US'); }
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
            if (gateIfTraining('activate model')) return;
            try {
                const response = await fetch(`/api/training/versions/${versionId}/activate`, { method: 'POST' });
                if (!response.ok) throw new Error('Activation failed');
                log(`Activated: ${versionId}`);
                showToast(`Version ${versionId} activated`, 'success');
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
                showToast('Activation failed: ' + e.message, 'error');
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

        async function resumeVersion(versionId, lastEpoch) {
            // Resume a failed/cancelled run from its last checkpoint. The
            // API returns 409 if another training is already in flight —
            // surface that as a toast so the user knows why nothing
            // happened.
            try {
                log(`Resuming ${versionId} from epoch ${lastEpoch}...`);
                const r = await fetch(`/api/training/versions/${versionId}/resume`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: '{}'
                });
                if (!r.ok) {
                    const err = await r.json().catch(() => ({}));
                    throw new Error(err.detail || err.message || `HTTP ${r.status}`);
                }
                const data = await r.json();
                showToast(`Resumed from epoch ${data.resumed_from_epoch}`, 'success');
                log(`Resumed: ${versionId} from epoch ${data.resumed_from_epoch}`);
                loadVersions();
            } catch (e) {
                log(`Resume failed: ${e.message}`, 'error');
                showToast('Resume failed: ' + e.message, 'error');
            }
        }

        async function doDelete(versionId) {
            try {
                const response = await fetch(`/api/training/versions/${versionId}`, { method: 'DELETE' });
                if (!response.ok) throw new Error('Delete failed');
                log(`Deleted: ${versionId}`);
                showToast(`Version deleted`, 'success');
                cancelDelete(versionId);
                loadVersions();
            } catch (e) {
                log(`Delete failed: ${e.message}`, 'error');
                showToast('Delete failed: ' + e.message, 'error');
                cancelDelete(versionId);
            }
        }

        // Module-level tracking so we can stop any in-flight preview from
        // anywhere (cancel button, another preview click, page navigation).
        // Stores the version_id whose audio is currently playing.
        let currentPreviewAudio = null;
        let currentPreviewVersionId = null;

        function stopPreview() {
            if (currentPreviewAudio) {
                try {
                    currentPreviewAudio.pause();
                    currentPreviewAudio.currentTime = 0;
                    if (currentPreviewAudio.src && currentPreviewAudio.src.startsWith('blob:')) {
                        URL.revokeObjectURL(currentPreviewAudio.src);
                    }
                } catch (_) {}
                if (currentPreviewVersionId) {
                    const btn = document.querySelector(`[data-preview-btn="${currentPreviewVersionId}"]`);
                    if (btn) btn.textContent = '▶ Preview';
                }
            }
            currentPreviewAudio = null;
            currentPreviewVersionId = null;
        }

        async function previewVersion(versionId) {
            if (gateIfTraining('preview voice')) return;
            // If THIS version is the one playing, button click = stop. If a
            // DIFFERENT version is playing, stop it first then start new one.
            if (currentPreviewVersionId === versionId && currentPreviewAudio) {
                stopPreview();
                return;
            }
            stopPreview();

            const previewText = document.getElementById('previewText').value.trim() || 'Hello, this is a test of my voice.';
            log(`Generating preview for ${versionId}: "${previewText}"...`);
            const btn = document.querySelector(`[data-preview-btn="${versionId}"]`);
            if (btn) btn.textContent = '⏳ Generating…';
            try {
                const response = await fetch(`/api/training/versions/${versionId}/preview`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: previewText })
                });
                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${response.status}`);
                }
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                currentPreviewAudio = audio;
                currentPreviewVersionId = versionId;
                audio.addEventListener('ended', () => {
                    if (currentPreviewVersionId === versionId) {
                        stopPreview();
                    }
                });
                audio.addEventListener('error', () => {
                    if (currentPreviewVersionId === versionId) {
                        stopPreview();
                    }
                });
                audio.play();
                if (btn) btn.textContent = '⏹ Stop';
                log(`Preview playing for ${versionId} (click again to stop)`);
            } catch (e) {
                if (btn) btn.textContent = '▶ Preview';
                log(`Preview failed: ${e.message}`, 'error');
                showToast('Preview failed: ' + e.message, 'error');
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
