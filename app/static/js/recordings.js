// System status poller + gateIfTraining moved to _status_bar.js
        // (RFC_M6 Phase 0-pre review #28). This file no longer needs its
        // own copy; SYS / gateIfTraining are on window.
        // ----------------------------------------

        // ==================== STATE ====================
        let allRecordings = [];
        let personas = [];
        let listeners = [];
        let expandedRecordings = new Set();
        let activeAudio = null;  // Currently playing audio element
        let currentPlayingId = null;  // Current playing segment/recording ID

        // ==================== ELEMENTS ====================
        const treeView = document.getElementById('treeView');
        const searchInput = document.getElementById('searchInput');
        const listenerFilter = document.getElementById('listenerFilter');
        const recBtn = document.getElementById('recBtn');
        const duration = document.getElementById('duration');
        const dbMeterFill = document.getElementById('dbMeterFill');
        const dbLevel = document.getElementById('dbLevel');
        const qualityIndicator = document.getElementById('qualityIndicator');
        const debugLogs = document.getElementById('debugLogs');
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const listenerSelect = document.getElementById('listenerSelect');
        const personaSelect = document.getElementById('personaSelect');

        // Modal state. The modal supports two flavours: 'persona' (with an
        // is_family checkbox) and 'listener' (with a default_emotion select).
        // `modalSubmit` is the async submit handler; it returns either
        //   { ok: true }                              → modal closes
        // or
        //   { ok: false, message, field?: 'name'|'id' } → inline error shown,
        //                                                modal stays open.
        let modalSubmit = null;
        let modalType = null;            // 'persona' | 'listener' | null
        let modalIdManuallyEdited = false;  // user typed in the id box themselves

        // ==================== LOGGING ====================
        function log(message, level = 'info', component = 'UI') {
            const time = new Date().toLocaleTimeString('zh-TW', { hour12: false });
            const entry = document.createElement('div');
            entry.className = `log-entry ${level}`;
            entry.innerHTML = `<span class="log-time">${time}</span><span class="log-component">[${component}]</span> ${message}`;
            debugLogs.appendChild(entry);
            debugLogs.scrollTop = debugLogs.scrollHeight;
            console.log(`[${level.toUpperCase()}] [${component}] ${message}`);
        }

        // ==================== HELPERS ====================
        function escapeHtml(str) {
            return String(str || '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
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

        // ==================== MODAL ====================
        // Small map of common Chinese kinship / nickname terms → snake_case
        // ASCII slugs. The ID input is a suggestion only — the user can edit
        // it freely before confirming, so this list just covers the obvious
        // family-member words we expect on this page.
        const _ZH_SLUG_MAP = {
            '媽': 'mom', '媽媽': 'mama', '阿媽': 'ama', '娘': 'niang',
            '爸': 'dad', '爸爸': 'baba', '阿爸': 'aba',
            '小孩': 'child', '孩子': 'kid', '寶寶': 'baby', '寶貝': 'baobei',
            '哥': 'gege', '哥哥': 'gege', '姐': 'jiejie', '姐姐': 'jiejie',
            '弟': 'didi', '弟弟': 'didi', '妹': 'meimei', '妹妹': 'meimei',
            '爺爺': 'yeye', '奶奶': 'nainai',
            '外公': 'waigong', '外婆': 'waipo',
            '阿公': 'agong', '阿嬤': 'ama',
            '叔叔': 'shushu', '阿姨': 'ayi', '舅舅': 'jiujiu', '姑姑': 'gugu',
            '老師': 'teacher', '同學': 'classmate', '朋友': 'friend',
            '長輩': 'elder', '記者': 'reporter', '預設': 'default',
            '小S': 'xiao_s', '小s': 'xiao_s',
        };

        function slugifyName(name) {
            if (!name) return '';
            let raw = String(name).trim();
            // Exact-match Chinese kinship terms first.
            if (Object.prototype.hasOwnProperty.call(_ZH_SLUG_MAP, raw)) {
                return _ZH_SLUG_MAP[raw];
            }
            // Substring replacement for combined terms like "Lily 媽媽".
            let slug = raw;
            for (const [zh, ascii] of Object.entries(_ZH_SLUG_MAP)) {
                if (slug.includes(zh)) slug = slug.split(zh).join(' ' + ascii + ' ');
            }
            slug = slug
                .toLowerCase()
                .replace(/[\s\-\.]+/g, '_')        // whitespace / dash / dot → _
                .replace(/[^a-z0-9_]/g, '')        // strip everything else
                .replace(/_+/g, '_')
                .replace(/^_+|_+$/g, '');
            // Server requires a leading letter — prepend 'x_' if needed.
            if (slug && !/^[a-z]/.test(slug)) {
                slug = 'x_' + slug;
            }
            return slug;
        }

        const _ID_PATTERN = /^[a-z][a-z0-9_]*$/;

        function _modalEls() {
            return {
                overlay: document.getElementById('addModal'),
                content: document.querySelector('#addModal .modal-content'),
                title: document.getElementById('modalTitle'),
                nameInput: document.getElementById('modalInput'),
                idInput: document.getElementById('modalIdInput'),
                isFamily: document.getElementById('modalIsFamily'),
                emotion: document.getElementById('modalEmotionSelect'),
                error: document.getElementById('modalError'),
                confirm: document.getElementById('modalConfirmBtn'),
            };
        }

        function _setModalError(msg, field) {
            const e = _modalEls();
            e.error.textContent = msg || '';
            e.nameInput.classList.toggle('invalid', field === 'name');
            e.idInput.classList.toggle('invalid', field === 'id');
        }

        function _resetModal() {
            const e = _modalEls();
            e.nameInput.value = '';
            e.idInput.value = '';
            e.isFamily.checked = true;
            e.emotion.value = '溫和';
            e.nameInput.classList.remove('invalid');
            e.idInput.classList.remove('invalid');
            e.error.textContent = '';
            e.confirm.disabled = false;
            modalIdManuallyEdited = false;
        }

        // Generic opener. Sets type ('persona'|'listener'), title, and the
        // submit handler. Submit handler is async and may return
        // { ok: false, message, field } to keep the modal open with an
        // inline error.
        function openModal({ type, title, defaults, onSubmit }) {
            modalType = type;
            modalSubmit = onSubmit;
            const e = _modalEls();
            e.content.setAttribute('data-type', type);
            e.title.textContent = title;
            _resetModal();
            if (defaults) {
                if (defaults.name) e.nameInput.value = defaults.name;
                if (defaults.id) {
                    e.idInput.value = defaults.id;
                    modalIdManuallyEdited = true;
                }
                if (typeof defaults.is_family === 'boolean') e.isFamily.checked = defaults.is_family;
                if (defaults.default_emotion) e.emotion.value = defaults.default_emotion;
            }
            e.overlay.classList.add('visible');
            e.nameInput.focus();
        }

        function closeModal() {
            const e = _modalEls();
            e.overlay.classList.remove('visible');
            modalSubmit = null;
            modalType = null;
            _resetModal();
        }

        async function confirmModal() {
            if (!modalSubmit) {
                closeModal();
                return;
            }
            const e = _modalEls();
            const name = e.nameInput.value.trim();
            // If user never touched the id input, derive it now from the
            // current name so empty-id submissions still work.
            if (!modalIdManuallyEdited && !e.idInput.value.trim()) {
                e.idInput.value = slugifyName(name);
            }
            const id = e.idInput.value.trim();

            if (!name) {
                _setModalError('請輸入名稱', 'name');
                e.nameInput.focus();
                return;
            }
            if (!id) {
                _setModalError('請輸入 ID（或先輸入名稱以自動產生）', 'id');
                e.idInput.focus();
                return;
            }
            if (!_ID_PATTERN.test(id)) {
                _setModalError('ID 格式不正確：只能小寫字母、數字、底線；以字母開頭', 'id');
                e.idInput.focus();
                return;
            }

            _setModalError('', null);
            e.confirm.disabled = true;
            try {
                const result = await modalSubmit({
                    name,
                    id,
                    is_family: e.isFamily.checked,
                    default_emotion: e.emotion.value,
                });
                if (result && result.ok === false) {
                    _setModalError(result.message || '新增失敗', result.field || null);
                    e.confirm.disabled = false;
                    return;
                }
                closeModal();
            } catch (err) {
                _setModalError('新增失敗: ' + (err && err.message ? err.message : err), null);
                e.confirm.disabled = false;
            }
        }

        // Live ID suggestion as the user types the name — only as long as
        // they haven't manually edited the ID field.
        document.addEventListener('input', (ev) => {
            if (ev.target && ev.target.id === 'modalInput' && !modalIdManuallyEdited) {
                const idEl = document.getElementById('modalIdInput');
                if (idEl) idEl.value = slugifyName(ev.target.value);
            }
            if (ev.target && ev.target.id === 'modalIdInput') {
                // Mark manual edit once user touches the id field directly.
                modalIdManuallyEdited = true;
            }
        });

        // Close modal on Escape; submit on Enter (but only when the modal
        // is actually open, so global key bindings don't fire spuriously).
        document.addEventListener('keydown', (e) => {
            const overlay = document.getElementById('addModal');
            if (!overlay || !overlay.classList.contains('visible')) return;
            if (e.key === 'Escape') {
                e.preventDefault();
                closeModal();
            } else if (e.key === 'Enter' && e.target && e.target.tagName !== 'BUTTON') {
                e.preventDefault();
                confirmModal();
            }
        });

        // ==================== INITIALIZATION ====================
        async function init() {
            await Promise.all([loadPersonas(), loadListeners(), loadRecordings()]);
            buildListenerFilter();
            buildRecorderDropdowns();
            loadVersions();
            log('Recordings page loaded', 'info', 'UI');
        }

        async function loadPersonas() {
            try {
                const res = await fetch('/api/personas/');
                const data = await res.json();
                personas = data.personas || data || [];
                log(`Loaded ${personas.length} personas`, 'debug', 'UI');
            } catch (e) {
                log(`Failed to load personas: ${e.message}`, 'error', 'UI');
                personas = [];
            }
        }

        async function loadListeners() {
            try {
                const res = await fetch('/api/listeners/');
                const data = await res.json();
                listeners = data.listeners || data || [];
                log(`Loaded ${listeners.length} listeners`, 'debug', 'UI');
            } catch (e) {
                log(`Failed to load listeners: ${e.message}`, 'error', 'UI');
                listeners = [];
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

        function buildRecorderDropdowns() {
            // Build listener dropdown
            const currentListener = listenerSelect.value;
            listenerSelect.innerHTML = listeners.map(l =>
                `<option value="${l.listener_id}" ${l.listener_id === currentListener ? 'selected' : ''}>${escapeHtml(l.name) || l.listener_id}</option>`
            ).join('');

            // Build persona dropdown
            const currentPersona = personaSelect.value;
            personaSelect.innerHTML = personas.map(p =>
                `<option value="${p.persona_id}" ${p.persona_id === currentPersona ? 'selected' : ''}>${escapeHtml(p.name) || p.persona_id}</option>`
            ).join('');
        }

        // ==================== VERSION LOADING ====================
        async function loadVersions() {
            const versionSelect = document.getElementById('versionSelect');
            try {
                const personaId = personaSelect.value;
                const [versionsRes, activeRes] = await Promise.all([
                    fetch(`/api/training/versions?persona_id=${personaId}`),
                    fetch(`/api/training/active?persona_id=${personaId}`)
                ]);
                const versionsData = await versionsRes.json();
                const activeData = await activeRes.json();
                const activeVersionId = activeData.version?.version_id;

                versionSelect.innerHTML = '<option value="">系統預設</option>';
                (versionsData.versions || []).forEach(v => {
                    if (v.status === 'ready') {
                        const label = v.nickname ? `${v.version_id} (${escapeHtml(v.nickname)})` : v.version_id;
                        const selected = v.version_id === activeVersionId ? 'selected' : '';
                        versionSelect.innerHTML += `<option value="${v.version_id}" ${selected}>${label}</option>`;
                    }
                });

                if (activeVersionId) {
                    versionSelect.value = activeVersionId;
                }
            } catch (e) {
                log(`Failed to load versions: ${e.message}`, 'error', 'UI');
            }
        }

        // ==================== RECORDINGS LOADING ====================
        async function loadRecordings() {
            try {
                const response = await fetch('/api/recordings/');
                const data = await response.json();
                const recordings = Array.isArray(data) ? data : (data.recordings || []);
                allRecordings = recordings;
                filterAndRender();
                log(`Loaded ${recordings.length} recordings`, 'info', 'UI');

                const failed = recordings.filter(r => r.status === 'failed');
                if (failed.length > 0) {
                    failed.forEach(r => {
                        const errMsg = r.error_message || r.processing_steps?.transcribe?.error_message || 'Unknown error';
                        log(`Recording ${r.recording_id} FAILED: ${errMsg}`, 'error', 'PIPELINE');
                    });
                }
            } catch (e) {
                log(`Failed to load recordings: ${e.message}`, 'error', 'UI');
                treeView.innerHTML = '<div class="empty-state">載入失敗</div>';
            }
        }

        function filterAndRender() {
            const query = searchInput.value.toLowerCase().trim();
            const listenerId = listenerFilter.value.toLowerCase();

            let filtered = allRecordings;
            if (listenerId) {
                filtered = filtered.filter(r => (r.listener_id || '').toLowerCase() === listenerId);
            }
            if (query) {
                filtered = filtered.filter(r => {
                    const title = (r.title || r.folder_name || r.recording_id || '').toLowerCase();
                    const transcription = (r.transcription?.text || '').toLowerCase();
                    return title.includes(query) || transcription.includes(query);
                });
            }
            renderTree(filtered);
        }

        // ==================== TREE VIEW RENDERING ====================
        function renderTree(recordings) {
            if (recordings.length === 0) {
                treeView.innerHTML = '<div class="empty-state">尚無錄音</div>';
                return;
            }

            treeView.innerHTML = recordings.map(r => renderRecordingFolder(r)).join('');

            // Restore expanded state
            expandedRecordings.forEach(id => {
                const el = document.getElementById(`rec-${id}`);
                if (el) {
                    el.querySelector('.segments-container').classList.add('expanded');
                    el.querySelector('.expand-icon').classList.add('expanded');
                    el.querySelector('.recording-header').classList.add('expanded');
                }
            });
        }

        function renderRecordingFolder(r) {
            const speakers = r.speaker_segments || [];
            const statusClass = r.status === 'processed' ? 'ok' : r.status === 'failed' ? 'failed' : r.status === 'processing' ? 'processing' : '';
            const statusText = r.status === 'raw' ? '待處理' : r.status === 'processed' ? '已完成' : r.status === 'processing' ? '處理中' : r.status === 'failed' ? '失敗' : r.status;

            const listenerName = getListenerName(r.listener_id);
            const personaName = getPersonaName(r.persona_id);
            const durationText = r.duration_seconds ? r.duration_seconds.toFixed(1) + 's' : '-';

            const canPlay = r.status === 'processed' || r.status === 'raw';

            // Group segments by speaker_id and aggregate data
            const speakerGroups = {};
            for (const seg of speakers) {
                const sid = seg.speaker_id;
                if (!speakerGroups[sid]) {
                    speakerGroups[sid] = {
                        speaker_id: sid,
                        total_duration: 0,
                        transcripts: [],
                        quality_scores: [],
                        segments: [],
                        persona_id: seg.persona_id || null,
                        listener_id: seg.listener_id || null,
                        audio_path: seg.audio_path || null,
                        voice_audit: null,
                    };
                }
                speakerGroups[sid].total_duration += seg.duration_seconds || 0;
                speakerGroups[sid].segments.push(seg);
                if (seg.transcription?.text) {
                    speakerGroups[sid].transcripts.push(seg.transcription.text);
                }
                if (seg.quality_score !== null && seg.quality_score !== undefined) {
                    speakerGroups[sid].quality_scores.push(seg.quality_score);
                }
                // voice_audit is per-speaker (same on every segment); first non-null wins.
                if (!speakerGroups[sid].voice_audit && seg.voice_audit) {
                    speakerGroups[sid].voice_audit = seg.voice_audit;
                }
                // Use first non-null persona/listener
                if (!speakerGroups[sid].persona_id && seg.persona_id) {
                    speakerGroups[sid].persona_id = seg.persona_id;
                }
                if (!speakerGroups[sid].listener_id && seg.listener_id) {
                    speakerGroups[sid].listener_id = seg.listener_id;
                }
                // Use first available audio path
                if (!speakerGroups[sid].audio_path && seg.audio_path) {
                    speakerGroups[sid].audio_path = seg.audio_path;
                }
            }
            const uniqueSpeakers = Object.values(speakerGroups).sort((a, b) => a.speaker_id.localeCompare(b.speaker_id));

            return `
                <div class="tree-recording" id="rec-${r.recording_id}">
                    <div class="recording-header" onclick="toggleRecording('${r.recording_id}')">
                        <div class="expand-icon">▶</div>
                        <div class="recording-title">
                            <span class="recording-title-text">${r.title || r.folder_name || r.recording_id}</span>
                            <span class="recording-badge ${statusClass}">${statusText}</span>
                            <span class="recording-badge" style="background:#333;color:#888">[${listenerName}]</span>
                            <span class="recording-badge" style="background:#333;color:#888">[${personaName}]</span>
                        </div>
                        <div class="recording-meta">
                            <span>⏱ ${durationText}</span>
                            ${uniqueSpeakers.length > 0 ? `<span>🔊 ${uniqueSpeakers.length} 說話者</span>` : ''}
                        </div>
                        <div class="recording-actions" onclick="event.stopPropagation()">
                            ${canPlay ? `<button class="action-btn play" id="play-full-${r.recording_id}" onclick="toggleFullRecording('${r.recording_id}')" title="播放全部">⏵</button>` : ''}
                            ${r.status === 'processed' || r.status === 'raw' || r.status === 'failed' ? `<button class="action-btn parse" onclick="parseRecording('${r.recording_id}')">${r.status === 'failed' ? '重新解析' : r.status === 'processed' ? '重新解析' : '解析'}</button>` : ''}
                            ${r.status === 'processing' ? `<span class="loading"></span>` : ''}
                            <button class="action-btn delete" onclick="confirmDeleteRecording('${r.recording_id}')">✕</button>
                        </div>
                    </div>
                    <div class="transcription-container" id="transcription-${r.recording_id}">
                        <div class="transcription-header" onclick="toggleTranscription('${r.recording_id}')">
                            <span class="expand-icon">▶</span>
                            <span class="transcription-label">📝 轉譯稿</span>
                            ${r.transcription?.text ? `<span class="transcription-preview">${r.transcription.text.substring(0, 60)}${r.transcription.text.length > 60 ? '...' : ''}</span>` : '<span style="color:#888">無轉譯稿</span>'}
                            ${r.transcription?.confidence ? `<span class="transcription-confidence">${(r.transcription.confidence * 100).toFixed(0)}%</span>` : ''}
                        </div>
                        <div class="transcription-content" id="transcription-content-${r.recording_id}" style="display:none;">
                            <div class="transcription-text">${r.transcription?.text || '無轉譯稿'}</div>
                        </div>
                    </div>
                    <div class="segments-container" id="segments-${r.recording_id}">
                        ${r.status === 'processing' ? renderProcessingState(r) : ''}
                        ${uniqueSpeakers.length === 0 && r.status !== 'processing' ? '<div style="padding:12px;color:#888;font-size:0.85rem;">尚無分段</div>' : ''}
                        ${uniqueSpeakers.map(sg => renderSpeakerRow(r, sg)).join('')}
                    </div>
                </div>
            `;
        }

        function renderSpeakerRow(r, speakerGroup) {
            const speakerId = speakerGroup.speaker_id;
            const duration = speakerGroup.total_duration;
            const transcript = speakerGroup.transcripts.join(' ').substring(0, 100);
            const quality = speakerGroup.quality_scores.length > 0
                ? Math.max(...speakerGroup.quality_scores)
                : null;
            const personaId = speakerGroup.persona_id || r.speaker_labels?.[speakerId] || '';
            const listenerId = speakerGroup.listener_id || r.listener_id || '';
            const segmentCount = speakerGroup.segments?.length || 0;

            // Quality badge class
            const qualityClass = quality !== null
                ? (quality >= 0.8 ? 'quality-excellent' : quality >= 0.6 ? 'quality-good' : quality >= 0.4 ? 'quality-fair' : 'quality-poor')
                : '';
            const qualityLabel = quality !== null
                ? (quality >= 0.8 ? '優秀' : quality >= 0.6 ? '良好' : quality >= 0.4 ? '一般' : '惡劣')
                : '';
            const qualityBadge = quality !== null
                ? `<span class="quality-badge ${qualityClass}">${qualityLabel} ${(quality * 100).toFixed(0)}%</span>`
                : '';

            // Voice-cloning audit badge — separate from the segment SNR/clarity
            // quality_badge above. Tells the user if this speaker's audio is
            // broadband enough to train a non-muffled clone. Tooltip shows the
            // plain-language warnings and key metrics.
            const va = speakerGroup.voice_audit;
            let voiceAuditBadge = '';
            if (va && va.level) {
                const level = va.level;
                const label = level === 'good' ? '✓ 音質良好' : level === 'marginal' ? '⚠ 音質勉強' : level === 'bad' ? '✕ 不適合訓練' : '?';
                const cls = level === 'good' ? 'va-good' : level === 'marginal' ? 'va-marginal' : 'va-bad';
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

            return `
                <div class="segment-row" id="seg-${r.recording_id}-${speakerId}">
                    <span class="speaker-icon">👤</span>
                    <div class="speaker-info">
                        <span class="speaker-name">${speakerId}</span>
                        <span class="speaker-segments-count">${segmentCount} 段</span>
                        ${qualityBadge}
                        ${voiceAuditBadge}
                        ${transcript ? `<span class="speaker-transcript">${transcript}${transcript.length >= 100 ? '...' : ''}</span>` : ''}
                    </div>
                    <div class="segment-dropdowns">
                        <div class="dropdown-group">
                            <label>人格</label>
                            <select onchange="updateSegment('${r.recording_id}', '${speakerId}', 'persona_id', this.value)">
                                <option value="">--</option>
                                ${personas.map(p => `<option value="${p.persona_id}" ${p.persona_id === personaId ? 'selected' : ''}>${escapeHtml(p.name)}</option>`).join('')}
                            </select>
                            <button class="add-btn" onclick="openAddPersonaModal()" title="新增人格">+</button>
                        </div>
                        <div class="dropdown-group">
                            <label>對</label>
                            <select onchange="updateSegment('${r.recording_id}', '${speakerId}', 'listener_id', this.value)">
                                <option value="">--</option>
                                ${listeners.map(l => `<option value="${l.listener_id}" ${l.listener_id === listenerId ? 'selected' : ''}>${escapeHtml(l.name)}</option>`).join('')}
                            </select>
                            <button class="add-btn" onclick="openAddListenerModal()" title="新增聆聽者">+</button>
                        </div>
                    </div>
                    <span class="segment-duration">${duration.toFixed(1)}s</span>
                    <div class="playback-controls">
                        <button class="play-btn" id="play-${r.recording_id}-${speakerId}" onclick="toggleSegmentPlayback('${r.recording_id}', '${speakerId}')">▶</button>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" id="progress-${r.recording_id}-${speakerId}" onclick="seekSegment(event, '${r.recording_id}', '${speakerId}')">
                            <div class="progress-fill" id="progress-fill-${r.recording_id}-${speakerId}"></div>
                        </div>
                        <span class="progress-time" id="time-${r.recording_id}-${speakerId}">0:00 / ${formatTime(duration)}</span>
                    </div>
                </div>
            `;
        }

        function renderProcessingState(r) {
            const steps = r.processing_steps || {};
            const stepNames = { denoise: '降噪', enhance: '增強', diarize: '標記說話者', transcribe: '轉錄' };
            const stepOrder = ['denoise', 'enhance', 'diarize', 'transcribe'];
            let currentStep = '';
            for (const step of stepOrder) {
                if (steps[step]?.status === 'in_progress') {
                    currentStep = stepNames[step] || step;
                    break;
                }
            }

            // If a recording has been "processing" for >5 minutes, the
            // BackgroundTask almost certainly died (real pipeline takes
            // ~90s max). Show a retry button so the user can recover
            // without waiting for a server restart sweep.
            let stuck = false;
            if (r.updated_at) {
                const updatedMs = Date.parse(r.updated_at);
                if (!Number.isNaN(updatedMs)) {
                    stuck = (Date.now() - updatedMs) > 5 * 60 * 1000;
                }
            }
            const retryBtn = stuck
                ? `<button class="action-btn parse" style="margin-left:8px" onclick="retryStuckRecording('${r.recording_id}')" title="處理已停滯，點擊重試">🔄 重試</button>`
                : '';

            return `
                <div class="processing-state">
                    <span class="loading"></span>
                    <span>處理中: ${currentStep || '準備中'}...</span>
                    ${retryBtn}
                </div>
            `;
        }

        async function retryStuckRecording(recordingId) {
            if (gateIfTraining('重試錄音')) return;
            log(`Retrying stuck recording: ${recordingId}`, 'info', 'PIPELINE');

            // Optimistic local update — swap the "重試" affordance for a
            // fresh status string so we don't re-fire on the very next
            // poll tick while the server is still updating updated_at.
            const el = document.getElementById(`rec-${recordingId}`);
            if (el) {
                const processingState = el.querySelector('.processing-state');
                if (processingState) {
                    processingState.innerHTML = '<span class="loading"></span><span>處理中: 重新啟動...</span>';
                }
            }

            try {
                const response = await fetch(`/api/recordings/${recordingId}/process`, {
                    method: 'POST'
                });
                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${response.status}`);
                }
                log(`Retry triggered for ${recordingId}`, 'info', 'PIPELINE');
                showToast('已重新啟動處理', 'success');
                // Re-fetch so the row updates with fresh server state
                // (status=processing + new updated_at → button hidden).
                loadRecordings();
            } catch (e) {
                log(`Retry failed: ${e.message}`, 'error', 'PIPELINE');
                showToast('重試失敗: ' + e.message, 'error');
                // Re-render so the retry button reappears for another try.
                loadRecordings();
            }
        }

        function renderSegmentRow(r, segment) {
            const speakerId = segment.speaker_id;
            const duration = segment.duration_seconds || 0;
            const transcript = segment.transcription?.text || '';
            const quality = segment.quality_score || null;
            const personaId = segment.persona_id || r.speaker_labels?.[speakerId] || '';
            const listenerId = segment.listener_id || r.listener_id || '';

            const qualityHtml = quality !== null
                ? `<span class="quality-indicator ${quality >= 0.6 ? 'quality-good' : 'quality-bad'}">${quality >= 0.6 ? '✓' : '⚠'}</span>`
                : '';

            return `
                <div class="segment-row" id="seg-${r.recording_id}-${speakerId}">
                    <span class="speaker-icon">👤</span>
                    <div class="speaker-info">
                        <span class="speaker-name">${speakerId}</span>
                        ${transcript ? `<span class="speaker-transcript">${transcript.substring(0, 50)}${transcript.length > 50 ? '...' : ''}</span>` : ''}
                    </div>
                    <div class="segment-dropdowns">
                        <div class="dropdown-group">
                            <label>人格</label>
                            <select onchange="updateSegment('${r.recording_id}', '${speakerId}', 'persona_id', this.value)">
                                <option value="">--</option>
                                ${personas.map(p => `<option value="${p.persona_id}" ${p.persona_id === personaId ? 'selected' : ''}>${escapeHtml(p.name)}</option>`).join('')}
                            </select>
                            <button class="add-btn" onclick="openAddPersonaModal()" title="新增人格">+</button>
                        </div>
                        <div class="dropdown-group">
                            <label>對</label>
                            <select onchange="updateSegment('${r.recording_id}', '${speakerId}', 'listener_id', this.value)">
                                <option value="">--</option>
                                ${listeners.map(l => `<option value="${l.listener_id}" ${l.listener_id === listenerId ? 'selected' : ''}>${escapeHtml(l.name)}</option>`).join('')}
                            </select>
                            <button class="add-btn" onclick="openAddListenerModal()" title="新增聆聽者">+</button>
                        </div>
                    </div>
                    <span class="segment-duration">${duration.toFixed(1)}s</span>
                    <div class="playback-controls">
                        <button class="play-btn" id="play-${r.recording_id}-${speakerId}" onclick="toggleSegmentPlayback('${r.recording_id}', '${speakerId}')">▶</button>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" id="progress-${r.recording_id}-${speakerId}" onclick="seekSegment(event, '${r.recording_id}', '${speakerId}')">
                            <div class="progress-fill" id="progress-fill-${r.recording_id}-${speakerId}"></div>
                        </div>
                        <span class="progress-time" id="time-${r.recording_id}-${speakerId}">0:00 / ${formatTime(duration)}</span>
                    </div>
                    ${qualityHtml}
                </div>
            `;
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

        function formatTime(seconds) {
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return `${m}:${s.toString().padStart(2, '0')}`;
        }

        // ==================== TOGGLE EXPAND/COLLAPSE ====================
        function toggleRecording(recordingId) {
            const el = document.getElementById(`rec-${recordingId}`);
            const segments = document.getElementById(`segments-${recordingId}`);
            const icon = el.querySelector('.expand-icon');
            const header = el.querySelector('.recording-header');

            if (segments.classList.contains('expanded')) {
                segments.classList.remove('expanded');
                icon.classList.remove('expanded');
                header.classList.remove('expanded');
                icon.textContent = '▶';
                expandedRecordings.delete(recordingId);
            } else {
                segments.classList.add('expanded');
                icon.classList.add('expanded');
                header.classList.add('expanded');
                icon.textContent = '▼';
                expandedRecordings.add(recordingId);
            }
        }

        function toggleTranscription(recordingId) {
            const content = document.getElementById(`transcription-content-${recordingId}`);
            const header = content?.previousElementSibling;
            const icon = header?.querySelector('.expand-icon');

            if (!content) return;

            if (content.style.display === 'none') {
                content.style.display = 'block';
                if (icon) {
                    icon.classList.add('expanded');
                    icon.textContent = '▼';
                }
            } else {
                content.style.display = 'none';
                if (icon) {
                    icon.classList.remove('expanded');
                    icon.textContent = '▶';
                }
            }
        }

        // ==================== SEGMENT PLAYBACK ====================
        // Single-button toggle: ▶ starts, ■ stops. Same button serves both
        // roles — when audio is playing, the play-btn's text/title flip to
        // stop semantics. On `ended` (natural finish) we reset back to ▶.
        function toggleSegmentPlayback(recordingId, speakerId) {
            const key = `${recordingId}-${speakerId}`;
            if (activeAudio && currentPlayingId === key) {
                stopSegment();
            } else {
                playSegment(recordingId, speakerId);
            }
        }

        function playSegment(recordingId, speakerId) {
            // Stop any currently playing audio
            stopSegment();

            const playBtn = document.getElementById(`play-${recordingId}-${speakerId}`);

            log(`Playing segment: ${speakerId}`, 'info', 'PLAYBACK');

            fetch(`/api/recordings/${recordingId}/speaker/${speakerId}/audio`)
                .then(res => {
                    if (!res.ok) throw new Error('Audio not available');
                    return res.blob();
                })
                .then(blob => {
                    const url = URL.createObjectURL(blob);
                    activeAudio = new Audio(url);
                    currentPlayingId = `${recordingId}-${speakerId}`;

                    activeAudio.addEventListener('timeupdate', () => {
                        updateProgress(recordingId, speakerId);
                    });

                    activeAudio.addEventListener('ended', () => {
                        resetPlaybackUI(recordingId, speakerId);
                        activeAudio = null;
                        currentPlayingId = null;
                    });

                    activeAudio.addEventListener('error', (e) => {
                        log(`Audio error: ${e.message}`, 'error', 'PLAYBACK');
                        resetPlaybackUI(recordingId, speakerId);
                        activeAudio = null;
                        currentPlayingId = null;
                    });

                    if (playBtn) {
                        playBtn.textContent = '■';
                        playBtn.title = '停止播放';
                    }
                    activeAudio.play();
                })
                .catch(e => {
                    log(`Failed to play segment: ${e.message}`, 'error', 'PLAYBACK');
                    showToast('播放失敗: ' + e.message, 'error');
                });
        }

        function stopSegment() {
            if (activeAudio) {
                activeAudio.pause();
                activeAudio.currentTime = 0;
                if (currentPlayingId) {
                    const [recordingId, speakerId] = currentPlayingId.split('-').slice(-2);
                    resetPlaybackUI(recordingId, speakerId);
                }
                activeAudio = null;
                currentPlayingId = null;
            }
        }

        function resetPlaybackUI(recordingId, speakerId) {
            const playBtn = document.getElementById(`play-${recordingId}-${speakerId}`);
            const progressFill = document.getElementById(`progress-fill-${recordingId}-${speakerId}`);
            if (playBtn) {
                playBtn.textContent = '▶';
                playBtn.title = '播放';
            }
            if (progressFill) progressFill.style.width = '0%';
        }

        function updateProgress(recordingId, speakerId) {
            if (!activeAudio) return;
            const progressFill = document.getElementById(`progress-fill-${recordingId}-${speakerId}`);
            const timeDisplay = document.getElementById(`time-${recordingId}-${speakerId}`);
            if (!progressFill || !timeDisplay) return;

            const percent = (activeAudio.currentTime / activeAudio.duration) * 100;
            progressFill.style.width = `${percent}%`;

            const current = formatTime(activeAudio.currentTime);
            const total = formatTime(activeAudio.duration || 0);
            timeDisplay.textContent = `${current} / ${total}`;
        }

        function seekSegment(event, recordingId, speakerId) {
            if (!activeAudio || currentPlayingId !== `${recordingId}-${speakerId}`) return;

            const progressBar = document.getElementById(`progress-${recordingId}-${speakerId}`);
            const rect = progressBar.getBoundingClientRect();
            const percent = (event.clientX - rect.left) / rect.width;
            activeAudio.currentTime = percent * activeAudio.duration;
        }

        // ==================== FULL RECORDING PLAYBACK ====================
        // Toggle pattern matching segment playback: same button starts/stops.
        function toggleFullRecording(recordingId) {
            const key = `full-${recordingId}`;
            if (activeAudio && currentPlayingId === key) {
                stopFullRecording();
            } else {
                playFullRecording(recordingId);
            }
        }

        function _resetFullRecordingBtn(recordingId) {
            const btn = document.getElementById(`play-full-${recordingId}`);
            if (btn) {
                btn.textContent = '⏵';
                btn.title = '播放全部';
            }
        }

        function stopFullRecording() {
            if (!activeAudio) return;
            const recordingId = (currentPlayingId || '').replace(/^full-/, '');
            activeAudio.pause();
            activeAudio.currentTime = 0;
            activeAudio = null;
            currentPlayingId = null;
            if (recordingId) _resetFullRecordingBtn(recordingId);
        }

        function playFullRecording(recordingId) {
            stopSegment();
            stopFullRecording();

            log(`Playing full recording: ${recordingId}`, 'info', 'PLAYBACK');

            fetch(`/api/recordings/${recordingId}/stream?stage=enhanced`)
                .then(res => {
                    if (!res.ok) throw new Error('Stream not available');
                    return res.blob();
                })
                .then(blob => {
                    const url = URL.createObjectURL(blob);
                    activeAudio = new Audio(url);
                    currentPlayingId = `full-${recordingId}`;

                    const btn = document.getElementById(`play-full-${recordingId}`);
                    if (btn) {
                        btn.textContent = '■';
                        btn.title = '停止播放';
                    }

                    activeAudio.addEventListener('ended', () => {
                        activeAudio = null;
                        currentPlayingId = null;
                        _resetFullRecordingBtn(recordingId);
                    });

                    activeAudio.addEventListener('error', () => {
                        activeAudio = null;
                        currentPlayingId = null;
                        _resetFullRecordingBtn(recordingId);
                    });

                    activeAudio.play();
                    showToast('開始播放', 'info');
                })
                .catch(e => {
                    log(`Playback failed: ${e.message}`, 'error', 'PLAYBACK');
                    showToast('播放失敗: ' + e.message, 'error');
                });
        }

        // ==================== SEGMENT UPDATE ====================
        async function updateSegment(recordingId, speakerId, field, value) {
            log(`Updating segment: ${speakerId} ${field} = ${value}`, 'info', 'UI');

            const payload = { [field]: value || null };

            try {
                const response = await fetch(`/api/recordings/${recordingId}/segments/${speakerId}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) throw new Error('Update failed');

                // Also update speaker_labels in metadata if persona changed
                if (field === 'persona_id') {
                    const rec = allRecordings.find(r => r.recording_id === recordingId);
                    if (rec) {
                        const labels = { ...(rec.speaker_labels || {}) };
                        if (value) {
                            labels[speakerId] = value;
                        } else {
                            delete labels[speakerId];
                        }
                        await fetch(`/api/recordings/${recordingId}/speakers`, {
                            method: 'PATCH',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ speaker_labels: labels })
                        });
                    }
                }

                showToast('已更新', 'success');
            } catch (e) {
                log(`Update failed: ${e.message}`, 'error', 'UI');
                showToast('更新失敗: ' + e.message, 'error');
            }
        }

        // ==================== PARSE / RE-PARSE ====================
        async function parseRecording(recordingId) {
            if (gateIfTraining('解析錄音')) return;
            log(`Parsing recording: ${recordingId}`, 'info', 'PIPELINE');

            const el = document.getElementById(`rec-${recordingId}`);
            const actionsDiv = el.querySelector('.recording-actions');
            const originalHTML = actionsDiv.innerHTML;
            actionsDiv.innerHTML = '<span class="loading"></span>';

            try {
                const response = await fetch(`/api/recordings/${recordingId}/process`, {
                    method: 'POST'
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || `HTTP ${response.status}`);
                }

                const result = await response.json();
                log(`Processing started: ${result.status} (polling for completion...)`, 'info', 'PIPELINE');
                showToast('開始解析錄音...', 'success');

                // Poll for completion
                await pollProcessingStatus(recordingId, actionsDiv, originalHTML);

            } catch (e) {
                log(`Parse failed: ${e.message}`, 'error', 'PIPELINE');
                showToast('解析失敗: ' + e.message, 'error');
                actionsDiv.innerHTML = originalHTML;
            }
        }

        async function pollProcessingStatus(recordingId, actionsDiv, originalHTML) {
            const maxWait = 300; // 5 minutes max
            const interval = 3000; // Poll every 3 seconds
            let waited = 0;

            while (waited < maxWait) {
                await new Promise(r => setTimeout(r, interval));
                waited += interval / 1000;

                try {
                    // Get updated recording data
                    const recResponse = await fetch('/api/recordings');
                    if (!recResponse.ok) {
                        log(`Poll ${waited}s: API returned ${recResponse.status}`, 'warning', 'PIPELINE');
                        continue;
                    }
                    const recData = await recResponse.json();
                    // API returns paginated {recordings: [...]} or direct [...]
                    let recArray;
                    if (Array.isArray(recData)) {
                        recArray = recData;
                    } else if (recData.recordings && Array.isArray(recData.recordings)) {
                        recArray = recData.recordings;
                    } else {
                        log(`Poll ${waited}s: unexpected response type: ${typeof recData}`, 'warning', 'PIPELINE');
                        if (recData && typeof recData === 'object') {
                            log(`Poll response keys: ${Object.keys(recData).join(', ')}`, 'warning', 'PIPELINE');
                        } else {
                            log(`Poll raw response: ${String(recData).substring(0, 100)}`, 'warning', 'PIPELINE');
                        }
                        continue;
                    }
                    const rec = recArray.find(r => r.recording_id === recordingId);

                    if (!rec) {
                        log(`Recording ${recordingId} not found during poll`, 'error', 'PIPELINE');
                        break;
                    }

                    log(`Poll ${waited}s: status=${rec.status}, segments=${rec.speaker_segments?.length || 0}`, 'info', 'PIPELINE');

                    if (rec.status === 'processed') {
                        log(`Parse complete! Found ${rec.speaker_segments?.length || 0} segments`, 'info', 'PIPELINE');
                        loadRecordings();
                        if (rec.speaker_segments?.length > 0) {
                            showToast(`解析完成！找到 ${rec.speaker_segments.length} 個片段`, 'success');
                        } else {
                            showToast('解析完成但沒有分段', 'warning');
                        }
                        return;
                    } else if (rec.status === 'failed') {
                        log(`Parse failed: recording status is failed`, 'error', 'PIPELINE');
                        showToast('解析失敗', 'error');
                        actionsDiv.innerHTML = originalHTML;
                        return;
                    }
                    // Still processing...

                } catch (e) {
                    log(`Poll error: ${e.message}`, 'warning', 'PIPELINE');
                }
            }

            log(`Parse timeout after ${maxWait}s`, 'error', 'PIPELINE');
            showToast('解析超時', 'error');
            actionsDiv.innerHTML = originalHTML;
        }

        // ==================== DELETE RECORDING ====================
        function confirmDeleteRecording(recordingId) {
            const el = document.getElementById(`rec-${recordingId}`);
            const actionsDiv = el.querySelector('.recording-actions');
            actionsDiv.innerHTML = `
                <div class="delete-confirm">
                    <span>確認刪除？</span>
                    <button class="action-btn delete" onclick="deleteRecording('${recordingId}')">刪除</button>
                    <button class="action-btn cancel-btn" onclick="loadRecordings()">取消</button>
                </div>
            `;
        }

        async function deleteRecording(recordingId) {
            log(`Deleting recording: ${recordingId}`, 'info', 'UI');
            try {
                const response = await fetch(`/api/recordings/${recordingId}`, { method: 'DELETE' });
                if (!response.ok) throw new Error('Delete failed');
                showToast('已刪除', 'success');
                loadRecordings();
            } catch (e) {
                log(`Delete failed: ${e.message}`, 'error', 'UI');
                showToast('刪除失敗: ' + e.message, 'error');
            }
        }

        // ==================== ADD PERSONA/LISTENER ====================
        // Maps an API JSON error envelope to an inline modal message. The
        // backend returns { error, message, details, detail } — see
        // app/api/_errors.py. We surface a friendly Chinese string keyed off
        // the `error` code, falling back to the server's message.
        function _modalErrorFromApi(status, body) {
            const code = (body && body.error) || '';
            const apiMsg = (body && (body.message || body.detail)) || '';
            switch (code) {
                case 'duplicate_id':
                    return { message: 'ID 重複，請改用其他 ID', field: 'id' };
                case 'invalid_id_format':
                    return { message: 'ID 格式不正確：只能小寫字母、數字、底線；以字母開頭', field: 'id' };
                case 'invalid_emotion':
                    return { message: '無效的情緒值', field: null };
                case 'fixed_persona_readonly':
                    return { message: '此 ID 為系統固定人格，無法建立', field: 'id' };
            }
            if (status === 400) return { message: apiMsg || '格式不正確', field: null };
            if (status === 409) return { message: apiMsg || 'ID 重複', field: 'id' };
            if (status === 422) return { message: apiMsg || '欄位驗證失敗', field: null };
            return { message: apiMsg || `新增失敗 (HTTP ${status})`, field: null };
        }

        function openAddPersonaModal() {
            openModal({
                type: 'persona',
                title: '新增人格',
                onSubmit: async ({ name, id, is_family }) => {
                    let res, body;
                    try {
                        res = await fetch('/api/personas/', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ persona_id: id, name, is_family }),
                        });
                        body = await res.json().catch(() => ({}));
                    } catch (e) {
                        return { ok: false, message: '網路錯誤: ' + e.message };
                    }
                    if (!res.ok) {
                        return { ok: false, ...(_modalErrorFromApi(res.status, body)) };
                    }
                    // Server returns the persona; auto-select by its real id.
                    const newId = (body && body.persona_id) || id;
                    await loadPersonas();
                    buildRecorderDropdowns();
                    personaSelect.value = newId;
                    // Refresh per-segment dropdowns so the new option appears.
                    filterAndRender();
                    showToast('新增成功', 'success');
                    log(`Persona created: ${newId} (${name})`, 'info', 'UI');
                    return { ok: true };
                },
            });
        }

        function openAddListenerModal() {
            openModal({
                type: 'listener',
                title: '新增聆聽者',
                defaults: { is_family: false, default_emotion: '溫和' },
                onSubmit: async ({ name, id, is_family, default_emotion }) => {
                    let res, body;
                    try {
                        res = await fetch('/api/listeners/', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                listener_id: id,
                                name,
                                is_family,
                                default_emotion,
                            }),
                        });
                        body = await res.json().catch(() => ({}));
                    } catch (e) {
                        return { ok: false, message: '網路錯誤: ' + e.message };
                    }
                    if (!res.ok) {
                        return { ok: false, ...(_modalErrorFromApi(res.status, body)) };
                    }
                    const newId = (body && body.listener_id) || id;
                    await loadListeners();
                    buildListenerFilter();
                    buildRecorderDropdowns();
                    listenerSelect.value = newId;
                    filterAndRender();
                    showToast('新增成功', 'success');
                    log(`Listener created: ${newId} (${name})`, 'info', 'UI');
                    return { ok: true };
                },
            });
        }

        // ==================== WEBRTC RECORDING ====================
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let startTime = 0;
        let durationInterval = null;
        let audioContext = null;
        let analyser = null;
        let currentStream = null;

        async function startRecording() {
            log('Starting WebRTC recording...', 'info', 'RECORDING');
            try {
                // Disable browser's voice-processing chain for TRAINING-DATA
                // capture. Chrome's default getUserMedia({audio:true}) enables
                // echoCancellation, autoGainControl, and noiseSuppression —
                // EC in particular applies a ~3.5–4 kHz lowpass that narrows
                // the captured signal to telephone band even on a 48 kHz mic.
                // For chat (the /ui page) those features make sense; for
                // training data they destroy the bandwidth we need for a
                // non-muffled clone.
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        autoGainControl: false,
                        noiseSuppression: false,
                        sampleRate: 48000,
                        channelCount: 1,
                    }
                });
                currentStream = stream;

                // Log the actual settings the browser applied — some browsers
                // ignore constraints silently and fall back to defaults.
                const track = stream.getAudioTracks()[0];
                if (track) {
                    log(`Mic settings: ${JSON.stringify(track.getSettings())}`, 'info', 'RECORDING');
                }

                audioContext = new AudioContext();
                analyser = audioContext.createAnalyser();
                const source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);
                analyser.fftSize = 256;

                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm',
                    audioBitsPerSecond: 128000,  // 128 kbps Opus — broadband; browser default can be 32 kbps voice-mode
                });
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

                mediaRecorder.start(100);
                isRecording = true;
                startTime = Date.now();

                recBtn.classList.add('recording');
                recBtn.querySelector('#recIcon').textContent = '■';
                recBtn.querySelector('#recText').textContent = '停止錄音';
                qualityIndicator.textContent = '錄音中...';

                durationInterval = setInterval(() => {
                    const elapsed = Math.floor((Date.now() - startTime) / 1000);
                    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
                    const secs = (elapsed % 60).toString().padStart(2, '0');
                    duration.textContent = `${mins}:${secs}`;
                    // Auto-stop sanity guard matches server MAX_DURATION_SECONDS.
                    // Bumped 300→3600 (1 hour) 2026-05-20 for client/demo
                    // longer-take workflows.
                    if (elapsed >= 3600) stopRecording();
                }, 1000);

                updateDbMeter();
                log('Recording started', 'info', 'RECORDING');
            } catch (e) {
                log(`Failed to start recording: ${e.message}`, 'error', 'RECORDING');
                alert('無法訪問麥克風: ' + e.message);
            }
        }

        function updateDbMeter() {
            if (!isRecording || !analyser) return;

            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(dataArray);

            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i] * dataArray[i];
            }
            const rms = Math.sqrt(sum / dataArray.length);
            const db = 20 * Math.log10(rms / 255);

            const percent = Math.min(100, Math.max(0, (db + 60) * 100 / 60));
            dbMeterFill.style.width = percent + '%';
            dbLevel.textContent = db > -60 ? `${db.toFixed(1)} dB` : '-∞ dB';

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

        function stopRecording() {
            if (!isRecording) return;

            log('Stopping recording...', 'info', 'RECORDING');
            isRecording = false;

            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }

            if (currentStream) {
                currentStream.getTracks().forEach(track => track.stop());
            }

            if (durationInterval) {
                clearInterval(durationInterval);
                durationInterval = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }

            recBtn.classList.remove('recording');
            recBtn.querySelector('#recIcon').textContent = '●';
            recBtn.querySelector('#recText').textContent = '開始錄音';
            duration.textContent = '00:00';
            dbMeterFill.style.width = '0%';
            dbLevel.textContent = '-∞ dB';
            qualityIndicator.textContent = '等待錄音...';
            qualityIndicator.style.color = '#888';
        }

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
                showToast('錄音上傳成功！正在處理...', 'success');
                loadRecordings();
                await triggerProcessing(result.recording_id);

            } catch (e) {
                log(`Upload failed: ${e.message}`, 'error', 'UPLOAD');
                showToast('上傳失敗: ' + e.message, 'error');
            }
        }

        async function triggerProcessing(recordingId) {
            if (gateIfTraining('處理錄音')) return;
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

        async function triggerBackup() {
            const btn = document.getElementById('backupBtn');
            btn.disabled = true;
            btn.textContent = '備份中...';
            log('Starting R2 backup...', 'info', 'BACKUP');

            try {
                const response = await fetch('/api/recordings/backup', { method: 'POST' });
                const result = await response.json();

                if (result.status === 'timeout') {
                    log(`Backup timeout: ${result.message}`, 'error', 'BACKUP');
                    showToast('備份超時！', 'error');
                } else if (result.status === 'success') {
                    log(`Backup complete! Lines: ${result.total_lines}`, 'info', 'BACKUP');
                    showToast('備份成功！', 'success');
                } else {
                    log(`Backup completed with issues. Errors: ${result.errors?.length || 0}`, 'warning', 'BACKUP');
                    if (result.errors?.length > 0) {
                        result.errors.forEach(e => log(`Backup error: ${e}`, 'error', 'BACKUP'));
                    }
                    if (result.output_lines?.length > 0) {
                        result.output_lines.slice(-10).forEach(l => log(`Backup output: ${l}`, 'info', 'BACKUP'));
                    }
                    showToast(`備份完成但有錯誤: ${result.errors?.length || 0}個`, 'warning');
                }
            } catch (e) {
                log(`Backup failed: ${e.message}`, 'error', 'BACKUP');
                showToast('備份失敗: ' + e.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '📤 備份到R2';
            }
        }

        // ==================== FILE UPLOAD ====================
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
            let successCount = 0;
            let failCount = 0;
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
                    successCount++;
                } catch (e) {
                    log(`File upload failed: ${e.message}`, 'error', 'UPLOAD');
                    failCount++;
                }
            }
            loadRecordings();
            if (failCount === 0 && successCount > 0) {
                showToast(`${successCount} 個檔案上傳成功，正在處理...`, 'success');
            } else if (failCount > 0) {
                showToast(`${failCount} 個檔案上傳失敗`, 'error');
            }
        }

        // ==================== EVENT LISTENERS ====================
        recBtn.addEventListener('click', () => {
            if (isRecording) stopRecording();
            else startRecording();
        });

        // Debug panel toggle
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

        // ==================== AUTO REFRESH ====================
        init();

        setInterval(() => {
            if (!document.hidden) {
                loadRecordings();
            }
        }, 5000);

        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                loadRecordings();
            }
        });
