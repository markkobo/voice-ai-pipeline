// Shared system-status poller. Loaded by every dev-UI page via
// <script src="/static/js/_status_bar.js" defer></script>.
//
// Polls /api/system/status every 5s and updates the status bar pills.
// Exposes:
//   window.SYS                   — { trainingActive, ttsReady, asrReady }
//   window.gateIfTraining(label) — alert + return true if training in flight
//   window.SYS_ON_UPDATE          — array of (status) => void hooks each
//                                   page can push into for page-specific
//                                   gating (e.g. training UI disables
//                                   startTrainingBtn during training).
//
// RFC_M6 Phase 0-pre review #28: replaces the 3× duplicated copy of
// this logic across standalone.js / recordings.js / training.js.

(function () {
    // Demo mode — `?demo=1` enables, `?demo=0` disables. State persists
    // for the session via sessionStorage so navigating between chat /
    // recordings / training keeps the clean product surface without
    // re-appending the query string to every link.
    const _demoQ = new URLSearchParams(location.search).get('demo');
    if (_demoQ === '1') sessionStorage.setItem('demoMode', '1');
    else if (_demoQ === '0') sessionStorage.removeItem('demoMode');
    if (sessionStorage.getItem('demoMode') === '1') {
        document.documentElement.classList.add('demo-mode');
        document.addEventListener('DOMContentLoaded', () => {
            document.body.classList.add('demo-mode');
        }, { once: true });
    }

    window.SYS = { trainingActive: false, ttsReady: false, asrReady: false };
    window.SYS_ON_UPDATE = window.SYS_ON_UPDATE || [];

    // Pretty-print a raw version id like `test_v9_20260527_165609_341271`
    // → `Test v9 (5/27 16:56)`. Falls back to the raw id if it doesn't
    // match the known pattern, so untrained / custom names still render.
    // Used by the status-bar voice pill + per-page version dropdowns.
    const PERSONA_DISPLAY = {
        xiao_s: 'Xiao S', test: 'Test', caregiver: 'Caregiver',
        elder_gentle: 'Elder — Gentle', elder_playful: 'Elder — Playful',
    };
    window.formatVersionName = function (raw) {
        if (!raw || raw === 'default') return 'System Default';
        // pattern: {persona}_v{N}_YYYYMMDD_HHMMSS_{hash}
        const m = raw.match(/^(.+?)_v(\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\d{2}_\w+$/);
        if (!m) return raw;
        const [, persona, vnum, , mm, dd, HH, MM] = m;
        const personaDisplay = PERSONA_DISPLAY[persona] || persona;
        return `${personaDisplay} v${vnum} (${parseInt(mm, 10)}/${parseInt(dd, 10)} ${HH}:${MM})`;
    };

    async function pollSystemStatus() {
        try {
            const res = await fetch('/api/system/status', { cache: 'no-store' });
            if (!res.ok) return;
            const s = await res.json();
            window.SYS.trainingActive = !!(s.training && s.training.active);
            window.SYS.ttsReady = !!(s.tts && s.tts.ready);
            window.SYS.asrReady = !!s.asr_ready;

            // VRAM bar
            const vBar = document.getElementById('sysVramFill');
            const vText = document.getElementById('sysVramText');
            if (vBar && vText) {
                if (s.vram && s.vram.available) {
                    const pct = Math.round((s.vram.used_mb / s.vram.total_mb) * 100);
                    vBar.style.width = pct + '%';
                    vBar.classList.toggle('warn', pct >= 70 && pct < 88);
                    vBar.classList.toggle('high', pct >= 88);
                    vText.textContent = `${s.vram.used_mb} / ${s.vram.total_mb} MB`;
                } else {
                    vText.textContent = 'no GPU';
                }
            }
            // Voice pill — pretty-print known persona_v{N}_date_hash pattern;
            // raw id stays in the pill title= for diagnostics.
            const voiceText = document.getElementById('sysVoiceText');
            const voicePill = document.getElementById('sysVoice');
            if (voiceText) {
                const rawV = s.tts && s.tts.active_version;
                voiceText.textContent = rawV ? window.formatVersionName(rawV) : 'System Default';
                if (voicePill) voicePill.title = rawV ? `Active TTS voice: ${rawV}` : 'Active TTS voice';
            }
            // ASR pill
            const asrEl = document.getElementById('sysAsr');
            const asrText = document.getElementById('sysAsrText');
            if (asrEl && asrText) {
                asrEl.classList.toggle('ok', !!s.asr_ready);
                asrText.textContent = s.asr_ready ? 'ready' : 'loading';
            }
            // Disk pill
            const diskText = document.getElementById('sysDiskText');
            if (diskText) diskText.textContent = s.disk_free_gb;
            // Training pill — formats:
            //   training phase, pct only   → "training 23%"
            //   training phase, both       → "training 23% 7/30"
            //   merging phase              → "merging…"  (parent is merging
            //                                  LoRA → base, can take several
            //                                  minutes for 1.7B — without
            //                                  this the pill would stay
            //                                  stuck on "training 100% 10/10")
            //   neither                    → "training…"  (fallback)
            // The pill uses `white-space: nowrap`; on narrow viewports the
            // whole pill wraps to a new row rather than being clipped.
            const tEl = document.getElementById('sysTraining');
            if (tEl) {
                if (window.SYS.trainingActive) {
                    const t = s.training || {};
                    const trainingText = document.getElementById('sysTrainingText');
                    if (trainingText) {
                        if (t.phase === 'merging') {
                            trainingText.textContent = 'merging…';
                        } else {
                            const parts = ['training'];
                            if (t.progress_pct != null) parts.push(t.progress_pct + '%');
                            if (t.current_epoch != null && t.total_epochs != null) {
                                parts.push(`${t.current_epoch}/${t.total_epochs}`);
                            }
                            trainingText.textContent = parts.length > 1
                                ? parts.join(' ')
                                : 'training…';
                        }
                    }
                    // Title for hover detail (helps on mobile via long-press).
                    if (t.phase === 'merging') {
                        tEl.title = 'merging LoRA adapter into base model';
                    } else if (t.current_loss != null) {
                        tEl.title = `training — loss ${t.current_loss.toFixed(3)}`;
                    } else {
                        tEl.title = 'training in progress';
                    }
                    tEl.style.display = '';
                } else {
                    tEl.style.display = 'none';
                }
            }
            document.body.classList.toggle('training-active', window.SYS.trainingActive);

            // Fire per-page hooks for additional gating.
            for (const hook of window.SYS_ON_UPDATE) {
                try { hook(s); }
                catch (e) { console.error('SYS_ON_UPDATE hook failed', e); }
            }
        } catch (e) {
            /* silent — next tick retries */
        }
    }

    window.gateIfTraining = function (label) {
        if (window.SYS.trainingActive) {
            alert(`Training in progress — "${label}" is unavailable (GPU is busy)`);
            return true;
        }
        return false;
    };

    // Expose a manual-poll hook so pages can trigger an immediate refresh
    // after an action that's expected to change status (e.g. chat persona
    // switch → server eagerly activates a new TTS model; user shouldn't
    // wait up to 5s to see it land in the status bar).
    window.SYS_FORCE_POLL = pollSystemStatus;

    setInterval(pollSystemStatus, 5000);
    pollSystemStatus();
})();
