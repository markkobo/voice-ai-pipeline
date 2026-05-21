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
    window.SYS = { trainingActive: false, ttsReady: false, asrReady: false };
    window.SYS_ON_UPDATE = window.SYS_ON_UPDATE || [];

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
            // Voice pill
            const voiceText = document.getElementById('sysVoiceText');
            if (voiceText) {
                voiceText.textContent = s.tts && s.tts.active_version
                    ? s.tts.active_version.replace('xiao_s_', '')
                    : '(base)';
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
            alert(`訓練進行中，無法執行「${label}」 (GPU is busy)`);
            return true;
        }
        return false;
    };

    setInterval(pollSystemStatus, 5000);
    pollSystemStatus();
})();
