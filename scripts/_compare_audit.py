#!/usr/bin/env python3
"""Compare baseline vs current voice_audit across all recordings.
Outputs:
  - per-recording diff
  - summary statistics
"""
import json, os, sys
from collections import Counter, defaultdict

BASELINE = "/home/rding/voice-ai-pipeline/scripts/_baseline_audit.json"
CURRENT  = "/home/rding/voice-ai-pipeline/scripts/_after_audit.json"
PROGRESS = "/home/rding/voice-ai-pipeline/scripts/_reprocess_progress.json"
OUT_MD   = "/home/rding/voice-ai-pipeline/scripts/_compare_report.md"

LEVEL_RANK = {"bad": 0, "marginal": 1, "good": 2, None: -1}

with open(BASELINE) as f: base = json.load(f)
with open(CURRENT)  as f: curr = json.load(f)
try:
    with open(PROGRESS) as f: prog = json.load(f)
except FileNotFoundError:
    prog = None

lines = []
def w(s=""):
    lines.append(s)

# Per-recording diff
w("# Pipeline Reprocess: Before vs After")
w()

# Overall counts
n_recs_before = len(base)
n_recs_after  = len(curr)
total_spk_before = sum(r["n_speakers"] for r in base.values())
total_spk_after  = sum(r["n_speakers"] for r in curr.values())

w(f"## Overall")
w(f"- Recordings in baseline: {n_recs_before}")
w(f"- Recordings in current : {n_recs_after}")
w(f"- Total speakers baseline: {total_spk_before}")
w(f"- Total speakers current : {total_spk_after}")
w(f"- Net speakers eliminated: {total_spk_before - total_spk_after}")
w()

# Reprocess outcomes
if prog:
    outcome = Counter(r["outcome"] for r in prog.get("results", []))
    w("## Reprocess outcomes (from progress log)")
    for k, v in outcome.items():
        w(f"- {k}: {v}")
    w(f"- Total elapsed: {prog.get('elapsed_overall_s',0):.0f}s ({prog.get('elapsed_overall_s',0)/60:.1f} min)")
    w()
    failed = [r for r in prog["results"] if r["outcome"] != "processed"]
    if failed:
        w("### Failed/timeout recordings")
        for r in failed:
            w(f"- {r['folder']}  outcome={r['outcome']}  err={r['error_message']}")
        w()

# Level transition matrix
transitions = Counter()
ghost_dropped_count = 0
new_speakers_count  = 0
peak_dbfs_before, peak_dbfs_after = [], []
rms_dbfs_before, rms_dbfs_after = [], []
bw_before, bw_after = [], []
silent_before, silent_after = [], []

improvements = []
regressions = []
ghost_drops = []
new_in_after = []

per_rec = []

for folder, b in base.items():
    c = curr.get(folder)
    if not c:
        per_rec.append({"folder": folder, "missing_in_after": True})
        continue
    b_spk = b.get("speakers", {})
    c_spk = c.get("speakers", {})
    spk_diffs = {}
    for sid in sorted(set(b_spk) | set(c_spk)):
        bs = b_spk.get(sid)
        cs = c_spk.get(sid)
        if bs and not cs:
            ghost_dropped_count += 1
            ghost_drops.append((folder, sid, bs.get("level"), bs.get("peak"), bs.get("rms_dbfs")))
            spk_diffs[sid] = {"before": bs, "after": None, "transition": f"{bs.get('level')} -> DROPPED"}
            continue
        if cs and not bs:
            new_speakers_count += 1
            new_in_after.append((folder, sid, cs.get("level")))
            spk_diffs[sid] = {"before": None, "after": cs, "transition": f"NEW -> {cs.get('level')}"}
            continue
        bl, cl = bs.get("level"), cs.get("level")
        transitions[(bl, cl)] += 1
        if LEVEL_RANK[cl] > LEVEL_RANK[bl]:
            improvements.append((folder, sid, bl, cl))
        elif LEVEL_RANK[cl] < LEVEL_RANK[bl]:
            regressions.append((folder, sid, bl, cl))
        if bs.get("peak_dbfs") is not None and cs.get("peak_dbfs") is not None:
            peak_dbfs_before.append(bs["peak_dbfs"]); peak_dbfs_after.append(cs["peak_dbfs"])
        if bs.get("rms_dbfs") is not None and cs.get("rms_dbfs") is not None:
            rms_dbfs_before.append(bs["rms_dbfs"]); rms_dbfs_after.append(cs["rms_dbfs"])
        if bs.get("effective_bandwidth_hz") is not None and cs.get("effective_bandwidth_hz") is not None:
            bw_before.append(bs["effective_bandwidth_hz"]); bw_after.append(cs["effective_bandwidth_hz"])
        if bs.get("silent_pct") is not None and cs.get("silent_pct") is not None:
            silent_before.append(bs["silent_pct"]); silent_after.append(cs["silent_pct"])
        spk_diffs[sid] = {
            "before": bs, "after": cs,
            "transition": f"{bl} -> {cl}",
            "rms_shift_db": (cs.get("rms_dbfs",0) - bs.get("rms_dbfs",0)) if (bs.get("rms_dbfs") is not None and cs.get("rms_dbfs") is not None) else None,
        }
    per_rec.append({"folder": folder, "speakers": spk_diffs})

w("## Speaker-level transition matrix (before -> after)")
w("```")
for (b_lv, c_lv), n in sorted(transitions.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
    w(f"  {str(b_lv):10s} -> {str(c_lv):10s}  : {n}")
w("```")
w()

def avg(a):
    return sum(a)/len(a) if a else 0.0

w("## Aggregate metric shifts (averages over matched speakers)")
n = len(peak_dbfs_before)
w(f"- matched speakers compared: {n}")
if n:
    w(f"- peak_dBFS:    {avg(peak_dbfs_before):+.2f} -> {avg(peak_dbfs_after):+.2f}  (Δ {avg(peak_dbfs_after)-avg(peak_dbfs_before):+.2f} dB)")
    w(f"- RMS_dBFS:     {avg(rms_dbfs_before):+.2f} -> {avg(rms_dbfs_after):+.2f}  (Δ {avg(rms_dbfs_after)-avg(rms_dbfs_before):+.2f} dB)")
    w(f"- bandwidth Hz: {avg(bw_before):.0f} -> {avg(bw_after):.0f}  (Δ {avg(bw_after)-avg(bw_before):+.0f} Hz)")
    w(f"- silent %:     {avg(silent_before):.2f} -> {avg(silent_after):.2f}  (Δ {avg(silent_after)-avg(silent_before):+.2f}%)")
w()

w(f"## Improvements: {len(improvements)} speakers")
for folder, sid, bl, cl in improvements:
    w(f"  - {folder} / {sid}: {bl} -> {cl}")
w()
w(f"## Regressions: {len(regressions)} speakers")
for folder, sid, bl, cl in regressions:
    w(f"  - {folder} / {sid}: {bl} -> {cl}")
w()
w(f"## Ghost speakers dropped: {len(ghost_drops)}")
for folder, sid, lvl, peak, rms in ghost_drops:
    w(f"  - {folder} / {sid}: was level={lvl}, peak={peak:.4f}, rms_dBFS={rms:.1f}")
w()
w(f"## New speakers in after-set: {len(new_in_after)}")
for folder, sid, lvl in new_in_after:
    w(f"  - {folder} / {sid}: level={lvl}")
w()

open(OUT_MD, "w").write("\n".join(lines))
print(f"wrote report: {OUT_MD}")
print()
print("\n".join(lines))
