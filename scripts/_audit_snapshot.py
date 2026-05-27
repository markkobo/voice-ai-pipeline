#!/usr/bin/env python3
"""Snapshot voice_audit per-speaker for every recording.
Usage: python3 _audit_snapshot.py [output_path]
"""
import json, os, sys
from collections import Counter, defaultdict

ROOT = "/home/rding/voice-ai-pipeline/data/recordings/raw"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/rding/voice-ai-pipeline/scripts/_audit_snapshot.json"

result = {}
folders = sorted(os.listdir(ROOT))
print(f"Found {len(folders)} folders")

for folder in folders:
    meta_path = os.path.join(ROOT, folder, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  SKIP {folder}: no metadata.json")
        continue
    try:
        with open(meta_path) as f:
            d = json.load(f)
    except Exception as e:
        print(f"  SKIP {folder}: cannot parse json: {e}")
        continue
    rec_id = d.get("recording_id")
    segs = d.get("speaker_segments", []) or []
    status = d.get("status")
    error = d.get("error_message")
    duration = d.get("duration_seconds")
    by_spk = defaultdict(list)
    for s in segs:
        sid = s.get("speaker_id")
        if sid:
            by_spk[sid].append(s)
    speakers = {}
    for sid, slist in by_spk.items():
        levels = Counter()
        peaks, rms, bw, silent, peak_raw = [], [], [], [], []
        warnings_all = set()
        total_dur = 0.0
        audit_found = 0
        for s in slist:
            va = s.get("voice_audit") or {}
            if va:
                audit_found += 1
                lvl = va.get("level")
                if lvl: levels[lvl] += 1
                m = va.get("metrics") or {}
                if m.get("peak_dbfs") is not None: peaks.append(m["peak_dbfs"])
                if m.get("rms_dbfs") is not None: rms.append(m["rms_dbfs"])
                if m.get("effective_bandwidth_hz") is not None: bw.append(m["effective_bandwidth_hz"])
                if m.get("silent_pct") is not None: silent.append(m["silent_pct"])
                if m.get("peak") is not None: peak_raw.append(m["peak"])
                for w in (va.get("warnings") or []):
                    warnings_all.add(w)
            d2 = s.get("duration_seconds") or 0.0
            total_dur += d2
        spk_summary = {
            "n_segments": len(slist),
            "total_duration_s": round(total_dur, 2),
            "level": levels.most_common(1)[0][0] if levels else None,
            "audit_found_on_n_segs": audit_found,
        }
        if peaks: spk_summary["peak_dbfs"] = peaks[0]
        if rms: spk_summary["rms_dbfs"] = rms[0]
        if bw: spk_summary["effective_bandwidth_hz"] = bw[0]
        if silent: spk_summary["silent_pct"] = silent[0]
        if peak_raw: spk_summary["peak"] = peak_raw[0]
        spk_summary["warnings"] = sorted(warnings_all)
        speakers[sid] = spk_summary
    result[folder] = {
        "recording_id": rec_id,
        "folder_name": folder,
        "status": status,
        "error_message": error,
        "duration_seconds": duration,
        "n_speakers": len(speakers),
        "speakers": speakers,
    }

with open(OUT, "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\nSnapshot saved to {OUT}")
print(f"  {len(result)} recordings")
status_count = Counter()
for r in result.values():
    status_count[r["status"]] += 1
print(f"  status: {dict(status_count)}")
print(f"  total speakers: {sum(r["n_speakers"] for r in result.values())}")
lvl_count = Counter()
for r in result.values():
    for s in r["speakers"].values():
        lvl_count[s.get("level")] += 1
print(f"  speaker levels: {dict(lvl_count)}")
