#!/usr/bin/env python3
"""Re-run all recordings through the new audio pipeline.

Sequential — single GPU + cuda_lock means we cannot parallelize.
After each /process call, polls /api/recordings/{id} until status changes
out of "processing" (-> "processed" or "failed").

Logs progress to stdout AND to a log file under scripts/.
"""
import json, os, sys, time, urllib.request, urllib.error
from datetime import datetime

BASE = "http://localhost:8080"
ROOT = "/home/rding/voice-ai-pipeline/data/recordings/raw"
LOG = "/home/rding/voice-ai-pipeline/scripts/_reprocess.log"
PROGRESS_JSON = "/home/rding/voice-ai-pipeline/scripts/_reprocess_progress.json"

# Recording known to be corrupt — skip per user note
SKIP = {"child_xiao_s_20260520_020451_154536"}

POLL_INTERVAL = 5      # seconds
MAX_POLL_TIME = 1800   # 30 minutes per recording max

def http_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read().decode("utf-8")

def http_post(url):
    req = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read().decode("utf-8")

def log(msg, fh):
    line = f"[{datetime.now().strftime("%H:%M:%S")}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()

def load_recordings():
    items = []
    for folder in sorted(os.listdir(ROOT)):
        if folder in SKIP:
            continue
        meta = os.path.join(ROOT, folder, "metadata.json")
        if not os.path.exists(meta):
            continue
        try:
            with open(meta) as f:
                d = json.load(f)
        except Exception:
            continue
        rid = d.get("recording_id")
        if rid:
            items.append((folder, rid, d.get("duration_seconds")))
    return items

def get_status(rid):
    try:
        st, body = http_get(f"{BASE}/api/recordings/{rid}")
        if st != 200:
            return None, f"http {st}"
        d = json.loads(body)
        return d.get("status"), d.get("error_message")
    except Exception as e:
        return None, str(e)

def reprocess_one(folder, rid, dur, fh):
    log(f"--- {folder}  rid={rid[:8]}  dur={dur:.1f}s ---", fh)
    t0 = time.time()
    try:
        st, body = http_post(f"{BASE}/api/recordings/{rid}/process")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        log(f"  POST failed: HTTP {e.code} body={body[:300]}", fh)
        return "post_failed", time.time() - t0, body
    except Exception as e:
        log(f"  POST exception: {e}", fh)
        return "post_failed", time.time() - t0, str(e)
    if st != 200:
        log(f"  POST returned {st}: {body[:200]}", fh)
        return "post_failed", time.time() - t0, body
    log(f"  POST ok, polling...", fh)
    # Poll
    poll_start = time.time()
    last_status = None
    while time.time() - poll_start < MAX_POLL_TIME:
        time.sleep(POLL_INTERVAL)
        status, err = get_status(rid)
        if status != last_status:
            log(f"  status={status} err={err}", fh)
            last_status = status
        if status in ("processed", "failed", "error"):
            elapsed = time.time() - t0
            log(f"  -> done in {elapsed:.1f}s  status={status}  err={err}", fh)
            return status, elapsed, err
    log(f"  TIMEOUT after {MAX_POLL_TIME}s, last status={last_status}", fh)
    return "timeout", time.time() - t0, "timed out"

def main():
    items = load_recordings()
    fh = open(LOG, "w")
    log(f"Reprocessing {len(items)} recordings (skipping known-corrupt: {SKIP})", fh)
    log(f"Total audio duration: {sum(d for _,_,d in items if d):.0f} s", fh)
    results = []
    overall_start = time.time()
    for i, (folder, rid, dur) in enumerate(items, 1):
        log(f"[{i}/{len(items)}] starting...", fh)
        try:
            outcome, elapsed, err = reprocess_one(folder, rid, dur or 0.0, fh)
        except KeyboardInterrupt:
            log("INTERRUPTED", fh)
            break
        results.append({
            "folder": folder,
            "recording_id": rid,
            "duration_seconds": dur,
            "outcome": outcome,
            "elapsed_seconds": elapsed,
            "error_message": err,
        })
        # Save progress after each
        with open(PROGRESS_JSON, "w") as pf:
            json.dump({
                "started_at": overall_start,
                "elapsed_overall_s": time.time() - overall_start,
                "completed": i,
                "total": len(items),
                "results": results,
            }, pf, ensure_ascii=False, indent=2)
    total_elapsed = time.time() - overall_start
    log(f"\nALL DONE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", fh)
    ok = sum(1 for r in results if r["outcome"] == "processed")
    fail = sum(1 for r in results if r["outcome"] != "processed")
    log(f"  processed: {ok}", fh)
    log(f"  not-processed: {fail}", fh)
    fh.close()

if __name__ == "__main__":
    main()
