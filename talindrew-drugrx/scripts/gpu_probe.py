"""Runs INSIDE a Colab session. Prints one JSON line describing the GPU's state.

Deliberately reports both torch's view and nvidia-smi's: torch only sees the
allocator of the *current* process, so a session whose training process has died
reports 0 GB allocated while nvidia-smi still shows a zombie holding the card.
Classifying on torch alone would call a wedged session idle and a healthy one
empty.
"""
import json, os, subprocess, time, glob

out = {}
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["gpu"] = p.name
        out["total_gb"] = round(p.total_memory / 1e9, 1)
    else:
        out["gpu"] = None
except Exception as e:
    out["gpu"] = f"error: {e}"

try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    used, total, util = [x.strip() for x in r.stdout.strip().split(",")]
    out["smi_used_gb"] = round(int(used) / 1024, 1)
    out["smi_total_gb"] = round(int(total) / 1024, 1)
    out["util_pct"] = int(util)
except Exception as e:
    out["smi_error"] = str(e)

# Our own training processes, with elapsed time so a hung one is visible.
try:
    r = subprocess.run(["ps", "-eo", "pid,etimes,pcpu,args"], capture_output=True, text=True, timeout=30)
    procs = []
    for ln in r.stdout.splitlines()[1:]:
        if "/content/scripts/" in ln and "grep" not in ln:
            bits = ln.split(None, 3)
            procs.append({"pid": int(bits[0]), "etimes": int(bits[1]),
                          "cpu": float(bits[2]), "cmd": bits[3][:110]})
    out["procs"] = procs
except Exception as e:
    out["procs_error"] = str(e)

# Freshness of anything the runs write, so "no process" can be told apart from
# "finished a minute ago".
newest, newest_f = 0.0, None
for pat in ("/content/out/**/*.log", "/content/out/**/*.json", "/content/out/*.log",
            "/content/ckpt/*.pt"):
    for f in glob.glob(pat, recursive=True):
        try:
            m = os.path.getmtime(f)
        except OSError:
            continue
        if m > newest:
            newest, newest_f = m, f
out["newest_artifact"] = newest_f
out["artifact_age_s"] = int(time.time() - newest) if newest else None
out["result_files"] = sorted(os.path.basename(f) for f in glob.glob("/content/out/*/*.json"))[:40]
print("GPUPROBE " + json.dumps(out))
