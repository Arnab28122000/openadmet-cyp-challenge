"""Read the CYP challenge leaderboard over HTTP and pull our own row out of every table.

The Space renders the whole leaderboard into the page, so there is no need to drive a browser
to read a score -- and reading it by eye off a screenshot is how a digit gets transcribed wrong.

Two traps this encodes, both of which cost time:

* The tables are Gradio components of type **`leaderboard`**, not `dataframe`, and their value is
  `{headers, data, metadata}`. A script that looks for `dataframe` finds nothing and reports an
  empty board.
* A table is only identifiable by walking `cfg["layout"]` and collecting the enclosing `tabitem`
  labels -- the component itself does not say which isoform or which track it belongs to.

    python scripts/cyp_board_fetch.py                     # our row on every board
    python scripts/cyp_board_fetch.py --wait-after "2026-10-09 17:09 UTC"
        # poll until our Overall-regression row is newer than that, then print it and exit 0

`--wait-after` is the one to use after submitting: the board shows only the latest row per user,
so this is what tells you the new score has landed and the old one is gone.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

SPACE = "https://openadmet-cyp-challenge.hf.space/"
OVERALL = "Direct Inhibition Prediction (Regression)"
US = "talindrew"


def fetch(url: str = SPACE, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (cyp-board-fetch)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _balanced(raw: str) -> str:
    """Slice the first balanced {...} out of raw, respecting strings and escapes."""
    depth, instr, esc = 0, False, False
    for i, ch in enumerate(raw):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[: i + 1]
    raise ValueError("unbalanced gradio_config")


def boards(html: str) -> dict[str, dict]:
    """{tab path -> {headers, data}} for every leaderboard component, named by its tabitems."""
    m = re.search(r"window\.gradio_config\s*=\s*(\{.*)", html, re.S)
    if not m:
        raise ValueError("no window.gradio_config in page")
    cfg = json.loads(_balanced(m.group(1)))
    by_id = {c["id"]: c for c in cfg["components"]}
    found: dict[str, dict] = {}

    def walk(node, path):
        comp = by_id.get(node.get("id"), {})
        props = comp.get("props", {})
        here = path + [str(props["label"])] if (
            comp.get("type") == "tabitem" and props.get("label")) else path
        if comp.get("type") == "leaderboard":
            val = props.get("value") or {}
            if val.get("data"):
                found[" / ".join(here)] = {"headers": val["headers"], "data": val["data"]}
        for ch in node.get("children") or []:
            walk(ch, here)

    walk(cfg["layout"], [])
    return found


def our_row(board: dict, who: str = US) -> dict | None:
    for r in board["data"]:
        if who in [str(x) for x in r]:
            return {k: re.sub(r"<[^>]+>", "", str(v)) for k, v in zip(board["headers"], r)}
    return None


def newest(board: dict) -> str:
    """The most recent submission timestamp anywhere on a board.

    This is the liveness probe for the scorer: if it is older than your own submission, the
    queue is stalled for everyone and your row is not lost. Checking it is what stops a pointless
    resubmission -- which would burn the next 12-hour window AND overwrite the pending result.
    """
    i = board["headers"].index("Submitted")
    stamps = sorted(str(r[i]) for r in board["data"] if str(r[i])[:2] == "20")
    return stamps[-1] if stamps else "?"


def show(bs: dict, who: str = US) -> dict:
    out = {}
    for tag, b in bs.items():
        row = our_row(b, who)
        if row is None:
            continue
        out[tag] = row
        skip = ("Rank", "Username", "Submitted", "Model Report Link")
        print(f"{tag}")
        print(f"    rank {row.get('Rank')} of {len(b['data'])}   submitted {row.get('Submitted')}")
        print("    " + "  ".join(f"{k}={v}" for k, v in row.items() if k not in skip))
    if not out:
        print(f"no '{who}' row on any board")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", default=US)
    ap.add_argument("--wait-after", default=None,
                    help="poll until the Overall regression row is newer than this timestamp")
    ap.add_argument("--every", type=int, default=180, help="seconds between polls")
    ap.add_argument("--max-wait", type=int, default=10800, help="give up after this many seconds")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    deadline = time.time() + args.max_wait
    while True:
        try:
            bs = boards(fetch())
        except Exception as e:  # a transient Space hiccup must not end a 3-hour wait
            print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] fetch failed: {e}", flush=True)
            if args.wait_after is None or time.time() >= deadline:
                return 2
            time.sleep(args.every)
            continue

        if args.wait_after is None:
            rows = show(bs, args.who)
            if args.json_out:
                open(args.json_out, "w").write(json.dumps(rows, indent=1))
            return 0

        key = next((k for k in bs if OVERALL in k and "Overall" in k and "Live" in k), None)
        if key is None:
            print("could not find the live Overall regression board", file=sys.stderr)
            return 2
        row = our_row(bs[key], args.who)
        stamp = (row or {}).get("Submitted")
        now = time.strftime("%H:%M:%SZ", time.gmtime())
        if row is not None and stamp != args.wait_after:
            print(f"\n[{now}] SCORED -- new row, submitted {stamp} (was {args.wait_after})\n",
                  flush=True)
            rows = show(bs, args.who)
            if args.json_out:
                open(args.json_out, "w").write(json.dumps(rows, indent=1))
            return 0
        # Report the newest row on the WHOLE board too. Without it, a stalled scorer and a lost
        # submission look identical from our row alone -- and on 2026-10-10 they were not: the
        # board's newest row was 2 h OLDER than our submission, so the queue was the problem.
        print(f"[{now}] still showing {stamp}; board newest {newest(bs[key])}; waiting",
              flush=True)
        if time.time() >= deadline:
            print("gave up waiting", flush=True)
            return 1
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
