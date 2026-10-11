"""Read the whole CYP challenge leaderboard over HTTP, with no browser.

A Gradio Space serialises its initial state into `window.gradio_config` inside the page
HTML, and each component's `props.value.data` holds **all** its rows -- not the dozen the
virtualised DOM renders. One ~800 KB fetch therefore gives every leaderboard the Space has:
Live and Interim, Regression and TDI and Structure, macro and per isoform.

This defeats, in one call, every trap in the browser route for *reading*: the 30-60 s lazy
render, the virtualisation, the search-box-needs-Return dance, and a CDP `Runtime.evaluate`
that timed out outright. It does **not** help with writing -- submitting still needs a real
browser with real clicks, because Gradio's Svelte handlers ignore synthetic DOM events.

The one trap that survives: a component id alone does not say which **tab** it is under, and
reading Interim when you wanted Live once produced a reported 0.9628 against a Live 0.8663.
So `tab_paths()` walks `cfg["layout"]` recursively and collects the `tabitem` labels down
each branch, which pins every component to its tab.

    python scripts/cyp_read_board.py --who talindrew
    python scripts/cyp_read_board.py --who talindrew --dump /tmp/board
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SPACE = "https://openadmet-cyp-challenge.hf.space/"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


def fetch_config(url: str = SPACE, cache: Path | None = None) -> dict:
    """Pull the page and brace-match the JSON after `window.gradio_config =`.

    The config is one very long line, so it is streamed to a file rather than through a
    shell variable; some Spaces also reject curl's default User-Agent outright.
    """
    html = subprocess.run(["curl", "-sL", "-A", UA, url],
                          capture_output=True, text=True, check=True).stdout
    i = html.find("window.gradio_config")
    if i < 0:
        raise SystemExit("no window.gradio_config in the page -- did the Space change?")
    i = html.index("{", i)
    depth, j, instr, esc = 0, i, False, False
    while j < len(html):
        c = html[j]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        elif c == '"':
            instr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    cfg = json.loads(html[i:j + 1])
    if cache:
        cache.write_text(json.dumps(cfg))
    return cfg


def tab_paths(cfg: dict) -> dict[int, list[str]]:
    """Map component id -> the list of enclosing tab labels, outermost first.

    This is the part that stops Live and Interim being confused: the component ids carry no
    hint of which tab they sit under, but the layout tree does.
    """
    byid = {c["id"]: c for c in cfg["components"]}
    out: dict[int, list[str]] = {}

    def walk(node: dict, path: list[str]) -> None:
        cid = node.get("id")
        comp = byid.get(cid, {})
        here = path
        if comp.get("type") == "tabitem":
            here = path + [str(comp.get("props", {}).get("label", "?"))]
        if cid is not None:
            out[cid] = here
        for ch in node.get("children", []) or []:
            walk(ch, here)

    walk(cfg["layout"], [])
    return out


def tables(cfg: dict) -> list[dict]:
    """Every dataframe-ish component that actually carries rows."""
    paths = tab_paths(cfg)
    found = []
    for c in cfg["components"]:
        v = (c.get("props") or {}).get("value")
        if isinstance(v, dict) and isinstance(v.get("data"), list) and v["data"]:
            found.append({
                "id": c["id"],
                "tabs": paths.get(c["id"], []),
                "headers": v.get("headers") or [],
                "rows": v["data"],
            })
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", default="talindrew", help="entry name to highlight")
    ap.add_argument("--dump", default=None, help="directory to write every table as CSV")
    ap.add_argument("--url", default=SPACE)
    args = ap.parse_args()

    cfg = fetch_config(args.url)
    ts = tables(cfg)
    print(f"[board] {len(ts)} tables with rows\n")

    dump = Path(args.dump) if args.dump else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    for t in ts:
        tab = " / ".join(t["tabs"]) or "(no tab)"
        print(f"--- id {t['id']:>4}  [{tab}]  {len(t['rows'])} rows")
        print(f"    cols: {', '.join(map(str, t['headers']))}")
        if dump:
            import csv
            p = dump / f"lb_{t['id']}.csv"
            with p.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(t["headers"])
                w.writerows(t["rows"])

        hits = [r for r in t["rows"]
                if any(args.who.lower() in str(x).lower() for x in r)]
        for r in hits:
            print("    >> " + " | ".join(
                f"{h}={v}" for h, v in zip(t["headers"], r)))
        print()

    if dump:
        print(f"[board] wrote {len(ts)} CSVs to {dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
