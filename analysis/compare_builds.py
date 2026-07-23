#!/usr/bin/env python3
"""Compare the nextpnr "Device utilisation" section across N runs.

Usage:
    uv run compare_builds.py 085 086 087
    uv run compare_builds.py runs/085 runs/087

For each run R it reads runs/R/tiliqua_build/top.tim, parses the
"Device utilisation" block and prints the resources side by side.
"""

import argparse
import re
import sys
from pathlib import Path

# Info:             TRELLIS_IO:      64/    197    32%
_UTIL_RE = re.compile(
    r"^Info:\s*(?P<name>[\w]+):\s*(?P<used>\d+)/\s*(?P<total>\d+)\s*(?P<pct>\d+)%"
)

# Info: Max frequency for clock '$glbnet$clk': 67.47 MHz (PASS at 60.00 MHz)
_FMAX_RE = re.compile(
    r"^Info: Max frequency for clock\s*'(?P<clock>[^']+)':\s*"
    r"(?P<freq>[\d.]+) MHz\s*\((?P<verdict>\w+) at (?P<target>[\d.]+) MHz\)"
)


def resolve_tim(run: str) -> Path:
    """Map a run argument to its top.tim path."""
    p = Path(run)
    if p.name == "top.tim" and p.is_file():
        return p
    if (p / "tiliqua_build" / "top.tim").is_file():
        return p / "tiliqua_build" / "top.tim"
    # bare run id like "087" -> runs/087/tiliqua_build/top.tim
    cand = Path("runs") / run / "tiliqua_build" / "top.tim"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"could not find top.tim for run {run!r} (tried {cand})")


def parse_util(tim_path: Path) -> dict[str, tuple[int, int, int]]:
    """Return {resource: (used, total, pct)} from the first util block."""
    util: dict[str, tuple[int, int, int]] = {}
    in_block = False
    for line in tim_path.read_text().splitlines():
        if "Device utilisation:" in line:
            in_block = True
            continue
        if in_block:
            m = _UTIL_RE.match(line)
            if m:
                util[m.group("name")] = (
                    int(m.group("used")),
                    int(m.group("total")),
                    int(m.group("pct")),
                )
            elif util:
                # first non-matching line after we've collected rows -> block done
                break
    if not util:
        raise ValueError(f"no Device utilisation block found in {tim_path}")
    return util


def parse_fmax(tim_path: Path) -> dict[str, str]:
    """Return {clock: "freq MHz (VERDICT at target)"} for the LAST report.

    top.tim contains several "Max frequency" reports; the final pair (one per
    clock) reflects the fully routed design, so keep only the last entry seen
    for each clock.
    """
    fmax: dict[str, str] = {}
    for line in tim_path.read_text().splitlines():
        m = _FMAX_RE.match(line)
        if m:
            fmax[m.group("clock")] = (
                f"{m.group('freq')} MHz "
                f"({m.group('verdict')} at {m.group('target')} MHz)"
            )
    return fmax


def fmt_cell(entry: tuple[int, int, int] | None) -> str:
    if entry is None:
        return "-"
    used, total, pct = entry
    return f"{used}/{total} ({pct}%)"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run ids (e.g. 087) or paths")
    ap.add_argument(
        "--changed-only",
        action="store_true",
        help="only show resources whose used count differs across runs",
    )
    opts = ap.parse_args(argv)

    utils: list[tuple[str, dict[str, tuple[int, int, int]]]] = []
    fmaxes: list[tuple[str, dict[str, str]]] = []
    for run in opts.runs:
        try:
            tim = resolve_tim(run)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        utils.append((run, parse_util(tim)))
        fmaxes.append((run, parse_fmax(tim)))
    # union of resource names, preserving first-seen order
    names: list[str] = []
    for _, u in utils:
        for name in u:
            if name not in names:
                names.append(name)

    if opts.changed_only:

        def changed(name: str) -> bool:
            useds = {u.get(name, (None,))[0] for _, u in utils}
            return len(useds) > 1

        names = [n for n in names if changed(n)]

    columns = [run for run, _ in utils]
    rows = []
    for name in names:
        rows.append([name] + [fmt_cell(u.get(name)) for _, u in utils])

    # clock-name union across runs, preserving first-seen order
    clocks: list[str] = []
    for _, f in fmaxes:
        for clk in f:
            if clk not in clocks:
                clocks.append(clk)
    for clk in clocks:
        rows.append([clk] + [f.get(clk, "-") for _, f in fmaxes])

    header = ["resource"] + columns
    widths = [
        max(len(header[c]), *(len(r[c]) for r in rows)) if rows else len(header[c])
        for c in range(len(header))
    ]

    def print_row(cells: list[str]) -> None:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)))

    print_row(header)
    print_row(["-" * w for w in widths])
    for r in rows:
        print_row(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
