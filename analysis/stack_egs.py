#!/usr/bin/env python
"""
Build per-example collages across runs.

Validation plots are saved per run as:
    runs/<run_id>/validation_plots/r<run_id>_e<NNNN>_eg<NN>.jpg

where <NNNN> is the (zero-padded) epoch and <NN> is the validation-example index.

Given a list of runs, for each example index (egNN) this picks the plot from the
*last* epoch present in each run, then stacks those per-run plots vertically into
a single image per example.
"""

import argparse
import os
import re
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

# r<run_id>_e<epoch>_eg<eg>.jpg
#   run_id may contain underscores / dots, so anchor on the _e<digits>_eg<digits> tail.
FNAME_RE = re.compile(r"^r(?P<run>.+)_e(?P<epoch>\d+)_eg(?P<eg>\d+)\.jpg$")


def find_plots(run_dir):
    """
    Return {eg: {epoch: path}} for a single run's validation_plots directory.
    """
    vp_dir = os.path.join(run_dir, "validation_plots")
    out = defaultdict(dict)
    if not os.path.isdir(vp_dir):
        return out
    for name in os.listdir(vp_dir):
        m = FNAME_RE.match(name)
        if not m:
            continue
        eg = int(m.group("eg"))
        epoch = int(m.group("epoch"))
        out[eg][epoch] = os.path.join(vp_dir, name)
    return out


def latest_per_eg(run_dir):
    """
    Return {eg: (epoch, path)} keeping only the highest epoch present per eg.
    """
    latest = {}
    for eg, by_epoch in find_plots(run_dir).items():
        best = max(by_epoch)
        latest[eg] = (best, by_epoch[best])
    return latest


def _label_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        return ImageFont.load_default()


def stack_vertically(entries, label=True, pad=8, bg=(255, 255, 255)):
    """
    entries: list of (caption, image_path). Stacks images vertically, each
    resized to the common (max) width, with an optional caption strip.
    """
    imgs = [(cap, Image.open(p).convert("RGB")) for cap, p in entries]
    width = max(im.width for _, im in imgs)

    font = _label_font() if label else None
    cap_h = 28 if label else 0

    # scale each image to the common width, preserving aspect ratio
    scaled = []
    for cap, im in imgs:
        if im.width != width:
            h = round(im.height * width / im.width)
            im = im.resize((width, h), Image.LANCZOS)
        scaled.append((cap, im))

    total_h = sum(im.height + cap_h + pad for _, im in scaled) + pad
    canvas = Image.new("RGB", (width + 2 * pad, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    y = pad
    for cap, im in scaled:
        if label:
            draw.text((pad, y), cap, fill=(0, 0, 0), font=font)
            y += cap_h
        canvas.paste(im, (pad, y))
        y += im.height + pad
    return canvas


def build_collages(runs, runs_root, out_dir, label=True):
    os.makedirs(out_dir, exist_ok=True)

    # run_id -> {eg: (epoch, path)}
    per_run = {}
    for run_id in runs:
        run_dir = os.path.join(runs_root, run_id)
        latest = latest_per_eg(run_dir)
        if not latest:
            print(f"warning: no validation plots found for run '{run_id}'")
        per_run[run_id] = latest

    # union of all eg indices across the requested runs
    all_egs = sorted({eg for latest in per_run.values() for eg in latest})
    if not all_egs:
        print("no examples found across the given runs; nothing to do")
        return

    for eg in all_egs:
        entries = []
        for run_id in runs:
            latest = per_run[run_id]
            if eg not in latest:
                print(f"warning: run '{run_id}' has no eg{eg:02d}; skipping")
                continue
            epoch, path = latest[eg]
            entries.append((f"{run_id}  (e{epoch:04d})", path))
        if not entries:
            continue

        collage = stack_vertically(entries, label=label)
        out_path = os.path.join(out_dir, f"eg{eg:02d}.jpg")
        collage.save(out_path, quality=95)
        print(f"wrote {out_path}  ({len(entries)} runs)")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help="run ids (directory names under --runs-root)",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="root directory containing the run directories (default: runs)",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/collages",
        help="output directory for the per-example collages",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="do not draw run_id/epoch captions above each panel",
    )
    return parser


def main():
    opts = build_parser().parse_args()
    build_collages(
        runs=opts.runs,
        runs_root=opts.runs_root,
        out_dir=opts.out_dir,
        label=not opts.no_label,
    )


if __name__ == "__main__":
    main()
