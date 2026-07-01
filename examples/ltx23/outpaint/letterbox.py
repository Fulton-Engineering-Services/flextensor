# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Letterbox a source video onto a larger canvas with pure-black bars for LTX 2.3 outpaint.

The Outpaint IC-LoRA fills pure-black (RGB 0,0,0) regions, so this helper pads the source
to the target canvas without rescaling the original content: the source frames are placed
verbatim and the extension regions are filled with exact black. The output of this script
is what you pass to ``serve_infer.py`` as ``--conditioning-video`` / ``conditioning_video``.

Two ways to specify the canvas:

- Margins: --left/--right/--top/--bottom black pixels to add on each side.
- Canvas:  --width/--height target canvas; the source is centered unless --x/--y are given.

The dark-scene gamma round-trip is handled by ``serve_infer.py`` (``--gamma``), not here.
"""

from __future__ import annotations

import argparse
import logging
import subprocess  # noqa: S404 - args are constructed locally from CLI input, not untrusted data.

LOGGER = logging.getLogger(__name__)

MULTIPLE = 64


def probe_dimensions(path: str) -> tuple[int, int]:
    output = subprocess.check_output(  # noqa: S603
        [  # noqa: S607
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            path,
        ],
        text=True,
    )
    width, height = (int(part) for part in output.strip().split(",")[:2])
    return width, height


def compute_canvas(
    src_w: int,
    src_h: int,
    args: argparse.Namespace,
) -> tuple[int, int, int, int]:
    """Return (canvas_w, canvas_h, x, y) for placing the source on the canvas."""
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            raise SystemExit("Provide both --width and --height for canvas mode.")
        canvas_w, canvas_h = args.width, args.height
        if canvas_w < src_w or canvas_h < src_h:
            raise SystemExit(
                f"Canvas {canvas_w}x{canvas_h} is smaller than source {src_w}x{src_h}; "
                "outpaint only extends the canvas."
            )
        x = args.x if args.x is not None else (canvas_w - src_w) // 2
        y = args.y if args.y is not None else (canvas_h - src_h) // 2
        if x < 0 or y < 0 or x + src_w > canvas_w or y + src_h > canvas_h:
            raise SystemExit("Source placement falls outside the canvas; check --x/--y.")
        return canvas_w, canvas_h, x, y

    canvas_w = src_w + args.left + args.right
    canvas_h = src_h + args.top + args.bottom
    return canvas_w, canvas_h, args.left, args.top


def validate_multiple_of_64(canvas_w: int, canvas_h: int, *, enforce: bool) -> None:
    bad = [name for name, value in (("width", canvas_w), ("height", canvas_h)) if value % MULTIPLE != 0]
    if not bad:
        return
    message = (
        f"Canvas {canvas_w}x{canvas_h} is not a multiple of {MULTIPLE} ({', '.join(bad)}). "
        f"LTX 2.3 two-stage requires dimensions divisible by {MULTIPLE} (e.g. 1280x704 for 720p)."
    )
    if enforce:
        raise SystemExit(message)
    LOGGER.warning("%s", message)


def letterbox(src_path: str, dst_path: str, canvas_w: int, canvas_h: int, x: int, y: int) -> None:
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            src_path,
            "-vf",
            f"pad=width={canvas_w}:height={canvas_h}:x={x}:y={y}:color=black",
            # Lossless so the padded bars stay pure black and the source content is unchanged.
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            "-c:a",
            "copy",
            dst_path,
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Source video path.")
    parser.add_argument("--output", required=True, help="Letterboxed output video path.")
    # Margin mode (black pixels to add per side).
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=0)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--bottom", type=int, default=0)
    # Canvas mode (explicit target canvas; source centered unless --x/--y given).
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help=f"Warn instead of failing when the canvas is not a multiple of {MULTIPLE}.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    src_w, src_h = probe_dimensions(args.input)
    canvas_w, canvas_h, x, y = compute_canvas(src_w, src_h, args)
    validate_multiple_of_64(canvas_w, canvas_h, enforce=not args.no_validate)
    letterbox(args.input, args.output, canvas_w, canvas_h, x, y)
    LOGGER.info(
        "Letterboxed %sx%s -> %sx%s (source at x=%s, y=%s); wrote %s",
        src_w,
        src_h,
        canvas_w,
        canvas_h,
        x,
        y,
        args.output,
    )


if __name__ == "__main__":
    main()
