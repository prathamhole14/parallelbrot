"""Command line renderer: ``parallelbrot [options]``."""

from __future__ import annotations

import argparse
import sys
import time

from . import (
    COLOR_SCHEMES,
    __version__,
    available_backends,
    compiled_backends,
    device_name,
    has_openmp,
    render,
    save_png,
)


def _size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def _center(text: str) -> tuple[float, float]:
    try:
        real, imag = (float(part) for part in text.split(",", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected REAL,IMAG, got {text!r}") from None
    return real, imag


def _describe() -> str:
    lines = [
        f"parallelbrot {__version__}",
        f"  compiled backends : {', '.join(compiled_backends())}",
        f"  usable now        : {', '.join(available_backends())}",
        f"  OpenMP            : {'yes' if has_openmp() else 'no'}",
    ]
    for backend in compiled_backends():
        if backend != "cpu":
            lines.append(f"  {backend} device      : {device_name(backend) or 'none found'}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parallelbrot",
        description="Render the Mandelbrot set to a PNG.",
        epilog="With no arguments, writes a 1920x1080 view of the whole set "
               "to mandelbrot.png.",
    )
    parser.add_argument("-o", "--output", default="mandelbrot.png", metavar="PATH")
    parser.add_argument("-s", "--size", type=_size, default=(1920, 1080), metavar="WxH")
    parser.add_argument("-c", "--center", type=_center, default=(-0.5, 0.0), metavar="RE,IM")
    parser.add_argument("-z", "--zoom", type=float, default=1.0)
    parser.add_argument("-i", "--iterations", type=int, default=128)
    parser.add_argument("--colors", choices=COLOR_SCHEMES, default="fire")
    parser.add_argument("-b", "--backend", choices=("auto", "cpu", "opencl", "cuda"), default="auto")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress the summary line")
    parser.add_argument("--info", action="store_true", help="report available backends and exit")
    parser.add_argument("--version", action="version", version=f"parallelbrot {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.info:
        print(_describe())
        return 0

    width, height = args.size
    started = time.perf_counter()
    try:
        image = render(
            width, height,
            center=args.center,
            zoom=args.zoom,
            max_iterations=args.iterations,
            color_scheme=args.colors,
            backend=args.backend,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"parallelbrot: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    try:
        save_png(args.output, image)
    except OSError as exc:
        print(f"parallelbrot: cannot write {args.output}: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"{args.output}  {width}x{height}  {args.iterations} iterations  {elapsed * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
