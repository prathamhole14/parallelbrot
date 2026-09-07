"""Tests for the command line interface."""

import struct
import zlib

import numpy as np
import pytest

import parallelbrot as pb
from parallelbrot.cli import main, write_png


def read_png(path):
    """Minimal PNG reader, so the tests do not depend on Pillow."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, pos = {}, 8
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        chunks.setdefault(tag, b"")
        chunks[tag] += data[pos + 8:pos + 8 + length]
        pos += 12 + length
    width, height, depth, color = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    raw = zlib.decompress(chunks[b"IDAT"])
    stride = width * 4 + 1
    rows = [raw[i * stride + 1:(i + 1) * stride] for i in range(height)]
    assert all(raw[i * stride] == 0 for i in range(height)), "expected filter type 0"
    return width, height, depth, color, np.frombuffer(b"".join(rows), np.uint8).reshape(height, width, 4)


def test_default_run_writes_a_png(tmp_path, capsys):
    out = tmp_path / "out.png"
    assert main(["-o", str(out), "-s", "64x48"]) == 0
    width, height, depth, color, pixels = read_png(out)
    assert (width, height, depth, color) == (64, 48, 8, 6)  # 8-bit RGBA
    assert pixels.shape == (48, 64, 4)
    assert (pixels[..., 3] == 255).all()
    assert str(out) in capsys.readouterr().out


def test_png_matches_the_rendered_array(tmp_path):
    image = pb.render(32, 24, zoom=10.0, color_scheme="ocean")
    out = tmp_path / "rt.png"
    write_png(str(out), image)
    _, _, _, _, pixels = read_png(out)
    assert np.array_equal(pixels, (np.clip(image, 0, 1) * 255.0 + 0.5).astype(np.uint8))


def test_options_are_applied(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    main(["-o", str(a), "-s", "32x32", "-z", "1", "--colors", "fire", "-q"])
    main(["-o", str(b), "-s", "32x32", "-z", "500", "--colors", "ocean", "-q"])
    assert a.read_bytes() != b.read_bytes()


def test_info_reports_backends(capsys):
    assert main(["--info"]) == 0
    out = capsys.readouterr().out
    assert "compiled backends" in out
    for backend in pb.compiled_backends():
        assert backend in out


def test_quiet_prints_nothing(tmp_path, capsys):
    main(["-o", str(tmp_path / "q.png"), "-s", "16x16", "-q"])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("argv", [
    ["-s", "0x10"],
    ["-s", "banana"],
    ["-c", "nope"],
    ["--colors", "chartreuse"],
    ["-b", "vulkan"],
])
def test_bad_arguments_exit_nonzero(argv, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["-o", str(tmp_path / "x.png"), *argv])
    assert exc.value.code != 0


def test_runtime_errors_are_reported(tmp_path, capsys):
    # zoom must be positive; the extension rejects it rather than argparse.
    assert main(["-o", str(tmp_path / "x.png"), "-s", "8x8", "-z", "0"]) == 1
    assert "parallelbrot:" in capsys.readouterr().err


def test_unwritable_output_is_reported(capsys):
    assert main(["-o", "/nonexistent-dir/x.png", "-s", "8x8"]) == 1
    assert "cannot write" in capsys.readouterr().err
