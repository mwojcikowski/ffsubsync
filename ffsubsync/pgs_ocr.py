# -*- coding: utf-8 -*-
"""
PGS subtitle OCR pipeline.

Workflow:
  1. Extract a PGS (Presentation Graphics Stream) track from a video to a .sup
     binary (optionally kept on disk via --pgs-ocr-dump-sup).
  2. Parse the .sup binary into display sets – each carrying a PTS timestamp,
     palette, and RLE-encoded bitmap object(s).
  3. Render each non-clear display set to a PIL RGBA image (optionally saved as
     numbered PNGs via --pgs-ocr-dump-pngs).
  4. OCR the image with ocrmac (macOS Vision framework).
  5. Write an SRT file whose timecodes come directly from the PGS stream.

Requirements:
    pip install ocrmac Pillow
"""

import logging
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Iterator, List, Optional, Tuple

import srt

from ffsubsync.ffmpeg_utils import ffmpeg_bin_path, subprocess_args
from ffsubsync.speech_transformers import find_pgs_stream

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PGS binary format constants
# ---------------------------------------------------------------------------

_TICK = 90_000  # PTS ticks per second
_MAGIC = b"PG"  # Every PGS packet starts with these two bytes

_SEG_PDS = 0x14  # Palette Definition Segment
_SEG_ODS = 0x15  # Object Definition Segment
_SEG_PCS = 0x16  # Presentation Composition Segment
_SEG_END = 0x80  # End of Display Set

_COMP_EPOCH_START = 0x80


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _PCSObject:
    object_id: int
    window_id: int
    x: int
    y: int
    forced: bool = False


@dataclass
class _ODS:
    object_id: int
    width: int
    height: int
    rle_data: bytes


@dataclass
class _DisplaySet:
    pts: float
    screen_w: int
    screen_h: int
    is_clear: bool
    comp_state: int = 0x00
    pcs_objects: List[_PCSObject] = field(default_factory=list)
    palette: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    objects: Dict[int, _ODS] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Colour conversion
# ---------------------------------------------------------------------------


def _ycbcr_to_rgba(y: int, cb: int, cr: int, a: int) -> Tuple[int, int, int, int]:
    """BT.601 YCbCr + alpha → RGBA (all values 0-255)."""
    r = max(0, min(255, round(y + 1.402 * (cr - 128))))
    g = max(0, min(255, round(y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128))))
    b = max(0, min(255, round(y + 1.772 * (cb - 128))))
    return r, g, b, a


# ---------------------------------------------------------------------------
# Segment parsers
# ---------------------------------------------------------------------------


def _parse_pds(data: bytes) -> Dict[int, Tuple[int, int, int, int]]:
    """Palette Definition Segment → ``{entry_id: (R, G, B, A)}``."""
    palette: Dict[int, Tuple[int, int, int, int]] = {}
    pos = 2  # skip palette_id(1) + version(1)
    while pos + 4 < len(data):
        eid = data[pos]
        palette[eid] = _ycbcr_to_rgba(
            data[pos + 1], data[pos + 2], data[pos + 3], data[pos + 4]
        )
        pos += 5
    return palette


def _parse_pcs(data: bytes) -> Tuple[int, int, int, bool, List[_PCSObject]]:
    """Return ``(screen_w, screen_h, comp_state, is_clear, pcs_objects)``."""
    if len(data) < 11:
        return 0, 0, 0x00, True, []
    w, h = struct.unpack_from(">HH", data, 0)
    comp_state = data[7]
    n_objects = data[10]
    objects: List[_PCSObject] = []
    pos = 11
    for _ in range(n_objects):
        if pos + 8 > len(data):
            break
        obj_id = struct.unpack_from(">H", data, pos)[0]
        win_id = data[pos + 2]
        flags = data[pos + 3]
        forced = bool(flags & 0x80)
        x = struct.unpack_from(">H", data, pos + 4)[0]
        y = struct.unpack_from(">H", data, pos + 6)[0]
        objects.append(_PCSObject(obj_id, win_id, x, y, forced))
        pos += 8
        if flags & 0x40:  # object_cropped_flag → 8 extra bytes
            pos += 8
    return w, h, comp_state, n_objects == 0, objects


def _accumulate_ods(data: bytes, buf: Dict[int, dict]) -> None:
    """Accumulate one ODS packet into *buf* (handles multi-packet objects)."""
    if len(data) < 4:
        return
    obj_id = struct.unpack_from(">H", data, 0)[0]
    seq_flag = data[3]
    is_first = bool(seq_flag & 0x80)
    is_last = bool(seq_flag & 0x40)
    body = data[4:]
    if is_first:
        if len(body) < 7:
            return
        w = struct.unpack_from(">H", body, 3)[0]
        h = struct.unpack_from(">H", body, 5)[0]
        buf[obj_id] = {
            "width": w,
            "height": h,
            "rle": bytearray(body[7:]),
            "done": is_last,
        }
    elif obj_id in buf:
        buf[obj_id]["rle"].extend(body)
        if is_last:
            buf[obj_id]["done"] = True


# ---------------------------------------------------------------------------
# RLE decoder
# ---------------------------------------------------------------------------


def _decode_rle(rle: bytes, width: int, height: int) -> bytes:
    """Decode PGS run-length encoding → flat bytes of palette indices.

    After a 0x00 escape, the second byte b1 encodes:
      b1 & 0xC0 == 0x00, b1 = 0  -> end of line
      b1 & 0xC0 == 0x00, b1 > 0  -> b1 transparent pixels
      b1 & 0xC0 == 0x40           -> long transparent: (b1&0x3F)<<8 | b2
      b1 & 0xC0 == 0x80           -> short colour run: (b1&0x3F) px of colour b2
      b1 & 0xC0 == 0xC0           -> long colour run: (b1&0x3F)<<8|b2 px of b3
    """
    total = width * height
    pixels = bytearray(total)
    pos = idx = 0
    n = len(rle)
    while pos < n and idx < total:
        b0 = rle[pos]
        pos += 1
        if b0:
            pixels[idx] = b0
            idx += 1
            continue
        if pos >= n:
            break
        b1 = rle[pos]
        pos += 1
        hi = b1 & 0xC0
        if hi == 0x00:
            if b1 == 0:
                if width:
                    idx = ((idx + width - 1) // width) * width
            else:
                idx = min(idx + b1, total)
        elif hi == 0x40:
            if pos >= n:
                break
            b2 = rle[pos]
            pos += 1
            idx = min(idx + (((b1 & 0x3F) << 8) | b2), total)
        elif hi == 0x80:
            run = b1 & 0x3F
            if pos >= n:
                break
            colour = rle[pos]
            pos += 1
            end = min(idx + run, total)
            pixels[idx:end] = bytes([colour]) * (end - idx)
            idx = end
        else:  # 0xC0
            if pos + 1 >= n:
                break
            b2 = rle[pos]
            pos += 1
            run = ((b1 & 0x3F) << 8) | b2
            colour = rle[pos]
            pos += 1
            end = min(idx + run, total)
            pixels[idx:end] = bytes([colour]) * (end - idx)
            idx = end
    return bytes(pixels)


# ---------------------------------------------------------------------------
# SUP file iterator
# ---------------------------------------------------------------------------


def iter_pgs_display_sets(sup_path: str) -> Iterator[_DisplaySet]:
    """Parse a ``.sup`` file and yield :class:`_DisplaySet` objects in order.

    Palettes are inherited within an epoch and reset on *Epoch Start*.
    """
    current_ds: Optional[_DisplaySet] = None
    ods_buf: Dict[int, dict] = {}
    epoch_palette: Dict[int, Tuple[int, int, int, int]] = {}

    with open(sup_path, "rb") as f:
        while True:
            magic = f.read(2)
            if len(magic) < 2:
                break
            if magic != _MAGIC:
                logger.warning("unexpected magic bytes %r; stopping parse", magic)
                break
            raw_header = f.read(9)
            if len(raw_header) < 9:
                break
            pts_raw, _dts_raw, seg_type = struct.unpack(">IIB", raw_header)
            pts = pts_raw / _TICK
            raw_len = f.read(2)
            if len(raw_len) < 2:
                break
            seg_len = struct.unpack(">H", raw_len)[0]
            data = f.read(seg_len)

            if seg_type == _SEG_PCS:
                if current_ds is not None:
                    yield current_ds
                w, h, comp_state, is_clear, pcs_objects = _parse_pcs(data)
                if comp_state == _COMP_EPOCH_START:
                    epoch_palette = {}
                current_ds = _DisplaySet(
                    pts=pts,
                    screen_w=w,
                    screen_h=h,
                    is_clear=is_clear,
                    comp_state=comp_state,
                    pcs_objects=pcs_objects,
                    palette=dict(epoch_palette),
                )
                ods_buf = {}
            elif seg_type == _SEG_PDS and current_ds is not None:
                new_entries = _parse_pds(data)
                current_ds.palette.update(new_entries)
                epoch_palette.update(new_entries)
            elif seg_type == _SEG_ODS and current_ds is not None:
                _accumulate_ods(data, ods_buf)
            elif seg_type == _SEG_END and current_ds is not None:
                for obj_id, obj_data in ods_buf.items():
                    current_ds.objects[obj_id] = _ODS(
                        object_id=obj_id,
                        width=obj_data["width"],
                        height=obj_data["height"],
                        rle_data=bytes(obj_data["rle"]),
                    )
                ods_buf = {}

    if current_ds is not None:
        yield current_ds


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------


def render_display_set(ds: _DisplaySet):
    """Render a :class:`_DisplaySet` to a PIL RGBA image.

    Returns ``None`` for clear / empty display sets.
    """
    try:
        from PIL import Image as PilImage
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for PGS OCR. Install it with: pip install Pillow"
        ) from exc

    if ds.is_clear or not ds.pcs_objects:
        return None

    canvas = PilImage.new("RGBA", (ds.screen_w, ds.screen_h), (0, 0, 0, 0))
    rendered_any = False
    for pcs_obj in ds.pcs_objects:
        ods = ds.objects.get(pcs_obj.object_id)
        if ods is None or ods.width == 0 or ods.height == 0:
            continue
        raw = _decode_rle(ods.rle_data, ods.width, ods.height)
        obj_img = PilImage.new("RGBA", (ods.width, ods.height))
        pix = obj_img.load()
        for row in range(ods.height):
            for col in range(ods.width):
                pix[col, row] = ds.palette.get(raw[row * ods.width + col], (0, 0, 0, 0))
        canvas.paste(obj_img, (pcs_obj.x, pcs_obj.y), obj_img)
        rendered_any = True
    return canvas if rendered_any else None


# ---------------------------------------------------------------------------
# OCR via ocrmac (macOS Vision framework)
# ---------------------------------------------------------------------------


def ocr_image(image, language: str = "pl-PL") -> str:
    """OCR *image* (PIL Image) with ocrmac.  Returns recognised text."""
    try:
        from ocrmac import ocrmac as _ocrmac
    except ImportError as exc:
        raise ImportError(
            "ocrmac is required for PGS OCR (macOS only). "
            "Install it with: pip install ocrmac"
        ) from exc
    annotations = _ocrmac.OCR(
        image,
        language_preference=[language, "en-US"],
        recognition_level="accurate",
    ).recognize()
    if not annotations:
        return ""
    return "\n".join(item[0] for item in annotations)


# ---------------------------------------------------------------------------
# Extract PGS stream → .sup via ffmpeg
# ---------------------------------------------------------------------------


def extract_sup_stream(
    fname: str,
    stream: str,
    output_sup: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> None:
    """Extract a PGS subtitle stream from *fname* to a raw ``.sup`` file."""
    ffmpeg_cmd = ffmpeg_bin_path("ffmpeg", gui_mode, ffmpeg_resources_path=ffmpeg_path)
    if not stream.startswith("0:"):
        stream = "0:" + stream
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-nostdin",
        "-loglevel",
        "warning",
        "-i",
        fname,
        "-map",
        stream,
        "-c",
        "copy",
        output_sup,
    ]
    logger.info("extracting PGS stream: %s", " ".join(cmd))
    ret = subprocess.call(cmd, **subprocess_args(include_stdout=False))
    if ret != 0:
        raise RuntimeError("ffmpeg failed to extract PGS stream (exit code %d)" % ret)
    logger.info("wrote %s", output_sup)


# ---------------------------------------------------------------------------
# Full pipeline: video/sup → OCR → SRT
# ---------------------------------------------------------------------------


def pgs_to_srt(
    fname: str,
    stream: Optional[str],
    output_srt: str,
    language: str = "pl-PL",
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
    dump_sup: Optional[str] = None,
    dump_pngs_dir: Optional[str] = None,
) -> int:
    """Extract a PGS track from *fname*, OCR each frame, write SRT.

    Parameters
    ----------
    fname:
        Source video (MKV etc.) containing a PGS track.
    stream:
        ffmpeg stream specifier (e.g. ``"s:0"`` or ``"0:s:2"``).
        ``None`` auto-detects the first ``hdmv_pgs_subtitle`` track.
    output_srt:
        Destination ``.srt`` path.
    language:
        BCP-47 tag for ocrmac (e.g. ``"pl-PL"``, ``"en-US"``).
    ffmpeg_path:
        Directory or binary path for ffmpeg/ffprobe. ``None`` uses PATH.
    gui_mode:
        Suppress progress output.
    dump_sup:
        If given, keep the extracted ``.sup`` at this path.
    dump_pngs_dir:
        If given, save each rendered subtitle frame as ``subtitle_NNNNN.png``
        in this directory (created if it doesn't exist).

    Returns
    -------
    Number of SRT entries written.
    """
    if stream is None:
        stream = find_pgs_stream(fname, ffmpeg_path, gui_mode)
        if stream is None:
            raise ValueError("No hdmv_pgs_subtitle stream found in %s" % fname)
    if not stream.startswith("0:"):
        stream = "0:" + stream

    own_sup = dump_sup is None
    sup_path = dump_sup if dump_sup is not None else tempfile.mktemp(suffix=".sup")
    try:
        extract_sup_stream(fname, stream, sup_path, ffmpeg_path, gui_mode)
        return _sup_to_srt(sup_path, output_srt, language, dump_pngs_dir)
    finally:
        if own_sup and os.path.exists(sup_path):
            os.remove(sup_path)


def sup_to_srt(
    sup_path: str,
    output_srt: str,
    language: str = "pl-PL",
    dump_pngs_dir: Optional[str] = None,
) -> int:
    """OCR a pre-existing ``.sup`` file and write an SRT."""
    return _sup_to_srt(sup_path, output_srt, language, dump_pngs_dir)


def _ocr_sup_to_timings(
    sup_path: str,
    language: str,
    dump_pngs_dir: Optional[str],
) -> List[Tuple[float, float, str]]:
    """Shared OCR loop: parse .sup, render, OCR, return (start_s, end_s, text) list."""
    import tqdm

    if dump_pngs_dir:
        os.makedirs(dump_pngs_dir, exist_ok=True)

    display_sets = list(iter_pgs_display_sets(sup_path))
    logger.info("parsed %d display sets from %s", len(display_sets), sup_path)

    results: List[Tuple[float, float, str]] = []
    pending: Optional[_DisplaySet] = None
    frame_idx = 0

    def _process(ds_show: _DisplaySet, end_pts: float) -> None:
        nonlocal frame_idx
        img = render_display_set(ds_show)
        if img is None:
            return
        frame_idx += 1
        if dump_pngs_dir:
            png_path = os.path.join(
                dump_pngs_dir, "subtitle_{:05d}.png".format(frame_idx)
            )
            img.save(png_path)
            logger.debug("saved %s", png_path)
        text = ocr_image(img, language).strip()
        if text:
            results.append((ds_show.pts, end_pts, text))

    with tqdm.tqdm(
        total=len(display_sets),
        desc="OCR PGS",
        unit="frame",
        dynamic_ncols=True,
    ) as pbar:
        for ds in display_sets:
            pbar.update(1)
            if ds.is_clear:
                if pending is not None:
                    _process(pending, ds.pts)
                    pending = None
                    pbar.set_postfix(subtitles=len(results))
            else:
                if pending is not None:
                    # Back-to-back non-clear sets: previous ends at the new PTS
                    _process(pending, ds.pts)
                    pbar.set_postfix(subtitles=len(results))
                pending = ds

        # Trailing non-clear set with no following clear event
        if pending is not None:
            _process(pending, pending.pts + 3.0)
            pbar.set_postfix(subtitles=len(results))

    logger.info("OCR produced %d subtitle entries", len(results))
    return results


def _sup_to_srt(
    sup_path: str,
    output_srt: str,
    language: str,
    dump_pngs_dir: Optional[str],
) -> int:
    """Iterate display sets, render, OCR, write SRT.  Returns entry count."""
    timings = _ocr_sup_to_timings(sup_path, language, dump_pngs_dir)
    entries = [
        srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=start_s),
            end=timedelta(seconds=end_s),
            content=text,
        )
        for i, (start_s, end_s, text) in enumerate(timings)
    ]
    with open(output_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(entries))
    logger.info("wrote SRT -> %s", output_srt)
    return len(entries)


def _sup_to_generic_subs(sup_path: str, language: str, dump_pngs_dir: Optional[str]):
    """OCR a .sup file and return the results as a list of GenericSubtitle.

    The timestamps on the returned subtitles come directly from the PGS stream.
    This is the building block for the combined PGS-OCR + cross-lingual-align
    pipeline that avoids writing an intermediate SRT file.
    """
    from ffsubsync.generic_subtitles import GenericSubtitle

    timings = _ocr_sup_to_timings(sup_path, language, dump_pngs_dir)
    subs = []
    for i, (start_s, end_s, text) in enumerate(timings):
        start = timedelta(seconds=start_s)
        end = timedelta(seconds=end_s)
        inner = srt.Subtitle(index=i + 1, start=start, end=end, content=text)
        subs.append(GenericSubtitle(start, end, inner))
    return subs


def pgs_to_generic_subs(
    fname: str,
    stream: Optional[str],
    language: str = "pl-PL",
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
    dump_sup: Optional[str] = None,
    dump_pngs_dir: Optional[str] = None,
):
    """Extract PGS from *fname*, OCR, return list of GenericSubtitle.

    Equivalent to :func:`pgs_to_srt` but returns in-memory subtitles instead
    of writing a file.  Use this when you want to feed the result directly into
    :func:`ffsubsync.cross_lingual_aligner.align_subtitles_by_content`.
    """
    if stream is None:
        stream = find_pgs_stream(fname, ffmpeg_path, gui_mode)
        if stream is None:
            raise ValueError("No hdmv_pgs_subtitle stream found in %s" % fname)
    if not stream.startswith("0:"):
        stream = "0:" + stream

    own_sup = dump_sup is None
    sup_path = dump_sup if dump_sup is not None else tempfile.mktemp(suffix=".sup")
    try:
        extract_sup_stream(fname, stream, sup_path, ffmpeg_path, gui_mode)
        return _sup_to_generic_subs(sup_path, language, dump_pngs_dir)
    finally:
        if own_sup and os.path.exists(sup_path):
            os.remove(sup_path)


def sup_to_generic_subs(
    sup_path: str,
    language: str = "pl-PL",
    dump_pngs_dir: Optional[str] = None,
):
    """OCR a pre-existing ``.sup`` file and return a list of GenericSubtitle."""
    return _sup_to_generic_subs(sup_path, language, dump_pngs_dir)


# ---------------------------------------------------------------------------
# Convenience: OCR pre-extracted PNGs with external timings
# ---------------------------------------------------------------------------


def ocr_pngs_to_srt(
    pngs: List[str],
    timings: List[Tuple[float, float]],
    output_srt: str,
    language: str = "pl-PL",
) -> int:
    """OCR a list of PNG files with paired ``(start_s, end_s)`` timings → SRT.

    *timings* can come from PGS metadata extracted via
    :func:`ffsubsync.speech_transformers._get_pgs_timings_via_ffprobe`.

    Parameters
    ----------
    pngs:
        Ordered list of PNG file paths (one per subtitle event).
    timings:
        Matching list of ``(start_seconds, end_seconds)`` tuples.
    output_srt:
        Output ``.srt`` path.
    language:
        BCP-47 OCR language tag.

    Returns
    -------
    Number of SRT entries written.
    """
    try:
        from PIL import Image as PilImage
    except ImportError as exc:
        raise ImportError("Pillow is required: pip install Pillow") from exc

    if len(pngs) != len(timings):
        raise ValueError(
            "pngs (%d) and timings (%d) must have the same length"
            % (len(pngs), len(timings))
        )

    import tqdm

    entries: List[srt.Subtitle] = []
    for png_path, (start_s, end_s) in tqdm.tqdm(
        zip(pngs, timings),
        total=len(pngs),
        desc="OCR PNGs",
        unit="img",
        dynamic_ncols=True,
    ):
        img = PilImage.open(png_path).convert("RGBA")
        text = ocr_image(img, language).strip()
        logger.debug("%s -> %r", os.path.basename(png_path), text)
        if text:
            entries.append(
                srt.Subtitle(
                    index=len(entries) + 1,
                    start=timedelta(seconds=start_s),
                    end=timedelta(seconds=end_s),
                    content=text,
                )
            )

    logger.info("OCR produced %d entries from %d PNGs", len(entries), len(pngs))
    with open(output_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(entries))
    logger.info("wrote SRT -> %s", output_srt)
    return len(entries)
