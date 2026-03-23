# -*- coding: utf-8 -*-
"""
Audio transcription pipeline.

Workflow:
  1. Extract the audio track from the reference video to a temporary mono 16 kHz
     WAV file using ffmpeg.
  2. Transcribe using one of two backends:
       - ``faster-whisper`` (default, cross-platform, CPU via CTranslate2)
       - ``mlx-whisper``    (Apple Silicon only, GPU/ANE via MLX)
  3. Return timed subtitle segments as a list of GenericSubtitle, or write to SRT.

Usage examples::

    # Transcribe audio of movie.mkv → movie.srt
    ffs movie.mkv --transcribe

    # Transcribe English audio + cross-linguially align Polish SRT
    ffs movie.mkv --transcribe --text-align -i polish_unaligned.srt -o polish_synced.srt

    # Use a larger model on Apple Silicon
    ffs movie.mkv --transcribe --transcribe-model large-v3 --transcribe-backend mlx-whisper \\
        --text-align -i polish.srt

Requirements:
    pip install faster-whisper          # cross-platform (default backend)
    pip install mlx-whisper             # Apple Silicon only (optional backend)
"""

import logging
import os
import subprocess
import tempfile
from datetime import timedelta
from typing import List, Optional, Tuple

import srt
import tqdm

from ffsubsync.ffmpeg_utils import ffmpeg_bin_path, subprocess_args
from ffsubsync.generic_subtitles import GenericSubtitle

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "base"
DEFAULT_WHISPER_LANGUAGE = "en"
DEFAULT_WHISPER_BACKEND = "faster-whisper"


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def extract_audio(
    video_path: str,
    output_audio: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> None:
    """Extract a mono 16 kHz WAV from *video_path* using ffmpeg.

    Whisper was trained on 16 kHz mono audio; resampling here avoids doing
    it inside the model library.
    """
    ffmpeg = ffmpeg_bin_path("ffmpeg", gui_mode, ffmpeg_resources_path=ffmpeg_path)
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-loglevel",
        "fatal",
        "-i",
        video_path,
        "-vn",  # drop video
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16 kHz
        "-f",
        "wav",
        output_audio,
    ]
    logger.info("extracting audio from %s...", video_path)
    ret = subprocess.call(cmd, **subprocess_args(include_stdout=False))
    if ret != 0:
        raise RuntimeError(
            "ffmpeg failed to extract audio from %r (exit code %d)" % (video_path, ret)
        )
    logger.info("...done")


# ---------------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------------


def _transcribe_faster_whisper(
    audio_path: str,
    model_name: str,
    language: Optional[str],
) -> List[Tuple[float, float, str]]:
    """Transcribe *audio_path* with faster-whisper.

    Returns a list of ``(start_s, end_s, text)`` tuples.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is required for --transcribe (default backend). "
            "Install it with: pip install faster-whisper"
        ) from exc

    logger.info("loading faster-whisper model '%s'...", model_name)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    logger.info("...done")

    logger.info("transcribing %s...", audio_path)
    segments_gen, info = model.transcribe(
        audio_path,
        language=language or None,
        beam_size=5,
        vad_filter=True,
    )
    logger.info(
        "detected language: %s (probability %.2f)",
        info.language,
        info.language_probability,
    )

    results: List[Tuple[float, float, str]] = []
    with tqdm.tqdm(desc="Transcribe", unit="seg", dynamic_ncols=True) as pbar:
        for seg in segments_gen:
            text = seg.text.strip()
            if text:
                results.append((float(seg.start), float(seg.end), text))
            pbar.update(1)
            pbar.set_postfix(segments=len(results))

    logger.info("transcribed %d segments", len(results))
    return results


def _transcribe_mlx_whisper(
    audio_path: str,
    model_name: str,
    language: Optional[str],
) -> List[Tuple[float, float, str]]:
    """Transcribe *audio_path* with mlx-whisper (Apple Silicon).

    Short model names (``base``, ``small``, ``medium``, ``large-v3``, …) are
    automatically expanded to the canonical ``mlx-community/whisper-<name>-mlx``
    HuggingFace repo.

    Returns a list of ``(start_s, end_s, text)`` tuples.
    """
    try:
        import mlx_whisper
    except ImportError as exc:
        raise ImportError(
            "mlx-whisper is required for --transcribe-backend mlx-whisper. "
            "Install it with: pip install mlx-whisper"
        ) from exc

    mlx_model = model_name
    if not mlx_model.startswith("mlx-community/"):
        mlx_model = "mlx-community/whisper-%s-mlx" % model_name

    logger.info("transcribing with mlx-whisper model '%s'...", mlx_model)
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=mlx_model,
        language=language or None,
        word_timestamps=False,
        fp16=False,
        verbose=False,
    )
    logger.info("...done")

    results: List[Tuple[float, float, str]] = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            results.append((float(seg["start"]), float(seg["end"]), text))

    logger.info("transcribed %d segments", len(results))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _timings_to_generic_subs(
    timings: List[Tuple[float, float, str]],
) -> List[GenericSubtitle]:
    """Convert ``(start_s, end_s, text)`` tuples to ``GenericSubtitle`` objects."""
    result = []
    for i, (start_s, end_s, text) in enumerate(timings):
        inner = srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=start_s),
            end=timedelta(seconds=end_s),
            content=text,
        )
        result.append(
            GenericSubtitle(timedelta(seconds=start_s), timedelta(seconds=end_s), inner)
        )
    return result


def transcribe_to_generic_subs(
    video_path: str,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = DEFAULT_WHISPER_LANGUAGE,
    backend: str = DEFAULT_WHISPER_BACKEND,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> List[GenericSubtitle]:
    """Transcribe the audio track of *video_path* and return timed subtitles.

    Parameters
    ----------
    video_path:
        Path to a video file whose audio will be transcribed.
    model_name:
        Whisper model size/name (``tiny``, ``base``, ``small``, ``medium``,
        ``large-v3``, or a full HuggingFace repo id).
    language:
        BCP-47 / ISO 639-1 language code, e.g. ``en`` or ``pl``.
        Pass ``None`` to let Whisper auto-detect.
    backend:
        ``"faster-whisper"`` (default, CPU, cross-platform) or
        ``"mlx-whisper"`` (Apple Silicon GPU/ANE).
    ffmpeg_path:
        Directory containing the ffmpeg/ffprobe binaries, or ``None`` to use
        the system PATH.
    gui_mode:
        Suppress subprocess console windows (Windows GUI use).

    Returns
    -------
    List of ``GenericSubtitle`` with transcribed text and Whisper timestamps.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    try:
        extract_audio(
            video_path,
            audio_path,
            ffmpeg_path=ffmpeg_path,
            gui_mode=gui_mode,
        )
        if backend == "mlx-whisper":
            timings = _transcribe_mlx_whisper(audio_path, model_name, language)
        else:
            timings = _transcribe_faster_whisper(audio_path, model_name, language)
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

    return _timings_to_generic_subs(timings)


def transcribe_to_srt(
    video_path: str,
    output_srt: str,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = DEFAULT_WHISPER_LANGUAGE,
    backend: str = DEFAULT_WHISPER_BACKEND,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> int:
    """Transcribe *video_path* and write an SRT file to *output_srt*.

    Returns the number of subtitle entries written.
    """
    subs = transcribe_to_generic_subs(
        video_path,
        model_name=model_name,
        language=language,
        backend=backend,
        ffmpeg_path=ffmpeg_path,
        gui_mode=gui_mode,
    )
    sub_list = [
        srt.Subtitle(
            index=i + 1,
            start=s.start,
            end=s.end,
            content=s.content,
        )
        for i, s in enumerate(subs)
    ]
    with open(output_srt, "w", encoding="utf-8") as fh:
        fh.write(srt.compose(sub_list))
    logger.info("wrote %d subtitle entries to %s", len(subs), output_srt)
    return len(subs)
