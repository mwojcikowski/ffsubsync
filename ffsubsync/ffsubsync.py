#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import cast, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ffsubsync.aligners import FFTAligner, MaxScoreAligner
from ffsubsync.constants import (
    DEFAULT_APPLY_OFFSET_SECONDS,
    DEFAULT_FRAME_RATE,
    DEFAULT_MAX_OFFSET_SECONDS,
    DEFAULT_MAX_SUBTITLE_SECONDS,
    DEFAULT_NON_SPEECH_LABEL,
    DEFAULT_START_SECONDS,
    DEFAULT_VAD,
    DEFAULT_ENCODING,
    FRAMERATE_RATIOS,
    SAMPLE_RATE,
    SUBTITLE_EXTENSIONS,
)
from ffsubsync.ffmpeg_utils import ffmpeg_bin_path
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
    DeserializeSpeechTransformer,
    PGSSpeechTransformer,
    make_subtitle_speech_pipeline,
)
from ffsubsync.cross_lingual_aligner import (
    align_subtitles_by_content,
    DEFAULT_MODEL as DEFAULT_TEXT_ALIGN_MODEL,
    DEFAULT_THRESHOLD as DEFAULT_TEXT_ALIGN_THRESHOLD,
)
from ffsubsync.pgs_ocr import (
    pgs_to_srt as _pgs_to_srt,
    sup_to_srt as _sup_to_srt,
    pgs_to_generic_subs as _pgs_to_generic_subs,
    sup_to_generic_subs as _sup_to_generic_subs,
)
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import SubtitleMerger, SubtitleShifter
from ffsubsync.version import get_version


logger: logging.Logger = logging.getLogger(__name__)


def override(args: argparse.Namespace, **kwargs: Any) -> Dict[str, Any]:
    args_dict = dict(args.__dict__)
    args_dict.update(kwargs)
    return args_dict


def _ref_format(ref_fname: Optional[str]) -> Optional[str]:
    if ref_fname is None:
        return None
    return ref_fname[-3:]


def make_test_case(
    args: argparse.Namespace, npy_savename: Optional[str], sync_was_successful: bool
) -> int:
    if npy_savename is None:
        raise ValueError("need non-null npy_savename")
    tar_dir = "{}.{}".format(
        args.reference, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    logger.info("creating test archive {}.tar.gz...".format(tar_dir))
    os.mkdir(tar_dir)
    try:
        log_path = "ffsubsync.log"
        if args.log_dir_path is not None and os.path.isdir(args.log_dir_path):
            log_path = os.path.join(args.log_dir_path, log_path)
        shutil.copy(log_path, tar_dir)
        shutil.copy(args.srtin[0], tar_dir)
        if sync_was_successful:
            shutil.move(args.srtout, tar_dir)
        if _ref_format(args.reference) in SUBTITLE_EXTENSIONS:
            shutil.copy(args.reference, tar_dir)
        elif args.serialize_speech or args.reference == npy_savename:
            shutil.copy(npy_savename, tar_dir)
        else:
            shutil.move(npy_savename, tar_dir)
        supported_formats = set(list(zip(*shutil.get_archive_formats()))[0])
        preferred_formats = ["gztar", "bztar", "xztar", "zip", "tar"]
        for archive_format in preferred_formats:
            if archive_format in supported_formats:
                shutil.make_archive(tar_dir, archive_format, os.curdir, tar_dir)
                break
        else:
            logger.error(
                "failed to create test archive; no formats supported "
                "(this should not happen)"
            )
            return 1
        logger.info("...done")
    finally:
        shutil.rmtree(tar_dir)
    return 0


def get_srt_pipe_maker(
    args: argparse.Namespace, srtin: Optional[str]
) -> Callable[[Optional[float]], Union[Pipeline, Callable[[float], Pipeline]]]:
    if srtin is None:
        srtin_format = "srt"
    else:
        srtin_format = os.path.splitext(srtin)[-1][1:]
    parser = make_subtitle_parser(fmt=srtin_format, caching=True, **args.__dict__)
    return lambda scale_factor: make_subtitle_speech_pipeline(
        **override(args, scale_factor=scale_factor, parser=parser)
    )


def get_framerate_ratios_to_try(args: argparse.Namespace) -> List[Optional[float]]:
    if args.no_fix_framerate:
        return []
    else:
        framerate_ratios = list(
            np.concatenate(
                [np.array(FRAMERATE_RATIOS), 1.0 / np.array(FRAMERATE_RATIOS)]
            )
        )
        if args.gss:
            framerate_ratios.append(None)
        return framerate_ratios


def try_sync(
    args: argparse.Namespace, reference_pipe: Optional[Pipeline], result: Dict[str, Any]
) -> bool:
    result["sync_was_successful"] = False
    sync_was_successful = True
    logger.info(
        "extracting speech segments from %s...",
        "stdin" if not args.srtin else "subtitles file(s) {}".format(args.srtin),
    )
    if not args.srtin:
        args.srtin = [None]
    for srtin in args.srtin:
        try:
            skip_sync = args.skip_sync or reference_pipe is None
            skip_infer_framerate_ratio = (
                args.skip_infer_framerate_ratio or reference_pipe is None
            )
            srtout = srtin if args.overwrite_input else args.srtout
            srt_pipe_maker = get_srt_pipe_maker(args, srtin)
            framerate_ratios = get_framerate_ratios_to_try(args)
            srt_pipes = [srt_pipe_maker(1.0)] + [
                srt_pipe_maker(rat) for rat in framerate_ratios
            ]
            for srt_pipe in srt_pipes:
                if callable(srt_pipe):
                    continue
                else:
                    srt_pipe.fit(srtin)
            if (
                not skip_infer_framerate_ratio
                and hasattr(reference_pipe[-1], "num_frames")
                and reference_pipe[-1].num_frames is not None
            ):
                inferred_framerate_ratio_from_length = (
                    float(reference_pipe[-1].num_frames)
                    / cast(Pipeline, srt_pipes[0])[-1].num_frames
                )
                logger.info(
                    "inferred frameratio ratio: %.3f"
                    % inferred_framerate_ratio_from_length
                )
                srt_pipes.append(
                    cast(
                        Pipeline, srt_pipe_maker(inferred_framerate_ratio_from_length)
                    ).fit(srtin)
                )
                logger.info("...done")
            logger.info("computing alignments...")
            if skip_sync:
                best_score = 0.0
                best_srt_pipe = cast(Pipeline, srt_pipes[0])
                offset_samples = 0
            else:
                (best_score, offset_samples), best_srt_pipe = MaxScoreAligner(
                    FFTAligner, srtin, SAMPLE_RATE, args.max_offset_seconds
                ).fit_transform(
                    reference_pipe.transform(args.reference),
                    srt_pipes,
                )
            if best_score < 0:
                sync_was_successful = False
            logger.info("...done")
            offset_seconds = (
                offset_samples / float(SAMPLE_RATE) + args.apply_offset_seconds
            )
            scale_step = best_srt_pipe.named_steps["scale"]
            speech_extractor = best_srt_pipe.named_steps.get("speech_extract")
            speech_arr = getattr(speech_extractor, "subtitle_speech_results_", None)
            normalized_score = (
                best_score / len(speech_arr)
                if speech_arr is not None and len(speech_arr) > 0
                else None
            )
            if normalized_score is not None:
                logger.info("score: %.3f (%.1f%%)", best_score, normalized_score * 100)
            else:
                logger.info("score: %.3f", best_score)
            logger.info("offset seconds: %.3f", offset_seconds)
            logger.info("framerate scale factor: %.3f", scale_step.scale_factor)
            output_steps: List[Tuple[str, TransformerMixin]] = [
                ("shift", SubtitleShifter(offset_seconds))
            ]
            if args.merge_with_reference:
                output_steps.append(
                    ("merge", SubtitleMerger(reference_pipe.named_steps["parse"].subs_))
                )
            output_pipe = Pipeline(output_steps)
            out_subs = output_pipe.fit_transform(scale_step.subs_)
            if args.output_encoding != "same":
                out_subs = out_subs.set_encoding(args.output_encoding)
            suppress_output_thresh = args.suppress_output_if_offset_less_than
            if offset_seconds >= (suppress_output_thresh or float("-inf")):
                bad_threshold = getattr(args, "bad_sync_threshold", None)
                skip_write = False
                score_for_threshold = (
                    normalized_score if normalized_score is not None else best_score
                )
                if (
                    bad_threshold is not None
                    and bad_threshold > 0
                    and score_for_threshold < bad_threshold
                ):
                    will_overwrite = srtout is not None and os.path.exists(srtout)
                    if will_overwrite and not args.gui_mode and not args.vlc_mode:
                        logger.warning(
                            "WARNING: alignment score %.1f%% is below --bad-sync-threshold %.1f%%; "
                            "subtitles may not match the reference (e.g. wrong episode).",
                            score_for_threshold * 100,
                            bad_threshold * 100,
                        )
                        response = (
                            input("Proceed with overwrite of '%s'? [y/N]: " % srtout)
                            .strip()
                            .lower()
                        )
                        if response not in ("y", "yes"):
                            logger.warning(
                                "Skipping write due to suspicious alignment score."
                            )
                            skip_write = True
                if not skip_write:
                    logger.info("writing output to {}".format(srtout or "stdout"))
                    out_subs.write_file(srtout)
            else:
                logger.warning(
                    "suppressing output because offset %s was less than suppression threshold %s",
                    offset_seconds,
                    args.suppress_output_if_offset_less_than,
                )
        except Exception:
            sync_was_successful = False
            logger.exception("failed to sync %s", srtin)
        else:
            result["offset_seconds"] = offset_seconds
            result["framerate_scale_factor"] = scale_step.scale_factor
            result["score"] = best_score
            result["score_normalized"] = normalized_score
    result["sync_was_successful"] = sync_was_successful
    return sync_was_successful


def try_sync_by_text(args: argparse.Namespace, result: Dict[str, Any]) -> bool:
    """Assign timestamps from a reference subtitle file to a target subtitle file
    using cross-lingual sentence embeddings and monotone DP alignment."""
    result["sync_was_successful"] = False
    try:
        ref_format = os.path.splitext(args.reference)[-1][1:]
        ref_parser = make_subtitle_parser(
            fmt=ref_format,
            encoding=args.reference_encoding or DEFAULT_ENCODING,
            max_subtitle_seconds=args.max_subtitle_seconds,
            start_seconds=args.start_seconds,
            strict=args.strict,
        )
        ref_parser.fit(args.reference)
        ref_subs = list(ref_parser.subs_)

        srtin = args.srtin[0]
        target_format = os.path.splitext(srtin)[-1][1:]
        target_parser = make_subtitle_parser(
            fmt=target_format,
            encoding=args.encoding,
            max_subtitle_seconds=args.max_subtitle_seconds,
            start_seconds=args.start_seconds,
            strict=args.strict,
        )
        target_parser.fit(srtin)
        target_subs_file = target_parser.subs_
        target_subs = list(target_subs_file)

        model_name = getattr(args, "text_align_model", DEFAULT_TEXT_ALIGN_MODEL)
        threshold = getattr(args, "text_align_threshold", DEFAULT_TEXT_ALIGN_THRESHOLD)

        stats: Dict[str, Any] = {}
        new_subs = align_subtitles_by_content(
            ref_subs,
            target_subs,
            model_name=model_name,
            threshold=threshold,
            out_stats=stats,
        )

        out_subs_file = target_subs_file.clone_props_for_subs(new_subs)
        if args.output_encoding != "same":
            out_subs_file = out_subs_file.set_encoding(args.output_encoding)

        srtout = srtin if args.overwrite_input else args.srtout
        bad_threshold = getattr(args, "bad_sync_threshold", None)
        match_ratio = stats.get("match_ratio", 1.0)
        if (
            bad_threshold is not None
            and bad_threshold > 0
            and match_ratio < bad_threshold
            and srtout is not None
            and os.path.exists(srtout)
            and not args.gui_mode
            and not args.vlc_mode
        ):
            logger.warning(
                "WARNING: text-alignment match ratio %.1f%% is below "
                "--bad-sync-threshold %.1f%%; subtitles may not match the reference.",
                match_ratio * 100,
                bad_threshold * 100,
            )
            response = (
                input("Proceed with overwrite of '%s'? [y/N]: " % srtout)
                .strip()
                .lower()
            )
            if response not in ("y", "yes"):
                logger.warning("Skipping write due to low text-alignment match ratio.")
                result["sync_was_successful"] = False
                return False
        logger.info("writing output to %s", srtout or "stdout")
        out_subs_file.write_file(srtout)
        result["sync_was_successful"] = True
        return True
    except Exception:
        logger.exception("failed to align subtitles by content")
        result["sync_was_successful"] = False
        return False


def try_pgs_ocr(args: argparse.Namespace, result: Dict[str, Any]) -> bool:
    """Extract PGS track, OCR each bitmap frame, write SRT.

    When ``--text-align`` is also set:
      - ``reference`` (first positional arg) = video / .sup file containing the
        PGS track to OCR (e.g. an MKV with English PGS subtitles).
      - ``srtin[0]`` (-i) = unaligned target SRT (e.g. Polish, wrong timestamps).
      - The OCR'd PGS subs are used as the *reference* for cross-lingual alignment
        and their timestamps are assigned to the target lines.
    """
    result["sync_was_successful"] = False
    try:
        sup_path = getattr(args, "pgs_ocr_dump_sup", None)
        pngs_dir = getattr(args, "pgs_ocr_dump_pngs", None)
        language = getattr(args, "pgs_ocr_lang", "en-US")
        stream = getattr(args, "pgs_ref_stream", None)
        if stream == "auto":
            stream = None

        if getattr(args, "text_align", False):
            # Combined mode:
            #   reference = video/sup to OCR → English subs with PGS timestamps
            #   srtin[0]  = unaligned target SRT (Polish) to assign timestamps to
            pgs_source = args.reference
            srtin = args.srtin[0]
            logger.info("combined PGS-OCR + text-align mode")
            logger.info("OCR source (PGS)  : %s", pgs_source)
            logger.info("target to align   : %s", srtin)

            # 1. OCR the PGS track → reference subs with timestamps
            if pgs_source.lower().endswith(".sup"):
                ref_subs = _sup_to_generic_subs(
                    pgs_source, language=language, dump_pngs_dir=pngs_dir
                )
            else:
                ref_subs = _pgs_to_generic_subs(
                    pgs_source,
                    stream=stream,
                    language=language,
                    ffmpeg_path=args.ffmpeg_path,
                    gui_mode=args.gui_mode,
                    dump_sup=sup_path,
                    dump_pngs_dir=pngs_dir,
                )

            # 2. Parse the unaligned target SRT
            target_format = os.path.splitext(srtin)[-1][1:]
            target_parser = make_subtitle_parser(
                fmt=target_format,
                encoding=args.encoding,
                max_subtitle_seconds=args.max_subtitle_seconds,
                start_seconds=args.start_seconds,
                strict=args.strict,
            )
            target_parser.fit(srtin)
            target_subs_file = target_parser.subs_
            target_subs = list(target_subs_file)

            # 3. Cross-lingual alignment: assigns PGS timestamps to target lines
            model_name = getattr(args, "text_align_model", DEFAULT_TEXT_ALIGN_MODEL)
            threshold = getattr(
                args, "text_align_threshold", DEFAULT_TEXT_ALIGN_THRESHOLD
            )
            ocr_stats: Dict[str, Any] = {}
            aligned_subs = align_subtitles_by_content(
                ref_subs,
                target_subs,
                model_name=model_name,
                threshold=threshold,
                out_stats=ocr_stats,
            )

            # 4. Write output preserving the target file's format/encoding
            out_file = target_subs_file.clone_props_for_subs(aligned_subs)
            if args.output_encoding != "same":
                out_file = out_file.set_encoding(args.output_encoding)
            srtout = srtin if args.overwrite_input else args.srtout
            if srtout is None:
                srtout = os.path.splitext(srtin)[0] + ".synced.srt"
                logger.info("auto-detected output path: %s", srtout)
            bad_threshold = getattr(args, "bad_sync_threshold", None)
            match_ratio = ocr_stats.get("match_ratio", 1.0)
            if (
                bad_threshold is not None
                and bad_threshold > 0
                and match_ratio < bad_threshold
                and srtout is not None
                and os.path.exists(srtout)
                and not args.gui_mode
                and not args.vlc_mode
            ):
                logger.warning(
                    "WARNING: text-alignment match ratio %.1f%% is below "
                    "--bad-sync-threshold %.1f%%; subtitles may not match the reference.",
                    match_ratio * 100,
                    bad_threshold * 100,
                )
                response = (
                    input("Proceed with overwrite of '%s'? [y/N]: " % srtout)
                    .strip()
                    .lower()
                )
                if response not in ("y", "yes"):
                    logger.warning(
                        "Skipping write due to low text-alignment match ratio."
                    )
                    result["sync_was_successful"] = False
                    return False
            logger.info("writing aligned output to %s", srtout)
            out_file.write_file(srtout)
            result["sync_was_successful"] = True
            return True

        # Plain OCR-only mode: reference = video/sup, srtout = output SRT
        srtout = args.srtout
        if srtout is None and args.srtin:
            srtout = args.srtin[0]
        if srtout is None:
            # Auto-derive output path from the reference filename
            srtout = os.path.splitext(args.reference)[0] + ".srt"
            logger.info("auto-detected output path: %s", srtout)

        ref = args.reference
        if ref is not None and ref.lower().endswith(".sup"):
            n = _sup_to_srt(ref, srtout, language=language, dump_pngs_dir=pngs_dir)
        else:
            n = _pgs_to_srt(
                ref,
                stream=stream,
                output_srt=srtout,
                language=language,
                ffmpeg_path=args.ffmpeg_path,
                gui_mode=args.gui_mode,
                dump_sup=sup_path,
                dump_pngs_dir=pngs_dir,
            )
        logger.info("wrote %d subtitle entries to %s", n, srtout)
        result["sync_was_successful"] = True
        return True
    except Exception:
        logger.exception("PGS OCR failed")
        result["sync_was_successful"] = False
        return False


def make_reference_pipe(args: argparse.Namespace) -> Pipeline:
    pgs_stream = getattr(args, "pgs_ref_stream", None)
    if pgs_stream is not None:
        # "auto" (bare --pgs-ref-stream flag) → let PGSSpeechTransformer auto-detect
        resolved_stream: Optional[str] = None if pgs_stream == "auto" else pgs_stream
        if resolved_stream is not None and not resolved_stream.startswith("0:"):
            resolved_stream = "0:" + resolved_stream
        return Pipeline(
            [
                (
                    "speech_extract",
                    PGSSpeechTransformer(
                        sample_rate=SAMPLE_RATE,
                        start_seconds=args.start_seconds,
                        ffmpeg_path=args.ffmpeg_path,
                        ref_stream=resolved_stream,
                        gui_mode=args.gui_mode,
                    ),
                ),
            ]
        )
    ref_format = _ref_format(args.reference)
    if ref_format in SUBTITLE_EXTENSIONS:
        if args.vad is not None:
            logger.warning("Vad specified, but reference was not a movie")
        return cast(
            Pipeline,
            make_subtitle_speech_pipeline(
                fmt=ref_format,
                **override(args, encoding=args.reference_encoding or DEFAULT_ENCODING),
            ),
        )
    elif ref_format in ("npy", "npz"):
        if args.vad is not None:
            logger.warning("Vad specified, but reference was not a movie")
        return Pipeline(
            [("deserialize", DeserializeSpeechTransformer(args.non_speech_label))]
        )
    else:
        vad = args.vad or DEFAULT_VAD
        if args.reference_encoding is not None:
            logger.warning(
                "Reference srt encoding specified, but reference was a video file"
            )
        ref_stream = args.reference_stream
        if ref_stream is not None and not ref_stream.startswith("0:"):
            ref_stream = "0:" + ref_stream
        return Pipeline(
            [
                (
                    "speech_extract",
                    VideoSpeechTransformer(
                        vad=vad,
                        sample_rate=SAMPLE_RATE,
                        frame_rate=args.frame_rate,
                        non_speech_label=args.non_speech_label,
                        start_seconds=args.start_seconds,
                        ffmpeg_path=args.ffmpeg_path,
                        ref_stream=ref_stream,
                        vlc_mode=args.vlc_mode,
                        gui_mode=args.gui_mode,
                    ),
                ),
            ]
        )


def extract_subtitles_from_reference(args: argparse.Namespace) -> int:
    stream = args.extract_subs_from_stream
    if not stream.startswith("0:s:"):
        stream = "0:s:{}".format(stream)
    elif not stream.startswith("0:") and stream.startswith("s:"):
        stream = "0:{}".format(stream)
    if not stream.startswith("0:s:"):
        logger.error(
            "invalid stream for subtitle extraction: %s", args.extract_subs_from_stream
        )
    ffmpeg_args = [
        ffmpeg_bin_path("ffmpeg", args.gui_mode, ffmpeg_resources_path=args.ffmpeg_path)
    ]
    ffmpeg_args.extend(
        [
            "-y",
            "-nostdin",
            "-loglevel",
            "fatal",
            "-i",
            args.reference,
            "-map",
            "{}".format(stream),
            "-f",
            "srt",
        ]
    )
    if args.srtout is None:
        ffmpeg_args.append("-")
    else:
        ffmpeg_args.append(args.srtout)
    logger.info(
        "attempting to extract subtitles to {} ...".format(
            "stdout" if args.srtout is None else args.srtout
        )
    )
    retcode = subprocess.call(ffmpeg_args)
    if retcode == 0:
        logger.info("...done")
    else:
        logger.error(
            "ffmpeg unable to extract subtitles from reference; return code %d", retcode
        )
    return retcode


def validate_args(args: argparse.Namespace) -> None:
    if args.vlc_mode:
        logger.setLevel(logging.CRITICAL)
    if args.reference is None:
        if args.apply_offset_seconds == 0 or not args.srtin:
            raise ValueError(
                "`reference` required unless `--apply-offset-seconds` specified"
            )
    if args.apply_offset_seconds != 0:
        if not args.srtin:
            args.srtin = [args.reference]
        if not args.srtin:
            raise ValueError(
                "at least one of `srtin` or `reference` must be specified to apply offset seconds"
            )
    if args.srtin:
        if len(args.srtin) > 1 and not args.overwrite_input:
            raise ValueError(
                "cannot specify multiple input srt files without overwriting"
            )
        if len(args.srtin) > 1 and args.make_test_case:
            raise ValueError("cannot specify multiple input srt files for test cases")
        if len(args.srtin) > 1 and args.gui_mode:
            raise ValueError("cannot specify multiple input srt files in GUI mode")
    if (
        args.make_test_case and not args.gui_mode
    ):  # this validation not necessary for gui mode
        if not args.srtin or args.srtout is None:
            raise ValueError(
                "need to specify input and output srt files for test cases"
            )
    if args.overwrite_input:
        if args.extract_subs_from_stream is not None:
            raise ValueError(
                "input overwriting not allowed for extracting subtitles from reference"
            )
        if not args.srtin:
            raise ValueError(
                "need to specify input srt if --overwrite-input "
                "is specified since we cannot overwrite stdin"
            )
        if args.srtout is not None:
            raise ValueError(
                "overwrite input set but output file specified; "
                "refusing to run in case this was not intended"
            )
    if args.extract_subs_from_stream is not None:
        if args.make_test_case:
            raise ValueError("test case is for sync and not subtitle extraction")
        if args.srtin:
            raise ValueError(
                "stream specified for reference subtitle extraction; "
                "-i flag for sync input not allowed"
            )
    if getattr(args, "text_align", False) and not getattr(args, "pgs_ocr", False):
        if args.reference is None:
            raise ValueError("--text-align requires a reference subtitle file")
        ref_fmt = _ref_format(args.reference)
        if ref_fmt not in SUBTITLE_EXTENSIONS:
            raise ValueError(
                "--text-align requires the reference to be a subtitle file "
                "(srt/ass/ssa/sub/vtt), got: %s" % ref_fmt
            )
        if not args.srtin:
            raise ValueError("--text-align requires an input subtitle file (-i)")
    if getattr(args, "pgs_ocr", False):
        if getattr(args, "text_align", False):
            # Combined mode:
            #   reference = video/sup with PGS to OCR (e.g. movie.mkv or english.sup)
            #   srtin[0]  = unaligned target SRT (e.g. polish.srt)
            if args.reference is None:
                raise ValueError(
                    "--pgs-ocr --text-align requires a reference video or .sup file "
                    "(first positional arg, e.g. movie.mkv)"
                )
            if not args.srtin:
                raise ValueError(
                    "--pgs-ocr --text-align requires an input subtitle file (-i), "
                    "e.g. polish_unaligned.srt"
                )
        else:
            if args.reference is None:
                raise ValueError("--pgs-ocr requires a reference video or .sup file")


def validate_file_permissions(args: argparse.Namespace) -> None:
    error_string_template = (
        "unable to {action} {file}; "
        "try ensuring file exists and has correct permissions"
    )
    if args.reference is not None and not os.access(args.reference, os.R_OK):
        raise ValueError(
            error_string_template.format(action="read reference", file=args.reference)
        )
    if args.srtin:
        for srtin in args.srtin:
            if srtin is not None and not os.access(srtin, os.R_OK):
                raise ValueError(
                    error_string_template.format(
                        action="read input subtitles", file=srtin
                    )
                )
    if (
        args.srtout is not None
        and os.path.exists(args.srtout)
        and not os.access(args.srtout, os.W_OK)
    ):
        raise ValueError(
            error_string_template.format(
                action="write output subtitles", file=args.srtout
            )
        )
    if args.make_test_case or args.serialize_speech:
        npy_savename = os.path.splitext(args.reference)[0] + ".npz"
        if os.path.exists(npy_savename) and not os.access(npy_savename, os.W_OK):
            raise ValueError(
                "unable to write test case file archive %s (try checking permissions)"
                % npy_savename
            )


def _setup_logging(
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[logging.FileHandler]]:
    log_handler = None
    log_path = None
    if args.make_test_case or args.log_dir_path is not None:
        log_path = "ffsubsync.log"
        if args.log_dir_path is not None and os.path.isdir(args.log_dir_path):
            log_path = os.path.join(args.log_dir_path, log_path)
        log_handler = logging.FileHandler(log_path)
        logger.addHandler(log_handler)
        logger.info("this log will be written to %s", os.path.abspath(log_path))
    return log_path, log_handler


def _npy_savename(args: argparse.Namespace) -> str:
    return os.path.splitext(args.reference)[0] + ".npz"


def _run_impl(args: argparse.Namespace, result: Dict[str, Any]) -> bool:
    if getattr(args, "pgs_ocr", False):
        return try_pgs_ocr(args, result)
    if getattr(args, "text_align", False):
        return try_sync_by_text(args, result)
    if args.extract_subs_from_stream is not None:
        result["retval"] = extract_subtitles_from_reference(args)
        return True
    if args.srtin is not None and (
        args.reference is None
        or (len(args.srtin) == 1 and args.srtin[0] == args.reference)
    ):
        return try_sync(args, None, result)
    reference_pipe = make_reference_pipe(args)
    logger.info("extracting speech segments from reference '%s'...", args.reference)
    reference_pipe.fit(args.reference)
    logger.info("...done")
    if args.make_test_case or args.serialize_speech:
        logger.info("serializing speech...")
        np.savez_compressed(
            _npy_savename(args), speech=reference_pipe.transform(args.reference)
        )
        logger.info("...done")
        if not args.srtin:
            logger.info(
                "unsynchronized subtitle file not specified; skipping synchronization"
            )
            return False
    return try_sync(args, reference_pipe, result)


def validate_and_transform_args(
    parser_or_args: Union[argparse.ArgumentParser, argparse.Namespace],
) -> Optional[argparse.Namespace]:
    if isinstance(parser_or_args, argparse.Namespace):
        parser = None
        args = parser_or_args
    else:
        parser = parser_or_args
        args = parser.parse_args()
    if (
        not args.srtin
        and args.reference is not None
        and args.extract_subs_from_stream is None
    ):
        ref = Path(args.reference)
        candidates = sorted(ref.parent.glob(ref.stem + "*.srt"))
        if len(candidates) == 1:
            args.srtin = [str(candidates[0])]
            logger.info(
                "no input subtitle specified; auto-detected: %s",
                args.srtin[0],
            )
        elif len(candidates) > 1:
            logger.warning(
                "no input subtitle specified and multiple candidates found (%s); "
                "please specify one with -i",
                ", ".join(str(p) for p in candidates),
            )
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(e)
        if parser is not None:
            parser.print_usage()
        return None
    if args.gui_mode and args.srtout is None:
        args.srtout = "{}.synced.srt".format(os.path.splitext(args.srtin[0])[0])
    try:
        validate_file_permissions(args)
    except ValueError as e:
        logger.error(e)
        return None
    ref_format = _ref_format(args.reference)
    if args.merge_with_reference and ref_format not in SUBTITLE_EXTENSIONS:
        logger.error(
            "merging synced output with reference only valid "
            "when reference composed of subtitles"
        )
        return None
    return args


def run(
    parser_or_args: Union[argparse.ArgumentParser, argparse.Namespace],
) -> Dict[str, Any]:
    sync_was_successful = False
    result = {
        "retval": 0,
        "offset_seconds": None,
        "framerate_scale_factor": None,
        "score": None,
        "score_normalized": None,
    }
    args = validate_and_transform_args(parser_or_args)
    if args is None:
        result["retval"] = 1
        return result
    log_path, log_handler = _setup_logging(args)
    try:
        sync_was_successful = _run_impl(args, result)
        result["sync_was_successful"] = sync_was_successful
        return result
    finally:
        if log_handler is not None and log_path is not None:
            log_handler.close()
            logger.removeHandler(log_handler)
            if args.make_test_case:
                result["retval"] += make_test_case(
                    args, _npy_savename(args), sync_was_successful
                )
            if args.log_dir_path is None or not os.path.isdir(args.log_dir_path):
                os.remove(log_path)


def add_main_args_for_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "reference",
        nargs="?",
        help=(
            "Reference (video, subtitles, or a numpy array with VAD speech) "
            "to which to synchronize input subtitles."
        ),
    )
    parser.add_argument(
        "-i", "--srtin", nargs="*", help="Input subtitles file (default=stdin)."
    )
    parser.add_argument(
        "-o", "--srtout", help="Output subtitles file (default=stdout)."
    )
    parser.add_argument(
        "--merge-with-reference",
        "--merge",
        action="store_true",
        help="Merge reference subtitles with synced output subtitles.",
    )
    parser.add_argument(
        "--make-test-case",
        "--create-test-case",
        action="store_true",
        help="If specified, serialize reference speech to a numpy array, "
        "and create an archive with input/output subtitles "
        "and serialized speech.",
    )
    parser.add_argument(
        "--reference-stream",
        "--refstream",
        "--reference-track",
        "--reftrack",
        default=None,
        help=(
            "Which stream/track in the video file to use as reference, "
            "formatted according to ffmpeg conventions. For example, 0:s:0 "
            "uses the first subtitle track; 0:a:3 would use the third audio track. "
            "You can also drop the leading `0:`; i.e. use s:0 or a:3, respectively. "
            "Example: `ffs ref.mkv -i in.srt -o out.srt --reference-stream s:2`"
        ),
    )
    parser.add_argument(
        "--pgs-ref-stream",
        "--pgsstream",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Use a PGS (Presentation Graphic Stream) image-based subtitle track from "
            "the reference MKV as the sync reference instead of audio VAD. "
            "Optionally specify the stream (leading `0:` is optional, e.g. `s:0` or `3`). "
            "Omit the value to auto-detect the first hdmv_pgs_subtitle track. "
            "Example: `ffs ref.mkv -i in.srt -o out.srt --pgs-ref-stream` (auto) "
            "or `ffs ref.mkv -i in.srt -o out.srt --pgs-ref-stream s:2` (explicit)."
        ),
    )


def add_cli_only_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="{package} {version}".format(
            package=__package__, version=get_version()
        ),
    )
    parser.add_argument(
        "--overwrite-input",
        action="store_true",
        help=(
            "If specified, will overwrite the input srt "
            "instead of writing the output to a new file."
        ),
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="What encoding to use for reading input subtitles "
        "(default=%s)." % DEFAULT_ENCODING,
    )
    parser.add_argument(
        "--max-subtitle-seconds",
        type=float,
        default=DEFAULT_MAX_SUBTITLE_SECONDS,
        help="Maximum duration for a subtitle to appear on-screen "
        "(default=%.3f seconds)." % DEFAULT_MAX_SUBTITLE_SECONDS,
    )
    parser.add_argument(
        "--start-seconds",
        type=int,
        default=DEFAULT_START_SECONDS,
        help="Start time for processing "
        "(default=%d seconds)." % DEFAULT_START_SECONDS,
    )
    parser.add_argument(
        "--max-offset-seconds",
        type=float,
        default=DEFAULT_MAX_OFFSET_SECONDS,
        help="The max allowed offset seconds for any subtitle segment "
        "(default=%d seconds)." % DEFAULT_MAX_OFFSET_SECONDS,
    )
    parser.add_argument(
        "--apply-offset-seconds",
        type=float,
        default=DEFAULT_APPLY_OFFSET_SECONDS,
        help="Apply a predefined offset in seconds to all subtitle segments "
        "(default=%d seconds)." % DEFAULT_APPLY_OFFSET_SECONDS,
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=DEFAULT_FRAME_RATE,
        help="Frame rate for audio extraction (default=%d)." % DEFAULT_FRAME_RATE,
    )
    parser.add_argument(
        "--skip-infer-framerate-ratio",
        action="store_true",
        help="If set, do not try to infer framerate ratio based on duration ratio.",
    )
    parser.add_argument(
        "--non-speech-label",
        type=float,
        default=DEFAULT_NON_SPEECH_LABEL,
        help="Label to use for frames detected as non-speech (default=%f)"
        % DEFAULT_NON_SPEECH_LABEL,
    )
    parser.add_argument(
        "--output-encoding",
        default="utf-8",
        help="What encoding to use for writing output subtitles "
        '(default=utf-8). Can indicate "same" to use same '
        "encoding as that of the input.",
    )
    parser.add_argument(
        "--reference-encoding",
        help="What encoding to use for reading / writing reference subtitles "
        "(if applicable, default=infer).",
    )
    parser.add_argument(
        "--vad",
        choices=[
            "subs_then_webrtc",
            "webrtc",
            "subs_then_auditok",
            "auditok",
            "subs_then_silero",
            "silero",
        ],
        default=None,
        help="Which voice activity detector to use for speech extraction "
        "(if using video / audio as a reference, default={}).".format(DEFAULT_VAD),
    )
    parser.add_argument(
        "--no-fix-framerate",
        action="store_true",
        help="If specified, subsync will not attempt to correct a framerate "
        "mismatch between reference and subtitles.",
    )
    parser.add_argument(
        "--serialize-speech",
        action="store_true",
        help="If specified, serialize reference speech to a numpy array.",
    )
    parser.add_argument(
        "--extract-subs-from-stream",
        "--extract-subtitles-from-stream",
        default=None,
        help="If specified, do not attempt sync; instead, just extract subtitles"
        " from the specified stream using the reference.",
    )
    parser.add_argument(
        "--suppress-output-if-offset-less-than",
        type=float,
        default=None,
        help="If specified, do not produce output if offset below provided threshold.",
    )
    parser.add_argument(
        "--bad-sync-threshold",
        type=float,
        default=0.7,
        help=(
            "If the normalized alignment score is below this value (0.0-1.0), warn and "
            "ask for confirmation before overwriting a file. "
            "Good matches typically score 0.5+ (50%%+); scores below this threshold "
            "can indicate subtitles from the wrong source (e.g. a switched episode). "
            "Set to 0 to disable. Default: 0.4 (40%%)."
        ),
    )
    parser.add_argument(
        "--ffmpeg-path",
        "--ffmpegpath",
        default=None,
        help="Where to look for ffmpeg and ffprobe. Uses the system PATH by default.",
    )
    parser.add_argument(
        "--log-dir-path",
        default=None,
        help=(
            "If provided, will save log file ffsubsync.log to this path "
            "(must be an existing directory)."
        ),
    )
    parser.add_argument(
        "--gss",
        action="store_true",
        help="If specified, use golden-section search to try to find"
        "the optimal framerate ratio between video and subtitles.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If specified, refuse to parse srt files with formatting issues.",
    )
    parser.add_argument(
        "--text-align",
        action="store_true",
        help=(
            "If specified, align target subtitles to reference subtitles using "
            "cross-lingual sentence embeddings instead of audio/speech timing. "
            "Useful when you have an English SRT (reference) and a Polish SRT "
            "(e.g. OCR'd from PGS Blu-ray) and want to assign the correct "
            "English timestamps to the Polish lines. "
            "Requires 'sentence-transformers' to be installed."
        ),
    )
    parser.add_argument(
        "--text-align-model",
        default=DEFAULT_TEXT_ALIGN_MODEL,
        help=(
            "Multilingual sentence-transformers model to use for --text-align "
            "(default=%s)." % DEFAULT_TEXT_ALIGN_MODEL
        ),
    )
    parser.add_argument(
        "--text-align-threshold",
        type=float,
        default=DEFAULT_TEXT_ALIGN_THRESHOLD,
        help=(
            "Minimum cosine similarity for a subtitle pair to be accepted as a match "
            "during --text-align (default=%.1f)." % DEFAULT_TEXT_ALIGN_THRESHOLD
        ),
    )
    parser.add_argument(
        "--pgs-ocr",
        action="store_true",
        help=(
            "Extract a PGS (bitmap) subtitle track from the reference video or a "
            "pre-existing .sup file, OCR each frame with ocrmac (macOS Vision), "
            "and write an SRT whose timecodes come directly from the PGS stream. "
            "Requires: pip install ocrmac Pillow"
        ),
    )
    parser.add_argument(
        "--pgs-ocr-lang",
        default="en-US",
        metavar="LANG",
        help=(
            "BCP-47 language tag for ocrmac recognition during --pgs-ocr "
            "(default=en-US). Examples: en-US, de-DE, fr-FR."
        ),
    )
    parser.add_argument(
        "--pgs-ocr-dump-sup",
        default=None,
        metavar="PATH",
        help="If specified with --pgs-ocr, keep the extracted .sup stream at PATH.",
    )
    parser.add_argument(
        "--pgs-ocr-dump-pngs",
        default=None,
        metavar="DIR",
        help=(
            "If specified with --pgs-ocr, save each subtitle bitmap as a numbered "
            "PNG in DIR (created if it does not exist). Useful for inspection."
        ),
    )
    parser.add_argument("--vlc-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gui-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-sync", action="store_true", help=argparse.SUPPRESS)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize subtitles with video.")
    add_main_args_for_cli(parser)
    add_cli_only_args(parser)
    return parser


def main() -> int:
    parser = make_parser()
    return run(parser)["retval"]


if __name__ == "__main__":
    sys.exit(main())
