# -*- coding: utf-8 -*-
"""
Cross-lingual subtitle timestamp assignment.

Given a reference SRT with correct timestamps (e.g. English) and a target SRT
whose text is in another language (e.g. Polish, from OCR'd PGS), assigns
timestamps from the reference to the target by cross-lingual sentence-embedding
followed by a DP monotone order-preserving alignment.
"""
import copy
import logging
import re
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_THRESHOLD = 0.3


def _strip_tags(text: str) -> str:
    """Remove HTML/ASS tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{[^}]+\}", " ", text)
    return " ".join(text.split())


class CrossLingualEmbedder:
    """Wraps sentence-transformers for multilingual subtitle embedding.

    The underlying model is loaded lazily on first call to :meth:`embed`.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for cross-lingual alignment. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        logger.info("loading sentence-transformers model '%s'...", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        logger.info("...done")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (N, D) float32 embedding matrix for *texts*."""
        self._load()
        clean = [_strip_tags(t) or " " for t in texts]
        embeddings = self._model.encode(
            clean,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities: (M, D) × (N, D) → (M, N).

    Rows are L2-normalised before the dot product, so values are in [-1, 1].
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A_norm @ B_norm.T).astype(np.float32)


def dp_monotone_align(
    S: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Tuple[int, int]]:
    """Find the max-weight monotone order-preserving matching.

    ``S[j, k]`` is the similarity between target line *j* and reference line *k*.
    The algorithm uses weighted-LCS style DP:

        dp[j, k] = best total similarity when target[0..j-1] is matched within ref[0..k-1]

    Transitions:
    - skip target[j-1]:  dp[j, k] = dp[j-1, k]
    - skip ref[k-1]:     dp[j, k] = dp[j, k-1]
    - match pair:        dp[j, k] = dp[j-1, k-1] + S[j-1, k-1]

    Returns a list of ``(target_idx, ref_idx)`` pairs (in order, above *threshold*).
    """
    M, N = S.shape
    dp = np.full((M + 1, N + 1), -np.inf, dtype=np.float64)
    dp[0, :] = 0.0  # zero target lines aligned → score 0 for any ref prefix

    for j in range(1, M + 1):
        dp[j, 0] = 0.0  # all j target lines unmatched → score 0
        for k in range(1, N + 1):
            skip_target = dp[j - 1, k]
            skip_ref = dp[j, k - 1]
            match = dp[j - 1, k - 1] + float(S[j - 1, k - 1])
            dp[j, k] = max(skip_target, skip_ref, match)

    # Backtrack to recover matched pairs (prefer match > skip_target > skip_ref)
    matches: List[Tuple[int, int]] = []
    j, k = M, N
    while j > 0 and k > 0:
        match_val = dp[j - 1, k - 1] + float(S[j - 1, k - 1])
        if abs(dp[j, k] - match_val) < 1e-9:
            if float(S[j - 1, k - 1]) >= threshold:
                matches.append((j - 1, k - 1))
            j -= 1
            k -= 1
        elif abs(dp[j, k] - dp[j - 1, k]) < 1e-9:
            j -= 1
        else:
            k -= 1
    matches.reverse()
    return matches


def assign_timestamps(
    target_subs: list,
    ref_subs: list,
    matches: List[Tuple[int, int]],
) -> list:
    """Assign reference timestamps to target subtitles from matched pairs.

    - Matched target lines receive their corresponding reference start/end.
    - Multiple target lines matched to the same reference line share its
      time window divided equally.
    - Unmatched target lines get linearly interpolated timestamps from their
      nearest matched neighbours.

    Returns a new list of ``GenericSubtitle`` with updated start/end and
    the original (language-specific) inner content preserved.
    """
    from ffsubsync.generic_subtitles import GenericSubtitle  # avoid circular import

    M = len(target_subs)

    # Group target indices by the reference index they map to
    en_to_pl: Dict[int, List[int]] = {}
    for pl_idx, en_idx in matches:
        en_to_pl.setdefault(en_idx, []).append(pl_idx)

    # Compute exact start/end for every matched target line
    assigned_start: Dict[int, timedelta] = {}
    assigned_end: Dict[int, timedelta] = {}
    for en_idx, pl_indices in en_to_pl.items():
        en_sub = ref_subs[en_idx]
        total_dur = (en_sub.end - en_sub.start).total_seconds()
        seg_dur = total_dur / len(pl_indices)
        for i, pl_idx in enumerate(sorted(pl_indices)):
            assigned_start[pl_idx] = en_sub.start + timedelta(seconds=i * seg_dur)
            assigned_end[pl_idx] = en_sub.start + timedelta(seconds=(i + 1) * seg_dur)

    anchors = sorted(assigned_start.keys())

    result = []
    for j in range(M):
        pl_sub = target_subs[j]

        if j in assigned_start:
            new_start = assigned_start[j]
            new_end = assigned_end[j]
        else:
            prev_anchors = [a for a in anchors if a < j]
            next_anchors = [a for a in anchors if a > j]

            if prev_anchors and next_anchors:
                prev_a = prev_anchors[-1]
                next_a = next_anchors[0]
                gap = (assigned_start[next_a] - assigned_end[prev_a]).total_seconds()
                steps = next_a - prev_a
                step_dur = gap / steps
                off = j - prev_a
                new_start = assigned_end[prev_a] + timedelta(
                    seconds=(off - 1) * step_dur
                )
                new_end = assigned_end[prev_a] + timedelta(seconds=off * step_dur)
            elif prev_anchors:
                prev_a = prev_anchors[-1]
                step = timedelta(milliseconds=500)
                off = j - prev_a
                new_start = assigned_end[prev_a] + step * (off - 1)
                new_end = assigned_end[prev_a] + step * off
            elif next_anchors:
                next_a = next_anchors[0]
                step = timedelta(milliseconds=500)
                off = next_a - j
                new_start = assigned_start[next_a] - step * off
                new_end = assigned_start[next_a] - step * (off - 1)
            else:
                # No anchors at all — fall back to original timestamps
                new_start = pl_sub.start
                new_end = pl_sub.end

        result.append(GenericSubtitle(new_start, new_end, copy.deepcopy(pl_sub.inner)))

    return result


def align_subtitles_by_content(
    ref_subs,
    target_subs,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> list:
    """Assign reference SRT timestamps to target subtitles via cross-lingual alignment.

    Parameters
    ----------
    ref_subs:
        Reference subtitles with correct timestamps (e.g. English ``GenericSubtitle`` list).
    target_subs:
        Target subtitles whose timestamps need to be assigned (e.g. Polish OCR'd from PGS).
    model_name:
        Multilingual ``sentence-transformers`` model name.
    threshold:
        Minimum cosine similarity for a match to be accepted (default 0.3).

    Returns
    -------
    List of ``GenericSubtitle`` with reference timestamps and original target text.
    """
    ref_list = list(ref_subs)
    target_list = list(target_subs)

    logger.info(
        "aligning %d target subtitle(s) against %d reference subtitle(s)...",
        len(target_list),
        len(ref_list),
    )

    embedder = CrossLingualEmbedder(model_name)
    logger.info("embedding reference subtitles...")
    ref_embeddings = embedder.embed([sub.content for sub in ref_list])
    logger.info("embedding target subtitles...")
    target_embeddings = embedder.embed([sub.content for sub in target_list])

    logger.info("computing cross-lingual similarity matrix...")
    S = cosine_similarity_matrix(target_embeddings, ref_embeddings)

    logger.info("running monotone DP alignment...")
    matches = dp_monotone_align(S, threshold=threshold)

    matched_count = len(matches)
    match_ratio = matched_count / len(target_list) if target_list else 0.0
    logger.info(
        "matched %d / %d target lines (%.1f%%)",
        matched_count,
        len(target_list),
        match_ratio * 100,
    )
    if match_ratio < 0.4:
        logger.warning(
            "less than 40%% of lines matched (%.1f%%). This may indicate OCR quality "
            "issues or mismatched subtitle files.",
            match_ratio * 100,
        )

    return assign_timestamps(target_list, ref_list, matches)
