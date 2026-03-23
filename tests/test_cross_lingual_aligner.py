# -*- coding: utf-8 -*-
import importlib.util
from datetime import timedelta

import numpy as np
import pytest
import srt

from ffsubsync.cross_lingual_aligner import (
    cosine_similarity_matrix,
    dp_monotone_align,
    assign_timestamps,
    compute_global_offset,
    apply_offset,
    align_subtitles_by_content,
)
from ffsubsync.generic_subtitles import GenericSubtitle


def _make_sub(index: int, start_s: float, end_s: float, text: str) -> GenericSubtitle:
    inner = srt.Subtitle(
        index=index,
        start=timedelta(seconds=start_s),
        end=timedelta(seconds=end_s),
        content=text,
    )
    return GenericSubtitle(timedelta(seconds=start_s), timedelta(seconds=end_s), inner)


# ---------------------------------------------------------------------------
# cosine_similarity_matrix
# ---------------------------------------------------------------------------


class TestCosineSimilarityMatrix:
    def test_shape(self):
        A = np.random.randn(3, 8).astype(np.float32)
        B = np.random.randn(5, 8).astype(np.float32)
        S = cosine_similarity_matrix(A, B)
        assert S.shape == (3, 5)

    def test_identical_vectors_give_one(self):
        A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        S = cosine_similarity_matrix(A, A)
        np.testing.assert_allclose(np.diag(S), [1.0, 1.0], atol=1e-5)

    def test_orthogonal_vectors_give_zero(self):
        A = np.array([[1.0, 0.0]], dtype=np.float32)
        B = np.array([[0.0, 1.0]], dtype=np.float32)
        S = cosine_similarity_matrix(A, B)
        np.testing.assert_allclose(S[0, 0], 0.0, atol=1e-5)

    def test_values_bounded(self):
        A = np.random.randn(10, 16).astype(np.float32)
        B = np.random.randn(10, 16).astype(np.float32)
        S = cosine_similarity_matrix(A, B)
        assert np.all(S >= -1.01) and np.all(S <= 1.01)


# ---------------------------------------------------------------------------
# dp_monotone_align
# ---------------------------------------------------------------------------


class TestDpMonotoneAlign:
    def test_perfect_diagonal(self):
        S = np.array(
            [[0.9, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.7]], dtype=np.float32
        )
        matches = dp_monotone_align(S, threshold=0.3)
        assert matches == [(0, 0), (1, 1), (2, 2)]

    def test_offset_match(self):
        S = np.array([[0.1, 0.9, 0.0], [0.1, 0.1, 0.8]], dtype=np.float32)
        matches = dp_monotone_align(S, threshold=0.3)
        assert matches == [(0, 1), (1, 2)]

    def test_below_threshold_excluded(self):
        S = np.array([[0.2, 0.0], [0.0, 0.2]], dtype=np.float32)
        matches = dp_monotone_align(S, threshold=0.3)
        assert matches == []

    def test_all_at_threshold_accepted(self):
        S = np.array([[0.3, 0.0], [0.0, 0.3]], dtype=np.float32)
        matches = dp_monotone_align(S, threshold=0.3)
        assert matches == [(0, 0), (1, 1)]

    def test_monotonicity_preserved(self):
        # Anti-diagonal: the DP must NOT emit a crossing pair.
        S = np.array([[0.0, 0.9], [0.9, 0.0]], dtype=np.float32)
        matches = dp_monotone_align(S, threshold=0.3)
        if len(matches) == 2:
            target_idxs = [m[0] for m in matches]
            ref_idxs = [m[1] for m in matches]
            assert target_idxs == sorted(target_idxs), "target indices must be sorted"
            assert ref_idxs == sorted(ref_idxs), "ref indices must be sorted"

    def test_empty_matrix(self):
        S = np.zeros((0, 5), dtype=np.float32)
        assert dp_monotone_align(S) == []

    def test_single_element_above_threshold(self):
        S = np.array([[0.8]], dtype=np.float32)
        matches = dp_monotone_align(S, threshold=0.3)
        assert matches == [(0, 0)]

    def test_more_target_than_ref(self):
        # 4 targets, 2 refs — only 2 can be matched
        S = np.zeros((4, 2), dtype=np.float32)
        S[1, 0] = 0.9
        S[3, 1] = 0.8
        matches = dp_monotone_align(S, threshold=0.3)
        assert matches == [(1, 0), (3, 1)]


# ---------------------------------------------------------------------------
# assign_timestamps
# ---------------------------------------------------------------------------


class TestAssignTimestamps:
    def _en(self):
        return [
            _make_sub(1, 1.0, 2.0, "Hello"),
            _make_sub(2, 3.0, 4.0, "World"),
            _make_sub(3, 5.0, 6.0, "Goodbye"),
        ]

    def _pl(self):
        return [
            _make_sub(1, 0.0, 0.5, "Cześć"),
            _make_sub(2, 0.0, 0.5, "Świat"),
            _make_sub(3, 0.0, 0.5, "Do widzenia"),
        ]

    def test_direct_1to1_match(self):
        result = assign_timestamps(self._pl(), self._en(), [(0, 0), (1, 1), (2, 2)])
        assert len(result) == 3
        assert result[0].start == timedelta(seconds=1.0)
        assert result[0].end == timedelta(seconds=2.0)
        assert result[1].start == timedelta(seconds=3.0)
        assert result[2].start == timedelta(seconds=5.0)

    def test_text_preserved(self):
        result = assign_timestamps(self._pl(), self._en(), [(0, 0), (1, 1), (2, 2)])
        assert result[0].content == "Cześć"
        assert result[1].content == "Świat"
        assert result[2].content == "Do widzenia"

    def test_unmatched_line_interpolated_between_anchors(self):
        # Match first and last; middle is interpolated.
        result = assign_timestamps(self._pl(), self._en(), [(0, 0), (2, 2)])
        assert len(result) == 3
        assert result[0].start == timedelta(seconds=1.0)
        assert result[2].start == timedelta(seconds=5.0)
        # Middle should sit between the two anchors
        assert result[1].start >= result[0].end
        assert result[1].end <= result[2].start

    def test_two_pl_to_one_en_subdivide(self):
        en = [_make_sub(1, 0.0, 2.0, "Hello World")]
        pl = [_make_sub(1, 0.0, 0.5, "Cześć"), _make_sub(2, 0.0, 0.5, "Świat")]
        result = assign_timestamps(pl, en, [(0, 0), (1, 0)])
        assert len(result) == 2
        assert result[0].start == timedelta(seconds=0.0)
        assert result[1].end == timedelta(seconds=2.0)
        # The two segments must be contiguous
        assert result[0].end == result[1].start

    def test_no_matches_keeps_original_timestamps(self):
        pl = self._pl()
        result = assign_timestamps(pl, self._en(), [])
        for orig, new in zip(pl, result):
            assert new.start == orig.start
            assert new.end == orig.end

    def test_unmatched_at_start_extrapolated_before_first_anchor(self):
        # Only last line is matched; first two should be extrapolated before it.
        result = assign_timestamps(self._pl(), self._en(), [(2, 2)])
        # All three should have start < 6.0 (the end of the last anchor)
        for sub in result:
            assert sub.end <= timedelta(seconds=6.0)


# ---------------------------------------------------------------------------
# compute_global_offset
# ---------------------------------------------------------------------------


class TestComputeGlobalOffset:
    def _en(self):
        return [
            _make_sub(1, 10.0, 11.0, "Hello"),
            _make_sub(2, 12.0, 13.0, "World"),
        ]

    def _pl(self):
        return [
            _make_sub(1, 8.0, 8.5, "Cześć"),
            _make_sub(2, 9.0, 9.5, "Świat"),
        ]

    def test_single_match_exact_offset(self):
        offset = compute_global_offset(self._pl(), self._en(), [(0, 0)])
        assert abs(offset.total_seconds() - 2.0) < 1e-6

    def test_two_matches_mean(self):
        # pair (0,0): 10.0 - 8.0 = 2.0 s; pair (1,1): 12.0 - 9.0 = 3.0 s
        # Only 2 samples → fallback to mean of all = 2.5
        offset = compute_global_offset(self._pl(), self._en(), [(0, 0), (1, 1)])
        assert abs(offset.total_seconds() - 2.5) < 1e-6

    def test_no_matches_returns_zero(self):
        offset = compute_global_offset(self._pl(), self._en(), [])
        assert offset == timedelta(0)


# ---------------------------------------------------------------------------
# apply_offset
# ---------------------------------------------------------------------------


class TestApplyOffset:
    def _subs(self):
        return [
            _make_sub(1, 1.0, 1.3, "Short"),
            _make_sub(2, 5.0, 6.5, "Normal"),
        ]

    def test_offset_applied_to_all(self):
        result = apply_offset(self._subs(), timedelta(seconds=3.0))
        assert result[0].start == timedelta(seconds=4.0)
        assert result[1].start == timedelta(seconds=8.0)

    def test_min_duration_clamped(self):
        # First sub is 0.3 s → should be extended to 1.0 s
        result = apply_offset(self._subs(), timedelta(0), min_duration_s=1.0)
        assert result[0].end - result[0].start == timedelta(seconds=1.0)
        # Second sub is already 1.5 s → untouched
        assert abs((result[1].end - result[1].start).total_seconds() - 1.5) < 1e-6

    def test_text_preserved(self):
        result = apply_offset(self._subs(), timedelta(seconds=1.0))
        assert result[0].content == "Short"
        assert result[1].content == "Normal"

    def test_negative_offset_clamped_to_zero(self):
        # Offset pushes start before 0 → clamp to 0
        result = apply_offset(self._subs(), timedelta(seconds=-5.0), min_duration_s=0)
        assert result[0].start == timedelta(0)

    def test_count_preserved(self):
        result = apply_offset(self._subs(), timedelta(seconds=2.0))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# align_subtitles_by_content (requires sentence-transformers)
# ---------------------------------------------------------------------------

_has_st = importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(not _has_st, reason="sentence-transformers not installed")
class TestAlignSubtitlesByContent:
    """Integration tests that load a real multilingual model."""

    _EN = [
        (1.0, 2.0, "Hello, how are you?"),
        (3.0, 4.0, "I am doing fine, thank you."),
        (5.0, 6.0, "Goodbye, see you later."),
    ]
    _PL_TEXTS = [
        "Cześć, jak się masz?",
        "Mam się dobrze, dziękuję.",
        "Do widzenia, do zobaczenia.",
    ]

    def _build_en(self):
        return [_make_sub(i + 1, s, e, t) for i, (s, e, t) in enumerate(self._EN)]

    def _build_pl(self):
        return [_make_sub(i + 1, 0.0, 0.5, t) for i, t in enumerate(self._PL_TEXTS)]

    def test_timestamps_assigned_within_reference_range(self):
        result = align_subtitles_by_content(self._build_en(), self._build_pl())
        assert len(result) == 3
        # Global offset applied: all subs should start within the reference window
        for sub in result:
            assert sub.start >= timedelta(seconds=0.0)
            assert sub.end >= timedelta(seconds=1.0)  # at least 1 s duration

    def test_minimum_duration_enforced(self):
        result = align_subtitles_by_content(self._build_en(), self._build_pl())
        for sub in result:
            duration = (sub.end - sub.start).total_seconds()
            assert duration >= 1.0, f"subtitle too short: {duration:.3f}s"

    def test_polish_text_preserved(self):
        result = align_subtitles_by_content(self._build_en(), self._build_pl())
        texts = [sub.content for sub in result]
        for pl_text in self._PL_TEXTS:
            assert pl_text in texts

    def test_chronological_order_preserved(self):
        result = align_subtitles_by_content(self._build_en(), self._build_pl())
        starts = [sub.start for sub in result]
        assert starts == sorted(starts)
