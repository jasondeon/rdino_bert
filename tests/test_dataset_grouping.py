import unittest

from dataset import (
    DiarizationSegment,
    SourceSpan,
    _primary_speaker_groups,
    _source_spans_for_window,
)


class PrimarySpeakerGroupingTests(unittest.TestCase):
    def test_unlabeled_gaps_are_preserved_within_primary_run(self) -> None:
        segments = [
            DiarizationSegment(1.0, 3.0, "subject"),
            DiarizationSegment(4.0, 6.0, "subject"),
            DiarizationSegment(7.0, 9.0, "interviewer"),
            DiarizationSegment(10.0, 12.0, "subject"),
            DiarizationSegment(14.0, 17.0, "subject"),
        ]

        primary, groups = _primary_speaker_groups(segments)

        self.assertEqual(primary, "subject")
        self.assertEqual(
            groups,
            [
                (SourceSpan(1.0, 6.0),),
                (SourceSpan(10.0, 17.0),),
            ],
        )

    def test_window_mapping_retains_the_continuous_gap(self) -> None:
        group = (SourceSpan(1.0, 6.0),)

        self.assertEqual(
            _source_spans_for_window(group, 1.0, 4.0),
            (SourceSpan(2.0, 5.0),),
        )

    def test_detected_other_speaker_is_a_hard_boundary(self) -> None:
        segments = [
            DiarizationSegment(0.0, 10.0, "subject"),
            DiarizationSegment(10.0, 11.0, "interviewer"),
            DiarizationSegment(11.0, 25.0, "subject"),
        ]

        _, groups = _primary_speaker_groups(segments)

        self.assertEqual(
            groups,
            [
                (SourceSpan(0.0, 10.0),),
                (SourceSpan(11.0, 25.0),),
            ],
        )

    def test_legacy_concatenation_policy_removes_unlabeled_gap(self) -> None:
        segments = [
            DiarizationSegment(1.0, 3.0, "subject"),
            DiarizationSegment(4.0, 6.0, "subject"),
        ]

        _, groups = _primary_speaker_groups(segments, gap_policy="concatenate")

        self.assertEqual(
            groups,
            [(SourceSpan(1.0, 3.0), SourceSpan(4.0, 6.0))],
        )


if __name__ == "__main__":
    unittest.main()
