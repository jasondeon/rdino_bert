import math
import unittest

import numpy as np

from evaluation_metrics import intraclass_correlation_2_1


class IntraclassCorrelationTests(unittest.TestCase):
    def test_perfect_absolute_agreement(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(intraclass_correlation_2_1(values, values), 1.0)

    def test_constant_offset_penalizes_absolute_agreement(self) -> None:
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = observed + 1.0
        self.assertAlmostEqual(
            intraclass_correlation_2_1(observed, predicted), 10.0 / 13.0
        )

    def test_constant_scores_are_undefined(self) -> None:
        values = np.ones(4)
        self.assertTrue(math.isnan(intraclass_correlation_2_1(values, values)))

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            intraclass_correlation_2_1([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
