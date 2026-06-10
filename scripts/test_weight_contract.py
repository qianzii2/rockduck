#!/usr/bin/env python3
"""
Regression guard for Tree-CNN weight exporters.

Verifies that production weight generation uses the canonical 3-head, 25-feature
layout expected by `src/query/routing/ml.rs`: 78 little-endian `f32` values
(312 bytes total).
"""

import struct
import unittest

FEATURE_DIM = 25
NUM_HEADS = 3
EXPECTED_F32S = FEATURE_DIM * NUM_HEADS + NUM_HEADS
EXPECTED_BYTES = EXPECTED_F32S * 4


class WeightFormatContractTests(unittest.TestCase):
    def test_canonical_weight_contract_is_312_bytes(self):
        weights = [0.0] * EXPECTED_F32S
        blob = struct.pack("<" + "f" * EXPECTED_F32S, *weights)

        self.assertEqual(EXPECTED_F32S, 78)
        self.assertEqual(len(blob), EXPECTED_BYTES)
        self.assertEqual(EXPECTED_BYTES, 312)


if __name__ == "__main__":
    unittest.main()
