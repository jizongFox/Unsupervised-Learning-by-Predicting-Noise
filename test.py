# test each component
import unittest
from unittest import TestCase

import torch

from _utils import calc_optimal_target_permutation, _pairwise_distances


class Test_calcult_target(TestCase):

    def test(self):
        feature = torch.randn(10, 100)
        target = torch.randn(10, 100)
        new_target = calc_optimal_target_permutation(feature, target)
        assert _pairwise_distances(feature, target).trace() > _pairwise_distances(feature, new_target).trace()


if __name__ == "__main__":
    unittest.main()
