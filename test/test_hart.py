#!/usr/bin/env python3

import unittest
from Dactyler.Hart import Hart


class HartTest(unittest.TestCase):
    @staticmethod
    def test_hart():
        hart = Hart()
        hart.load_corpus()
        hart.advise()
        assert 1 == 2, "Something happened"


if __name__ == "__main__":
    unittest.main()  # run all tests
