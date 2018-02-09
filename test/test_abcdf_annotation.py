#!/usr/bin/env python3

import unittest
from DCorpus.DCorpus import ABCDHeader, ABCDAnnotation


class ABCDAnnotationTest(unittest.TestCase):
    @staticmethod
    def test_parse():
        annot = Hart()
        hart.load_corpus()
        hart.advise()
        assert 1 == 2, "Something happened"


if __name__ == "__main__":
    unittest.main()  # run all tests
