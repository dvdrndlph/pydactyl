#!/usr/bin/env python3
import re
import unittest
from pydactyl.dactyler.Parncutt import Jacobs, TrigramNode
from pydactyl.dcorpus.DCorpus import DCorpus
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter
import TestConstant

last_digit = {
    'A': 3,
    # B is cyclic pattern
    'C': 1,
    'D': 2,
    'E': 2,
    'F': 1,
    'G': 1
}

subcosts = {
    'A': {
        '>35453423': {'str': 0, 'sma': 0, 'larj': 0, 'weaj': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13231213': {'str': 0, 'sma': 1, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13231323': {'str': 0, 'sma': 1, 'larj': 1, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>24342313': {'str': 0, 'sma': 1, 'larj': 0, 'weaj': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>35453413': {'str': 0, 'sma': 3, 'larj': 0, 'weaj': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13231423': {'str': 0, 'sma': 3, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>15453423': {'str': 0, 'sma': 4, 'larj': 0, 'weaj': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>25453423': {'str': 0, 'sma': 4, 'larj': 0, 'weaj': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>45453423': {'str': 0, 'sma': 0, 'larj': 1, 'weaj': 3, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13231313': {'str': 0, 'sma': 2, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        # '>14341423': {'str': 0, 'sma': 7, 'larj': 0, 'weaj': 3, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        # '>14342413': {'str': 0, 'sma': 7, 'larj': 0, 'weaj': 3, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        # '>25352313': {'str': 0, 'sma': 13, 'lajr': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
    },
    # For Piece B, weak finger (weaj) costs in published Jacobs paper is one less than
    # our values when the first finger is a weak finger.
    # Bad wea cost for >4235 in B: 2 should be 1
    # Bad wea cost for >4215 in B: 2 should be 1
    # Bad wea cost for >4125 in B: 2 should be 1
    #
    # But all other costs are the same as ours. ???? FIXME ???? Is Parncutt including
    # the costs for the transition from the fourth note to the first or not? We are
    # applying all rule costs, as if an n+1 length sequence. For now, we just bump up
    # the wea cost for each sequence that has a weak finger (4 for JAcobs) on the first note.
    #
    # 345
    'B': {
        '>4235': {'str': 0, 'sma': 0, 'larj': 0, 'weaj': 2, '345': 1, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 4, 'pa1': 0},
        '>3214': {'str': 0, 'sma': 3, 'larj': 1, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 1, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>3124': {'str': 0, 'sma': 0, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 1, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>4215': {'str': 0, 'sma': 5, 'larj': 0, 'weaj': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 4, 'pa1': 0},
        '>3215': {'str': 0, 'sma': 7, 'larj': 1, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 4, 'pa1': 0},
        '>1235': {'str': 0, 'sma': 9, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 4, 'pa1': 0},
        '>3125': {'str': 0, 'sma': 4, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 4, 'pa1': 0},
        '>3235': {'str': 0, 'sma': 2, 'larj': 1, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 4, 'pa1': 0},
        '>2124': {'str': 0, 'sma': 2, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>4125': {'str': 0, 'sma': 4, 'larj': 0, 'weaj': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 4, 'pa1': 0},
        '>1234': {'str': 4, 'sma': 7, 'larj': 2, 'weaj': 1, '345': 1, '3t4': 1, 'bl4': 1, 'bl1': 0, 'bl5': 0, 'pa1': 0},
    },
    'C': {
        '>32351': {'str': 0, 'sma': 2, 'larj': 3, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>42351': {'str': 0, 'sma': 2, 'larj': 2, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>21241': {'str': 0, 'sma': 0, 'larj': 1, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>21251': {'str': 0, 'sma': 2, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>31241': {'str': 0, 'sma': 0, 'larj': 1, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>31251': {'str': 0, 'sma': 2, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>31351': {'str': 0, 'sma': 2, 'larj': 1, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>12351': {'str': 0, 'sma': 6, 'larj': 2, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>21351': {'str': 0, 'sma': 2, 'larj': 1, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>32151': {'str': 0, 'sma': 8, 'larj': 1, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>21231': {'str': 4, 'sma': 0, 'larj': 3, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
    },
    'D': {
        '>1323132': {'str': 0, 'sma': 2, 'larj': 1, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>1324132': {'str': 0, 'sma': 4, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>1323142': {'str': 0, 'sma': 6, 'larj': 1, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>2412132': {'str': 0, 'sma': 3, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>2435132': {'str': 0, 'sma': 6, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 2, 'pa1': 0},
        '>1212132': {'str': 0, 'sma': 0, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 6, 'bl5': 0, 'pa1': 0},
        '>1213242': {'str': 0, 'sma': 2, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 6, 'bl5': 0, 'pa1': 0},
        '>1323242': {'str': 0, 'sma': 2, 'larj': 1, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>2312132': {'str': 2, 'sma': 1, 'larj': 2, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>1213232': {'str': 0, 'sma': 0, 'larj': 1, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 6, 'bl5': 0, 'pa1': 0},
        '>2423132': {'str': 0, 'sma': 4, 'larj': 1, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
    },
    'E': {
        '>14523152': {'str': 0, 'sma': 6, 'larj': 2, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>14515342': {'str': 2, 'sma': 7, 'larj': 4, 'weaj': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>14515352': {'str': 0, 'sma': 11, 'larj': 2, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>14515452': {'str': 2, 'sma': 7, 'larj': 4, 'weaj': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>14523142': {'str': 2, 'sma': 4, 'larj': 4, 'weaj': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>14535342': {'str': 2, 'sma': 2, 'larj': 5, 'weaj': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>14524152': {'str': 0, 'sma': 9, 'larj': 2, 'weaj': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>14525342': {'str': 2, 'sma': 6, 'larj': 4, 'weaj': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>14525452': {'str': 2, 'sma': 6, 'larj': 4, 'weaj': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>14535452': {'str': 2, 'sma': 2, 'larj': 5, 'weaj': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13415342': {'str': 4, 'sma': 5, 'larj': 6, 'weaj': 2, '345': 0, '3t4': 2, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
    },
    'F': {
        '>124251': {'str': 2, 'sma': 2, 'larj': 3, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>123151': {'str': 2, 'sma': 2, 'larj': 2, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>123251': {'str': 6, 'sma': 2, 'larj': 7, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>124151': {'str': 0, 'sma': 3, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>135251': {'str': 2, 'sma': 4, 'larj': 3, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>125251': {'str': 2, 'sma': 6, 'larj': 3, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>235251': {'str': 2, 'sma': 4, 'larj': 4, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>123141': {'str': 2, 'sma': 0, 'larj': 2, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>124141': {'str': 0, 'sma': 1, 'larj': 0, 'weaj': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>134251': {'str': 6, 'sma': 2, 'larj': 5, 'weaj': 1, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
    },
    'G': {
        '>3432321': {'str': 0, 'sma': 0, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>2321321': {'str': 0, 'sma': 1, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>3431321': {'str': 0, 'sma': 2, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>4542321': {'str': 0, 'sma': 2, 'larj': 0, 'weaj': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>2432321': {'str': 0, 'sma': 4, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>4532321': {'str': 0, 'sma': 4, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>2132321': {'str': 0, 'sma': 6, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>2321421': {'str': 0, 'sma': 5, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>2321431': {'str': 0, 'sma': 4, 'larj': 0, 'weaj': 1, '345': 0, '3t4': 0, 'bl4': 1, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>3132321': {'str': 0, 'sma': 8, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>1321321': {'str': 0, 'sma': 3, 'larj': 0, 'weaj': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>3432431': {'str': 0, 'sma': 3, 'larj': 0, 'weaj': 2, '345': 0, '3t4': 1, 'bl4': 1, 'bl1': 0, 'bl5': 0, 'pa1': 0},
    }
}

subcost_names = ('str', 'sma', 'larj', 'pcc', 'pcs', 'weaj', '345', '3t4', 'bl4', 'bl1', 'bl5', 'pa1')

solutions = {
    'A': {
        6: {'>24342313': True},
        7: {'>13231213': True},
        8: {'>14342313': True, '>35453423': True},
        9: {'>13231323': True, '>13231423': True, '>13242313': True},
        10: {'>12142313': True, '>13232313': True, '>24342323': True}
    },
    'B': {
        7: {'>3124': True, '>3214': True, '>4235': True},
        11: {'>4215': True},
        14: {'>1235': True, '>2124': True, '>3125': True, '>3215': True},
        15: {'>3235': True, '>4125': True}
    },
    'C': {
        9: {'>21241': True, '>21251': True},
        10: {'>31241': True, '>31251': True, '>32351': True, '>42351': True},
        11: {'>31351': True, '>42151': True},
        12: {'>21351': True, '>32151': True}
    },
    'D': {
        12: {'>1323132': True, '>1324132': True},
        13: {'>2412132': True},
        14: {'>1212132': True, '>1213242': True, '>1323142': True},
        15: {'>1323242': True, '>2413242': True, '>2435132': True},
        16: {'>1213232': True}
    },
    'E': {
        24: {'>14523152': True},
        28: {'>14523142': True},
        31: {'>14512152': True},
        32: {'>14513152': True, '>14515342': True, '>14515352': True, '>14524152': True},
        33: {'>13412152': True, '>13523152': True, '>14515452': True}
    },
    'F': {
        17: {'>124251': True},
        19: {'>124151': True},
        20: {'>123151': True, '>124141': True},
        22: {'>123141': True, '>124131': True},
        23: {'>135141': True, '>135251': True},
        24: {'>125251': True, '>135131': True}
    },
    'G': {
        2: {'>3432321': True},
        3: {'>2321321': True},
        5: {'>3431321': True},
        6: {'>2432321': True, '>4542321': True},
        7: {'>2132321': True},
        8: {'>2321421': True, '>2321431': True, '>3132321': True, '>3432421': True}
    }
}


class JacobsTest(unittest.TestCase):
    def test_distance(self):
        jacobs = Jacobs(segmenter=ManualDSegmenter(), segment_combiner="cost")
        c4e4_dist = jacobs.distance(from_midi=60, to_midi=64)
        e4g4_dist = jacobs.distance(from_midi=64, to_midi=67)
        self.assertEqual(c4e4_dist, e4g4_dist, "Bad distance")
        e4c4_dist = jacobs.distance(from_midi=64, to_midi=60)
        g4e4_dist = jacobs.distance(from_midi=67, to_midi=64)
        self.assertEqual(e4c4_dist, g4e4_dist, "Bad negative distance")

    def test_four_note_example(self):
        jacobs = Jacobs(segmenter=ManualDSegmenter(), segment_combiner="cost")
        d_corpus = DCorpus(corpus_str=TestConstant.FOUR_NOTES)
        jacobs.load_corpus(d_corpus=d_corpus)
        suggestions, costs, details = jacobs.generate_advice(staff="upper", k=2)
        self.assertEqual(len(suggestions), 2, "No loops in that dog in top ten")
        # jacobs.report_on_advice(suggestions, costs, details)

    @staticmethod
    def test_cycles():
        jake = Jacobs()
        jake.segment_combiner(method="cost")
        d_corpus = DCorpus(corpus_str=TestConstant.FOUR_NOTES)
        jake.load_corpus(d_corpus=d_corpus)
        suggestions, costs, details = jake.generate_advice(staff="upper", cycle=4, k=2)
        assert len(suggestions) == 2, "No loops in that dog in top ten"
        # jake.report_on_advice(suggestions, costs, details)
        d_corpus = DCorpus(corpus_str=TestConstant.PARNCUTT_HUMAN_FRAGMENT['B'])
        jake.load_corpus(d_corpus=d_corpus)
        suggestions, costs, details = jake.generate_advice(staff="upper", cycle=4, k=16)
        assert len(suggestions) == 16, "There should be 16 cyclic fingerings!"
        # jake.report_on_advice(suggestions, costs, details)

    @staticmethod
    def test_fingering_counts():
        jake = Jacobs(pruning_method="none")
        jake.segment_combiner(method="cost")
        d_corpus = DCorpus(corpus_str=TestConstant.FOUR_NOTES)
        jake.load_corpus(d_corpus=d_corpus)
        suggestions, costs, details = jake.generate_advice(staff="upper", k=2)
        assert jake.last_segment_pruned_count() == 320, "Bad none pruning on open-ended problem"
        # suggestions, costs, details = jake.generate_advice(staff="upper", last_digit=5, k=10)
        # print(suggestions)
        # print(jake.last_segment_pruned_count())

    @staticmethod
    def test_bl1():
        jake = Jacobs()
        jake.segment_combiner(method="cost")
        midi_1 = None
        handed_digit_1 = '-'
        midi_2 = 61
        handed_digit_2 = '>1'
        midi_3 = 65
        handed_digit_3 = '>3'

        trigram_node = TrigramNode(midi_1, handed_digit_1, midi_2, handed_digit_2, midi_3, handed_digit_3)
        cost, costs = jake.trigram_node_cost(trigram_node=trigram_node)
        assert costs['bl1'] == 3, "Bad bl1 cost"

    @staticmethod
    def test_good_rules():
        jake = Jacobs()
        jake.segment_combiner(method="cost")

        for id in subcosts:
            d_corpus = DCorpus(corpus_str=TestConstant.PARNCUTT_HUMAN_FRAGMENT[id])
            jake.load_corpus(d_corpus=d_corpus)
            if id == 'B':
                suggestions, costs, details = jake.generate_advice(staff="upper", cycle=4, k=20)
            else:
                suggestions, costs, details = jake.generate_advice(staff="upper", last_digit=last_digit[id], k=30)
            details_for_sugg = dict()
            for i in range(len(details)):
                details_for_sugg[suggestions[i]] = details[i][0]  # 0 index because we only have one segment

            jake.report_on_advice(suggestions, costs, details)
            for gold_sugg in subcosts[id]:
                assert gold_sugg in details_for_sugg, \
                    "Missing suggestion {0} in {1}".format(gold_sugg, id)
                for rule in subcosts[id][gold_sugg]:
                    if rule == '345':
                        continue
                    gold_cost = subcosts[id][gold_sugg][rule]
                    cost = details_for_sugg[gold_sugg][rule]
                    assert cost == gold_cost, \
                        "Bad {0} cost for {1} in {2}: {3} should be {4}".format(rule, gold_sugg, id, cost, gold_cost)

    # # @staticmethod
    # # def test_corrected_jake():
    # #     jake = Jacobs(pruning_method="none")
    # #     jake.segment_combiner(method="cost")
    # #
    # #     for id in subcosts:
    # #         if id != 'B':
    # #             continue
    # #         d_corpus = DCorpus(corpus_str=TestConstant.PARNCUTT_HUMAN_FRAGMENT[id])
    # #         jake.load_corpus(d_corpus=d_corpus)
    # #         if id == 'B':
    # #             suggestions, costs, details = jake.generate_advice(staff="upper", cycle=4, k=500)
    # #         else:
    # #             suggestions, costs, details = jake.generate_advice(staff="upper", last_digit=last_digit[id], k=500)
    # #         playable_count = jake.last_segment_pruned_count()
    #         # jake.report_on_advice(suggestions, costs, details)
    #
    #         # Output for LaTeX report.
    #         # for i in range(len(details)):
    #         #     sugg = suggestions[i]
    #         #     cost = costs[i]
    #         #     subcosts_for_sugg = details[i][0]  # 0 index because we only have one segment
    #         #     record_str = "{0}&{1}&".format(i+1, sugg[1:])
    #         #     for subcost_name in subcost_names:
    #         #         record_str += str(subcosts_for_sugg[subcost_name]) + '&'
    #         #     record_str += str(int(cost)) + "\\\\"
    #         #     print(record_str)
    #         # print("Piece {0}, playable fingerings: {1}".format(id, playable_count))
    #
    #     # The solutions from original Parncutt paper are inconsistent for painfully well-documented reasons.
    #     # Leave the following commented out.
    #     # for id in solutions:
    #     #     d_corpus = DCorpus(corpus_str=TestConstant.PARNCUTT_HUMAN_FRAGMENT[id])
    #     #     jake.load_corpus(d_corpus=d_corpus)
    #     #     suggestions, costs, details = jake.generate_advice(staff="upper", last_digit=last_digit[id], k=20)
    #     #     jake.report_on_advice(suggestions, costs, details)
    #     #     for i in range(10):
    #     #         assert costs[i] in solutions[id],\
    #     #             "Missing cost {0} in {1}".format(costs[i], id)
    #     #         assert suggestions[i] in solutions[id][int(costs[i])],\
    #     #             "Missing {0} cost suggestion {1} in {2}".format(costs[i], suggestions[i], id)
    #

    @staticmethod
    def test_jake():
        jake = Jacobs()
        d_corpus = DCorpus(corpus_str=TestConstant.ONE_NOTE)
        jake.load_corpus(d_corpus=d_corpus)
        upper_rh_advice = jake.advise(staff="upper")
        right_re = re.compile('^>\d$')
        assert right_re.match(upper_rh_advice), "Bad one-note, right-hand, upper-staff advice"
        # both_advice = jake.advise(staff="both")
        # both_re = re.compile('^>\d@$')
        # assert both_re.match(both_advice), "Bad one-note, segregated, both-staff advice"

        jake = Jacobs()
        d_corpus = DCorpus(corpus_str=TestConstant.ONE_BLACK_NOTE_PER_STAFF)
        jake.load_corpus(d_corpus=d_corpus)
        upper_advice = jake.advise(staff="upper")

        right_re = re.compile('^>2$')
        assert right_re.match(upper_advice), "Bad black-note, upper-staff advice"
        lower_advice = jake.advise(staff="lower")
        left_re = re.compile('^<2$')
        assert left_re.match(lower_advice), "Bad black-note, upper-staff advice"
        both_advice = jake.advise(staff="both")
        both_re = re.compile('^>2@<2$')
        assert both_re.match(both_advice), "Bad black-note, both-staff advice"
        lower_advice = jake.advise(staff="lower", first_digit=3)
        lower_re = re.compile('^<3$')
        assert lower_re.match(lower_advice), "Bad preset black-note, both-staff advice"

        jake = Jacobs()
        d_corpus = DCorpus(corpus_str=TestConstant.TWO_WHITE_NOTES_PER_STAFF)
        jake.load_corpus(d_corpus=d_corpus)
        upper_advice = jake.advise(staff="upper")
        right_re = re.compile('^>\d\d$')
        assert right_re.match(upper_advice), "Bad two white-note, upper-staff advice"
        upper_advice = jake.advise(staff="upper", first_digit=2, last_digit=4)
        right_re = re.compile('^>24$')
        assert right_re.match(upper_advice), "Bad preset two white-note, upper-staff advice"

    @staticmethod
    def test_distance_metrics():
        jake = Jacobs()
        d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE_SHORT)
        jake.load_corpus(d_corpus=d_corpus)
        complete_rh_advice = jake.advise(staff="upper")
        complete_rh_advice_len = len(complete_rh_advice)
        right_re = re.compile('^>\d+$')
        assert right_re.match(complete_rh_advice), "Bad right-hand, upper-staff advice"
        rh_advice = jake.advise(staff="upper", offset=3, first_digit=4)
        short_advice_len = len(rh_advice)
        assert complete_rh_advice_len - 3 == short_advice_len, "Bad offset for advise() call"
        ff_re = re.compile('^>4\d+$')
        assert ff_re.match(rh_advice), "Bad first finger constraint"

        rh_advice = jake.advise(staff="upper", offset=10, first_digit=1, last_digit=3)
        short_advice_len = len(rh_advice)
        assert complete_rh_advice_len - 10 == short_advice_len, "Bad offset for advise() call"
        ff_re = re.compile('^>1\d+3$')
        assert ff_re.match(rh_advice), "Bad first and last finger constraints"

        lh_advice = jake.advise(staff="lower")
        left_re = re.compile('^<\d+$')
        assert left_re.match(lh_advice), "Bad left-hand, lower-staff advice"
        combo_advice = jake.advise(staff="both")
        clean_combo_advice = re.sub('[><&]', '',  combo_advice)
        d_score = d_corpus.d_score_by_index(index=0)
        gold_fingering = d_score.abcdf(index=0)
        clean_gold_fingering = re.sub('[><&]', '',  gold_fingering)

        combo_re = re.compile('^>\d+@<\d+$')
        assert combo_re.match(combo_advice), "Bad combined advice"
        hamming_evaluations = jake.evaluate_strike_distance(method="hamming", staff="both")
        assert hamming_evaluations[0] > 0, "Undetected Hamming costs"
        assert hamming_evaluations[4] == 0, "Bad fish in Hamming barrel"

        natural_evaluations = jake.evaluate_strike_distance(method="natural", staff="both")
        assert natural_evaluations[0] > 0, "Undetected natural costs"
        assert natural_evaluations[4] == 0, "Bad fish in natural barrel"

        pivot_evaluations = jake.evaluate_strike_distance(method="pivot", staff="both")
        assert pivot_evaluations[0] > 0, "Undetected pivot costs"
        assert pivot_evaluations[4] == 0, "Bad fish in pivot barrel"

        # suggestions, costs, details = jake.generate_advice(staff="upper", k=2)
        # jake.report_on_advice(suggestions, costs, details)
        # suggestions, costs, details = jake.generate_advice(staff="lower", k=2)
        # jake.report_on_advice(suggestions, costs, details)

    @staticmethod
    def test_reentry():
        jake = Jacobs()

        # We cannot use the longer example A_MAJ_SCALE because the gold standard fingering
        # requires hand repositionings not allowed by the Parncutt model. This reinforces the need
        # for segmentation and also (maybe) the need for a more inclusive option for Parncutt where
        # all paths are possible but some are just very expensive, as we have in Sayegh.
        d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE_SHORT)
        jake.load_corpus(d_corpus=d_corpus)

        reentry_hamming_evals = jake.evaluate_strike_reentry(method="hamming", staff="upper", gold_indices=[2, 3])
        # Note we are not picking Beringer for the real gold standard because Beringer and Parncutt agree
        # on the fingering for this scale.
        # for rhe in reentry_hamming_evals:
            # print("RHE:{0}".format(rhe))
        assert reentry_hamming_evals[0] > 0, "Undetected upper Hamming reentry costs"
        assert reentry_hamming_evals[1] == 0, "Bad fish in upper-staff Hamming reentry barrel"

        reentry_hamming_evals = jake.evaluate_strike_reentry(method="hamming", staff="both", gold_indices=[2, 3])
        # for rhe in reentry_hamming_evals:
            # print("RHE:{0}".format(rhe))
        assert reentry_hamming_evals[0] > 0, "Undetected both-staff Hamming reentry costs"
        assert reentry_hamming_evals[1] == 0, "Bad fish in both-staff Hamming reentry barrel"
        hamming_score = reentry_hamming_evals[0]

        reentry_natural_evals = jake.evaluate_strike_reentry(method="natural", staff="both", gold_indices=[2, 3])
        # for rne in reentry_natural_evals:
            # print("RNE:{0}".format(rne))
        assert reentry_natural_evals[0] > 0, "Undetected natural reentry costs"
        assert reentry_natural_evals[1] == 0, "Bad fish in natural reentry barrel"
        natural_score = reentry_natural_evals[0]
        assert natural_score > hamming_score, "Reentry: Natural <= Hamming"

        reentry_pivot_evals = jake.evaluate_strike_reentry(method="pivot", staff="both", gold_indices=[2, 3])
        # for rpe in reentry_pivot_evals:
            # print("RPE:{0}".format(rpe))
        assert reentry_pivot_evals[0] > 0, "Undetected pivot reentry costs"
        assert reentry_pivot_evals[1] == 0, "Bad fish in pivot reentry barrel"
        pivot_score = reentry_pivot_evals[0]
        assert natural_score < pivot_score, "Reentry: Natural >= Pivot"

    @staticmethod
    def test_pivot_alignment():
        jake = Jacobs()
        d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE_SHORT)
        jake.load_corpus(d_corpus=d_corpus)

        evaluations = jake.evaluate_pivot_alignment(staff="both")
        # for he in hamming_evaluations:
        # print(he)
        assert evaluations[0] > 0, "Undetected pivot alignment costs"
        assert evaluations[4] == 0, "Bad fish in pivot alignment barrel"


if __name__ == "__main__":
    unittest.main()  # run all tests
