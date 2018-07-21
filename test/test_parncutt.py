#!/usr/bin/env python3
import re
import unittest
from didactyl.dactyler.Parncutt import Parncutt
from didactyl.dcorpus.DCorpus import DCorpus
import TestConstant

last_digit = {
    'A': 3,
    # B is cyclic pattern, which is not supported yet
    'C': 1,
    'D': 2,
    'E': 2,
    'F': 1,
    'G': 1
}

sma_lar_sums = {
    'A': {
        '>24342313': 1, '>13231213': 1, '>14342313': 3, '>35453423': 0, '>13231323': 4,
        '>13231423': 4, '>13242313': 3, '>12142313': 4, '>13232313': 3, '>24342323': 2
    },
    'C': {
        '>21241': 2, '>21251': 2, '>31241': 2, '>31251': 2, '>32351': 8,
        '>42351': 6, '>31351': 4, '>42151': 8, '>21351': 4, '>32151': 10
    },
    'D': {
        '>1323132': 4, '>1324132': 4, '>2412132': 3, '>1212132': 0, '>1213242': 2,
        '>1323142': 8, '>1323242': 4, '>2413242': 5, '>2435132': 6,'>1213232': 2
    },
    'E': {
        '>14523152': 8, '>14523142': 10, '>14512152': 10, '>14513152': 12, '>14515342': 14,
        '>14515352': 14, '>14524152': 13, '>13412152': 10, '>13523152': 14, '>14515452': 14
    },
    'F': {
        '>124251': 8, '>124151': 3, '>123151': 6, '>124141': 1, '>123141': 4,
        '>124131': 3, '>135141': 3, '>135251': 10, '>125251': 12, '>135131': 5
    },
    'G': {
        '>3432321': 0, '>2321321': 1, '>3431321': 2, '>2432321': 4, '>4542321': 2,
        '>2132321': 6, '>2321421': 5, '>2321431': 4, '>3132321': 8, '>3432421': 4
    }
}

subcosts = {
    'A': {
        '>24342313': {'wea': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13231213': {'wea': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>14342313': {'wea': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>35453423': {'wea': 4, '345': 3, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13231323': {'wea': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13231423': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13242313': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>12142313': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13232313': {'wea': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>24342323': {'wea': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0}
    },
    'C': {
        '>21241': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>21251': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>31241': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>31251': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>32351': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>42351': {'wea': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>31351': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>42151': {'wea': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>21351': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>32151': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
    },
    'D': {
        '>1323132': {'wea': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>1324132': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>2412132': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>1212132': {'wea': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 6, 'bl5': 0, 'pa1': 0},
        '>1213242': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 6, 'bl5': 0, 'pa1': 0},
        '>1323142': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>1323242': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>2413242': {'wea': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>2435132': {'wea': 2, '345': 1, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 2, 'pa1': 0},
        '>1213232': {'wea': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 6, 'bl5': 0, 'pa1': 0},
    },
    'E': {
        '>14523152': {'wea': 3, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>14523142': {'wea': 3, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>14512152': {'wea': 3, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>14513152': {'wea': 3, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>14515342': {'wea': 4, '345': 1, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>14515352': {'wea': 4, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>14524152': {'wea': 4, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>13412152': {'wea': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>13523152': {'wea': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>14515452': {'wea': 5, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>13415342': {'wea': 3, '345': 1, '3t4': 2, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
    },
    'F': {
        '>124251': {'wea': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>124151': {'wea': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>123151': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>124141': {'wea': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>123141': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>124131': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>135141': {'wea': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>135251': {'wea': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>125251': {'wea': 2, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>135131': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 5, 'bl5': 0, 'pa1': 0},
        '>123251': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
    },
    'G': {
        '>3432321': {'wea': 1, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>2321321': {'wea': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>3431321': {'wea': 1, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>2432321': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>4542321': {'wea': 3, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>2132321': {'wea': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>2321421': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>2321431': {'wea': 1, '345': 0, '3t4': 0, 'bl4': 1, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>3132321': {'wea': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>3432421': {'wea': 2, '345': 0, '3t4': 1, 'bl4': 0, 'bl1': 0, 'bl5': 0, 'pa1': 0},
        '>1321321': {'wea': 0, '345': 0, '3t4': 0, 'bl4': 0, 'bl1': 3, 'bl5': 0, 'pa1': 0},
        '>3432431': {'wea': 2, '345': 0, '3t4': 1, 'bl4': 1, 'bl1': 0, 'bl5': 0, 'pa1': 0},
    }
}

solutions = {
    'A': {
        6: {'>24342313': True},
        7: {'>13231213': True},
        8: {'>14342313': True, '>35453423': True},
        9: {'>13231323': True, '>13231423': True, '>13242313': True},
        10: {'>12142313': True, '>13232313': True, '>24342323': True}
    },
    # B is cyclic pattern, which is not supported yet
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


class ParncuttTest(unittest.TestCase):

    @staticmethod
    def test_something():
        parncutt = Parncutt()
        parncutt.segment_combination_method(method="cost")
        d_corpus = DCorpus(corpus_str=TestConstant.FOUR_NOTES)
        parncutt.load_corpus(d_corpus=d_corpus)
        suggestions, costs, details = parncutt.generate_advice(staff="upper", cycle=4, k=2)
        assert len(suggestions) == 2, "No loops in that dog in top ten"
        parncutt.report_on_advice(suggestions, costs, details)
        d_corpus = DCorpus(corpus_str=TestConstant.PARNCUTT_LOOP_FRAGMENT)
        parncutt.load_corpus(d_corpus=d_corpus)
        suggestions, costs, details = parncutt.generate_advice(staff="upper", cycle=4, k=16)
        assert len(suggestions) == 16, "There should be 16 cyclic fingerings!"
        parncutt.report_on_advice(suggestions, costs, details)

    # @staticmethod
    # def test_something():
    #     parncutt = Parncutt()
    #     parncutt.segment_combination_method(method="cost")
    #     d_corpus = DCorpus(corpus_str=TestConstant.FOUR_NOTES)
    #     parncutt.load_corpus(d_corpus=d_corpus)
    #     suggestions, costs, details = parncutt.generate_advice(staff="upper", k=2)
    #     assert suggestions, "Ahhh!"
    #
    # @staticmethod
    # def test_sma_lar():
    #     parncutt = Parncutt()
    #     parncutt.segment_combination_method(method="cost")
    #
    #     for id in sma_lar_sums:
    #         d_corpus = DCorpus(corpus_str=TestConstant.PARNCUTT_HUMAN_FRAGMENT[id])
    #         parncutt.load_corpus(d_corpus=d_corpus)
    #         suggestions, costs, details = parncutt.generate_advice(staff="upper", last_digit=last_digit[id], k=20)
    #         details_for_sugg = dict()
    #         for i in range(len(details)):
    #             details_for_sugg[suggestions[i]] = details[i][0]  # 0 index because we only have one segment
    #
    #         # parncutt.report_on_advice(suggestions, costs, details)
    #         for gold_sugg in sma_lar_sums[id]:
    #             assert gold_sugg in details_for_sugg,\
    #                 "Missing suggestion {0} in {1}".format(gold_sugg, id)
    #             sma = details_for_sugg[gold_sugg]['sma']
    #             lar = details_for_sugg[gold_sugg]['lar']
    #             assert sma + lar == sma_lar_sums[id][gold_sugg],\
    #                 "Bad sma + lar total for {0} in {1}".format(gold_sugg, id)
    #
    # @staticmethod
    # def test_bl1():
    #     parncutt = Parncutt()
    #     parncutt.segment_combination_method(method="cost")
    #     midi_1 = None
    #     handed_digit_1 = '-'
    #     midi_2 = 61
    #     handed_digit_2 = '>1'
    #     midi_3 = 65
    #     handed_digit_3 = '>3'
    #
    #     cost, costs = parncutt.trigram_node_cost(midi_1, handed_digit_1, midi_2, handed_digit_2, midi_3, handed_digit_3)
    #     assert costs['bl1'] == 3, "Bad bl1 cost"
    #
    # @staticmethod
    # def test_rules_6_through_12():
    #     parncutt = Parncutt()
    #     parncutt.segment_combination_method(method="cost")
    #
    #     for id in subcosts:
    #         d_corpus = DCorpus(corpus_str=TestConstant.PARNCUTT_HUMAN_FRAGMENT[id])
    #         parncutt.load_corpus(d_corpus=d_corpus)
    #         suggestions, costs, details = parncutt.generate_advice(staff="upper", last_digit=last_digit[id], k=30)
    #         details_for_sugg = dict()
    #         for i in range(len(details)):
    #             details_for_sugg[suggestions[i]] = details[i][0]  # 0 index because we only have one segment
    #
    #         # parncutt.report_on_advice(suggestions, costs, details)
    #         for gold_sugg in subcosts[id]:
    #             assert gold_sugg in details_for_sugg, \
    #                 "Missing suggestion {0} in {1}".format(gold_sugg, id)
    #             for rule in subcosts[id][gold_sugg]:
    #                 # if rule == 'bl1':
    #                     # continue
    #                 gold_cost = subcosts[id][gold_sugg][rule]
    #                 cost = details_for_sugg[gold_sugg][rule]
    #                 assert cost == gold_cost, \
    #                     "Bad {0} cost for {1} in {2}".format(rule, gold_sugg, id)
    #
    #     # for id in solutions:
    #     #     d_corpus = DCorpus(corpus_str=TestConstant.PARNCUTT_HUMAN_FRAGMENT[id])
    #     #     parncutt.load_corpus(d_corpus=d_corpus)
    #     #     suggestions, costs, details = parncutt.generate_advice(staff="upper", last_digit=last_digit[id], k=20)
    #     #     parncutt.report_on_advice(suggestions, costs, details)
    #     #     for i in range(10):
    #     #         assert costs[i] in solutions[id],\
    #     #             "Missing cost {0} in {1}".format(costs[i], id)
    #     #         assert suggestions[i] in solutions[id][int(costs[i])],\
    #     #             "Missing {0} cost suggestion {1} in {2}".format(costs[i], suggestions[i], id)
    #
    #
    # @staticmethod
    # def test_parncutt_edges():
    #     parncutt = Parncutt()
    #     d_corpus = DCorpus(corpus_str=TestConstant.ONE_NOTE)
    #     parncutt.load_corpus(d_corpus=d_corpus)
    #     upper_rh_advice = parncutt.advise(staff="upper")
    #     right_re = re.compile('^>\d$')
    #     assert right_re.match(upper_rh_advice), "Bad one-note, right-hand, upper-staff advice"
    #     # both_advice = parncutt.advise(staff="both")
    #     # both_re = re.compile('^>\d@$')
    #     # assert both_re.match(both_advice), "Bad one-note, segregated, both-staff advice"
    #
    #     parncutt = Parncutt()
    #     d_corpus = DCorpus(corpus_str=TestConstant.ONE_BLACK_NOTE_PER_STAFF)
    #     parncutt.load_corpus(d_corpus=d_corpus)
    #     upper_advice = parncutt.advise(staff="upper")
    #
    #     right_re = re.compile('^>2$')
    #     assert right_re.match(upper_advice), "Bad black-note, upper-staff advice"
    #     lower_advice = parncutt.advise(staff="lower")
    #     left_re = re.compile('^<2$')
    #     assert left_re.match(lower_advice), "Bad black-note, upper-staff advice"
    #     both_advice = parncutt.advise(staff="both")
    #     both_re = re.compile('^>2@<2$')
    #     assert both_re.match(both_advice), "Bad black-note, both-staff advice"
    #     lower_advice = parncutt.advise(staff="lower", first_digit=3)
    #     lower_re = re.compile('^<3$')
    #     assert lower_re.match(lower_advice), "Bad preset black-note, both-staff advice"
    #
    #     parncutt = Parncutt()
    #     d_corpus = DCorpus(corpus_str=TestConstant.TWO_WHITE_NOTES_PER_STAFF)
    #     parncutt.load_corpus(d_corpus=d_corpus)
    #     upper_advice = parncutt.advise(staff="upper")
    #     right_re = re.compile('^>\d\d$')
    #     assert right_re.match(upper_advice), "Bad two white-note, upper-staff advice"
    #     upper_advice = parncutt.advise(staff="upper", first_digit=2, last_digit=4)
    #     right_re = re.compile('^>24$')
    #     assert right_re.match(upper_advice), "Bad preset two white-note, upper-staff advice"
    #
    # @staticmethod
    # def test_distance_metrics():
    #     parncutt = Parncutt()
    #     d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE)
    #     parncutt.load_corpus(d_corpus=d_corpus)
    #     complete_rh_advice = parncutt.advise(staff="upper")
    #     complete_rh_advice_len = len(complete_rh_advice)
    #     right_re = re.compile('^>\d+$')
    #     assert right_re.match(complete_rh_advice), "Bad right-hand, upper-staff advice"
    #     rh_advice = parncutt.advise(staff="upper", offset=3, first_digit=4)
    #     short_advice_len = len(rh_advice)
    #     assert complete_rh_advice_len - 3 == short_advice_len, "Bad offset for advise() call"
    #     ff_re = re.compile('^>4\d+$')
    #     assert ff_re.match(rh_advice), "Bad first finger constraint"
    #
    #     rh_advice = parncutt.advise(staff="upper", offset=10, first_digit=1, last_digit=3)
    #     short_advice_len = len(rh_advice)
    #     assert complete_rh_advice_len - 10 == short_advice_len, "Bad offset for advise() call"
    #     ff_re = re.compile('^>1\d+3$')
    #     assert ff_re.match(rh_advice), "Bad first and last finger constraints"
    #
    #     lh_advice = parncutt.advise(staff="lower")
    #     left_re = re.compile('^<\d+$')
    #     assert left_re.match(lh_advice), "Bad left-hand, lower-staff advice"
    #     combo_advice = parncutt.advise(staff="both")
    #     clean_combo_advice = re.sub('[><&]', '',  combo_advice)
    #     d_score = d_corpus.d_score_by_index(index=0)
    #     gold_fingering = d_score.abcdf(index=0)
    #     clean_gold_fingering = re.sub('[><&]', '',  gold_fingering)
    #
    #     combo_re = re.compile('^>\d+@<\d+$')
    #     assert combo_re.match(combo_advice), "Bad combined advice"
    #     hamming_evaluations = parncutt.evaluate_strike_distance(method="hamming", staff="both")
    #     assert hamming_evaluations[0] > 0, "Undetected Hamming costs"
    #     assert hamming_evaluations[3] == 0, "Bad fish in Hamming barrel"
    #
    #     natural_evaluations = parncutt.evaluate_strike_distance(method="natural", staff="both")
    #     assert natural_evaluations[0] > 0, "Undetected natural costs"
    #     assert natural_evaluations[3] == 0, "Bad fish in natural barrel"
    #
    #     pivot_evaluations = parncutt.evaluate_strike_distance(method="pivot", staff="both")
    #     assert pivot_evaluations[0] > 0, "Undetected pivot costs"
    #     assert pivot_evaluations[3] == 0, "Bad fish in pivot barrel"
    #
    #
    # @staticmethod
    # def test_reentry():
    #     parncutt = Parncutt()
    #
    #     # We cannot use the longer example A_MAJ_SCALE because the gold standard fingering
    #     # requires hand repositionings not allowed by the Parncutt model. This reinforces the need
    #     # for segmentation and also (maybe) the need for a more inclusive option for Parncutt where
    #     # all paths are possible but some are just very expensive, as we have in Sayegh.
    #     d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE_SHORT)
    #     parncutt.load_corpus(d_corpus=d_corpus)
    #
    #     reentry_hamming_evals = parncutt.evaluate_strike_reentry(method="hamming", staff="upper", gold_indices=[2, 3])
    #     # Note we are not picking Beringer for the real gold standard because Beringer and Parncutt agree
    #     # on the fingering for this scale.
    #     # for rhe in reentry_hamming_evals:
    #         # print("RHE:{0}".format(rhe))
    #     assert reentry_hamming_evals[0] > 0, "Undetected upper Hamming reentry costs"
    #     assert reentry_hamming_evals[1] == 0, "Bad fish in upper-staff Hamming reentry barrel"
    #
    #     reentry_hamming_evals = parncutt.evaluate_strike_reentry(method="hamming", staff="both", gold_indices=[2, 3])
    #     # for rhe in reentry_hamming_evals:
    #         # print("RHE:{0}".format(rhe))
    #     assert reentry_hamming_evals[0] > 0, "Undetected both-staff Hamming reentry costs"
    #     assert reentry_hamming_evals[1] == 0, "Bad fish in both-staff Hamming reentry barrel"
    #     hamming_score = reentry_hamming_evals[0]
    #
    #     reentry_natural_evals = parncutt.evaluate_strike_reentry(method="natural", staff="both", gold_indices=[2, 3])
    #     # for rne in reentry_natural_evals:
    #         # print("RNE:{0}".format(rne))
    #     assert reentry_natural_evals[0] > 0, "Undetected natural reentry costs"
    #     assert reentry_natural_evals[1] == 0, "Bad fish in natural reentry barrel"
    #     natural_score = reentry_natural_evals[0]
    #     assert natural_score > hamming_score, "Reentry: Natural <= Hamming"
    #
    #     reentry_pivot_evals = parncutt.evaluate_strike_reentry(method="pivot", staff="both", gold_indices=[2, 3])
    #     # for rpe in reentry_pivot_evals:
    #         # print("RPE:{0}".format(rpe))
    #     assert reentry_pivot_evals[0] > 0, "Undetected pivot reentry costs"
    #     assert reentry_pivot_evals[1] == 0, "Bad fish in pivot reentry barrel"
    #     pivot_score = reentry_pivot_evals[0]
    #     assert natural_score < pivot_score, "Reentry: Natural >= Pivot"
    #
    # @staticmethod
    # def test_pivot_alignment():
    #     parncutt = Parncutt()
    #     d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE)
    #     parncutt.load_corpus(d_corpus=d_corpus)
    #
    #     evaluations = parncutt.evaluate_pivot_alignment(staff="both")
    #     # for he in hamming_evaluations:
    #     # print(he)
    #     assert evaluations[0] > 0, "Undetected pivot alignment costs"
    #     assert evaluations[3] == 0, "Bad fish in pivot alignment barrel"


if __name__ == "__main__":
    unittest.main()  # run all tests
