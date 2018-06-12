__author__ = 'David Randolph'
# Copyright (c) 2014-2018 David A. Randolph.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
from abc import ABC, abstractmethod
import pickle
import re
import copy
import networkx as nx
from itertools import islice
from datetime import datetime
from didactyl.dactyler import Constant
from didactyl.dcorpus.DCorpus import DCorpus
from didactyl.dcorpus.DNote import AnnotatedDNote
from didactyl.dcorpus.DAnnotation import DAnnotation
import os


class Dactyler(ABC):
    """Base class for all fingering algorithms."""

    # FIXME: The log should be timestamped and for the specific algorithm being used.
    SQUAWK_OUT_LOUD = False
    DELETE_LOG = True

    def __init__(self, segmenting=False):
        self._d_corpus = None
        timestamp = datetime.now().isoformat()
        self._log_file_path = '/tmp/dactyler_' + self.__class__.__name__ + '_' + timestamp + '.log'
        self._log = open(self._log_file_path, 'a')
        self._segmenting = segmenting

        # One of "cost," "normal," or "rank"
        self._segment_combination_method = "normal"
        self._staff_combination_method = "naive"

    def __del__(self):
        self._log.close()
        if Dactyler.DELETE_LOG:
            os.remove(self._log_file_path)

    def segmenting(self, segmenting=None):
        if segmenting is None:
            return self._segmenting
        self._segmenting = segmenting

    def segment_combination_method(self, method=None):
        if method is None:
            return self._segment_combination_method
        self._segment_combination_method = method

    def staff_combination_method(self, method=None):
        if method is None:
            return self._staff_combination_method
        self._staff_combination_method = method

    @staticmethod
    def combine_abcdf_segments(segments):
        """
        Combine the array of abcDF strings into a single, simplified abcDF string.
        """
        abcdf = ''
        current_hand = None
        for seg in segments:
            for ch in seg:
                if ch == "<" or ch == ">":
                    if ch != current_hand:
                        abcdf += ch
                        current_hand = ch
                else:
                    abcdf += ch
        return abcdf

    def combine_staves(self, upper_suggestions, upper_costs, lower_suggestions, lower_costs, upper_length, lower_length, k=1):
        """
        Apply the staff_combination_method to combine solutions for upper and lower staves.
        :param upper_suggestions:
        :param upper_costs:
        :param lower_suggestions:
        :param lower_costs:
        :param upper_length:
        :param lower_length:
        :param k:
        :return:
        """
        # FIXME
        suggestions = list()
        costs = list()
        if self.staff_combination_method() == "naive":
            if len(upper_suggestions) == len(lower_suggestions):
                for i in range(len(upper_suggestions)):
                    suggestions.append(upper_suggestions[i] + "@" + lower_suggestions[i])
                    costs.append(upper_costs[i] + lower_costs[i])
            elif len(upper_suggestions) < len(lower_suggestions):
                for lower_index in range(len(lower_suggestions)):
                    upper_index = lower_index // len(lower_suggestions)
                    suggestions.append(upper_suggestions[upper_index] + "@" + lower_suggestions[lower_index])
                    costs.append(upper_costs[upper_index] + lower_costs[lower_index])
            else:
                for upper_index in range(len(upper_suggestions)):
                    lower_index = upper_index // len(upper_suggestions)
                    suggestions.append(upper_suggestions[upper_index] + "@" + lower_suggestions[lower_index])
                    costs.append(upper_costs[upper_index] + lower_costs[lower_index])
        return suggestions, costs

    def combine_segments(self, suggestions_for_segment, costs_for_segment, segment_lengths, k=1):
        """
        Given a list of fingering suggestions for an ordered list of fragments in an entire score,
        combine them in the k-best ways. This reduces to a k-shortest path search problem, where the arc costs
        are simply the costs associated with the suggested segment fingering. How these segment fingering costs
        are calculated affects the rankings of the complete end-to-end suggestions returned. The simplest method is
        perhaps the truest--we just use the cost reported for the segment in question. The cost of the entire
        sequence should be the sum of the individual segment costs. However, this will make the most difficult and
        longest segment fingerings the most consistent in the returned solutions. These may be the very segments
        for which a user would appreciate more suggestions. These would also be the segments that would have less
        agreement among pianists. It might be better to normalize the cost by the number of notes in the segment
        or even to only look at the ordinal rank of segment suggestions, so we see a similar amount of variability
        for each segment. These three costing approaches are supported here, and must be specified in via the
        self.segment_combination_method as one of "cost," "normal," or "rank."
        :param suggestions_for_segment: A list of lists of suggested fingerings for each segment.
        :param costs_for_segment: A corresponding list of lists of costs for each suggestion.
        :param segment_lengths: A corresponding list of lists of the lengths of each segment. Used to
        normalize the contribution of a segment's contribution to the total cost.
        :param k: The number of advice segments to return. The actual number returned may be less,
        but will be no more, than this number.
        :return: suggestions, costs: Two lists are returned. The first contains suggested fingering
        solutions as abcDF strings. The second list contains the respective costs of each suggestion.
        """
        g = nx.DiGraph()
        g.add_node(0, suggestion=None)
        prior_slice_node_ids = list()
        prior_slice_node_ids.append(0)
        node_id = 1
        for segment_index in range(len(suggestions_for_segment)):
            slice_node_ids = list()
            suggestions = suggestions_for_segment[segment_index]
            costs = costs_for_segment[segment_index]
            for suggestion_id in range(len(suggestions)):
                g.add_node(node_id, suggestion=suggestions[suggestion_id])
                for prior_node_id in prior_slice_node_ids:
                    if self._segment_combination_method == 'cost':
                        cost = costs[suggestion_id]
                    elif self._segment_combination_method == 'normal':
                        cost = costs[suggestion_id] / segment_lengths[segment_index]
                    elif self._segment_combination_method == 'rank':
                        cost = suggestion_id
                    else:
                        raise Exception("Unsupported method: {0}".format(self._segment_combination_method))
                    g.add_edge(prior_node_id, node_id, weight=cost)
                slice_node_ids.append(node_id)
                node_id += 1
            if len(slice_node_ids) > 0:
                prior_slice_node_ids = copy.copy(slice_node_ids)

        g.add_node(node_id, suggestion=None)
        for prior_node_id in prior_slice_node_ids:
            g.add_edge(prior_node_id, node_id, cost=0)

        if k is None or k == 1:
            path = nx.shortest_path(g, source=0, target=node_id, weight="weight")
            abcdf_segments = list()
            for node_id in path:
                node = g.nodes[node_id]
                if node["suggestion"]:
                    abcdf_segments.append(node["suggestion"])
            suggestion = Dactyler.combine_abcdf_segments(abcdf_segments)
            cost = nx.shortest_path_length(g, source=0, target=node_id, weight="weight")
            return [suggestion], [cost]
        else:
            suggestions = list()
            costs = list()
            k_best_paths = list(islice(nx.shortest_simple_paths(g, source=0, target=node_id, weight="weight"), k))
            for path in k_best_paths:
                sub_g = g.subgraph(path)
                cost = sub_g.size(weight="weight")
                abcdf_segments = list()
                for node_id in path:
                    node = g.nodes[node_id]
                    if node["suggestion"]:
                        abcdf_segments.append(node["suggestion"])
                suggestion = Dactyler.combine_abcdf_segments(abcdf_segments)
                suggestions.append(suggestion)
                costs.append(cost)
            return suggestions, costs

    @staticmethod
    def hand_digit(digit, staff):
        """
        Determine the handed digit for the input digit string.
        If the string already has an assigned digit, just return it.
        Otherwise assign the default hand for the specified staff.
        :return: The abcDF hand digit (">2" or "<3").
        """
        if digit is None:
            return None

        handed_re = re.compile('^[<>]\d$')
        if handed_re.match(str(digit)):
            return digit

        staff_prefix = ">"
        if staff == "lower":
            staff_prefix = "<"
        handed_digit = staff_prefix + str(digit)
        return handed_digit

    @staticmethod
    def digit_hand(handed_digit):
        """
        Determine the hand specified in the handed digit.
        :return: The abcDF hand specifier (">" or "<").
        """
        handed_re = re.compile('^([<>]{1})\d$')
        mat = handed_re.match(str(handed_digit))
        hand = mat.group(1)
        if hand != "<" and hand != ">":
            raise Exception("Ill-formed handed digit: {0}".format(handed_digit))
        return hand

    def squawk(self, msg):
        self._log.write(str(msg) + "\n")
        if Dactyler.SQUAWK_OUT_LOUD:
            print(str(msg))

    def squeak(self, msg):
        self._log.write(str(msg))
        if Dactyler.SQUAWK_OUT_LOUD:
            print(str(msg), end="")

    @staticmethod
    def one_note_advise(d_note, staff="upper", first_digit=None, last_digit=None):
        """
        Dispense advice for the degenerate case where we have only one note to consider in isolation.
        Unless otherwise constrained, we prefer the thumb for a white key and an index finger
        for a black key.
        :param d_note: The note to consider.
        :param staff: The staff containing the note.
        :param first_digit: The digit to use is constrained to be this.
        :param last_digit: The digit is constrained to be this.
        :return: Advice in the form of a handed digit (e.g., ">2").
        """
        if staff != "upper" and staff != "lower":
            raise Exception("One note advice not available for {0} staff.".format(staff))
        if first_digit and last_digit and first_digit != last_digit:
            raise Exception("Ambiguous digit constraint: {0} and {1}".format(first_digit, last_digit))

        if staff == "upper":
            advice = ">"
        else:
            advice = "<"

        digit = "1"
        if first_digit:
            digit = str(first_digit)
        elif last_digit:
            digit = last_digit
        elif d_note.is_black():
            digit = "2"
        advice += digit

        return advice

    @abstractmethod
    def generate_segment_advice(self, segment, staff, offset, handed_first_digit=None, handed_last_digit=None, k=1):
        """
        Abstract method must be implemented by derived classes to generate a set of up to k ranked fingering
        suggestions for the given segment.
        :param segment: The segment to work with, as a music21 score object.
        :param staff: The staff (one of "upper" or "lower") from which the segment was derived.
        :param offset: The zero-based index to begin the returned advice.
        :param handed_first_digit: Constrain the solution to begin with this finger.
        :param handed_last_digit: Constrain the solution to end with this finger.
        :param k: The number of advice segments to return. The actual number returned may be less,
        but will be no more, than this number.
        :return: suggestions, costs: Two lists are returned. The first contains suggested fingering
        solutions as abcDF strings. The second list contains the respective costs of each suggestion.
        """
        return list(), list()

    @staticmethod
    def generate_standard_graph_advice(g, target_id, k):
        """
        Apply standard shortest path algorithms to determine set of optimal fingerings based on
        a standardized networkx graph.
        :param g: The weighted graph. Weights must be specified via a "weight" edge parameter. Fingerings
        must be set on each "handed_digit" node parameter.
        :param target_id: The node id (key) for the last node or end point in the graph.
        :param k: The number of
        :return: suggestions, costs: Two lists are returned. The first contains suggested fingering
        solutions as abcDF strings. The second list contains the respective costs of each suggestion.
        """
        if k is None or k == 1:
            path = nx.shortest_path(g, source=0, target=target_id, weight="weight")
            segment_abcdf = ''
            for node_id in path:
                node = g.nodes[node_id]
                if node["handed_digit"]:
                    segment_abcdf += node["handed_digit"]
            cost = nx.shortest_path_length(g, source=0, target=target_id, weight="weight")
            return [segment_abcdf], [cost]
        else:
            sugg_map = dict()
            suggestions = list()
            costs = list()
            k_best_paths = list(islice(nx.shortest_simple_paths(g, source=0, target=target_id, weight="weight"), k))
            for path in k_best_paths:
                sub_g = g.subgraph(path)
                suggestion_cost = sub_g.size(weight="weight")
                # print("SUBGRAPH COST: {0}".format(suggestion_cost))
                # for (from_id, to_id) in sub_g.edges:
                #     edge_weight = sub_g[from_id][to_id]['weight']
                #     print("{0} cost for edge ({1}, {2})".format(edge_weight, from_id, to_id))
                segment_abcdf = ''
                for node_id in path:
                    node = g.nodes[node_id]
                    if node["handed_digit"]:
                        segment_abcdf += node["handed_digit"]
                suggestions.append(segment_abcdf)
                if segment_abcdf in sugg_map:
                    sugg_map[segment_abcdf] += 1
                else:
                    sugg_map[segment_abcdf] = 1
                costs.append(suggestion_cost)

            print("TOTAL: {0} DISTINCT: {1}".format(len(suggestions), len(sugg_map)))
            return suggestions, costs

    def generate_advice(self, score_index=0, staff="upper", offset=0, first_digit=None, last_digit=None, k=1):
        """
        Generate advice for the specified score. This method only supports segregated advice.
        :param score_index:
        :param staff: One of "upper," "lower," or "both." Note that requesting "both" staves will result in
        the pairwise combination of k suggestions from upper and lower staves and the conflation of the staff
        costs if the default "naive" staff_combination_method is used. We only guarantees that the first suggestion
        returned is globally optimal. Additional combination methods may be provided, but the caller should
        generate advice for upper and lower staves separately to be able to combine advice more intelligently.
        :param offset:
        :param first_digit: Constrain the advice to begin with this digit (1-5). Inconsistent with "both"
        staff parameter.
        :param last_digit: Constrain the advice to end with this digit (1-5). Inconsistent with "both"
        staff parameter.
        :param k: The number of suggestions and corresponding costs to return. (Up to this number may be
        returned.)
        :return: suggestions, costs: Two lists are returned. The first contains suggested fingering
        solutions as abcDF strings. The second contains the respective costs of each suggestion.
        """
        d_scores = self._d_corpus.d_score_list()
        if score_index >= len(d_scores):
            raise Exception("Score index out of range")

        d_score = d_scores[score_index]
        if staff == "both":
            if d_score.part_count() < 2:
                raise Exception("Only one staff present.")

            if offset or first_digit or last_digit:
                raise Exception("Ambiguous use to offset and/or first/last digit for both staves.")

            upper_suggestions, upper_costs = self.generate_advice(score_index=score_index, staff="upper", k=k)
            lower_suggestions, lower_costs = self.generate_advice(score_index=score_index, staff="lower", k=k)
            upper_length = d_score.upper_d_part().length()
            lower_length = d_score.lower_d_part().length()
            return self.combine_staves(upper_suggestions=upper_suggestions, upper_costs=upper_costs,
                                       lower_suggestions=lower_suggestions, lower_costs=lower_costs,
                                       upper_length=upper_length, lower_length=lower_length, k=k)

        if staff != "upper" and staff != "lower":
            raise Exception("Segregated advice is only dispensed one staff at a time.")

        handed_first_digit = Dactyler.hand_digit(digit=first_digit, staff=staff)
        handed_last_digit = Dactyler.hand_digit(digit=last_digit, staff=staff)

        if d_score.part_count() == 1:
            d_part = d_score.combined_d_part()
        else:
            # We support (segregated) left hand fingerings. By segregated, we
            # mean the right hand is dedicated to the upper staff, and the
            # left hand is dedicated to the lower staff.
            d_part = d_score.d_part(staff=staff)

        segments = d_part.orderly_note_stream_segments(offset=offset)
        segment_index = 0
        last_segment_index = len(segments) - 1
        segment_lengths = list()
        suggestions_for_segment = list()
        costs_for_segment = list()
        for segment in segments:
            segment_offset = 0
            segment_handed_first = None
            segment_handed_last = None
            if segment_index == 0:
                segment_offset = offset
                segment_handed_first = handed_first_digit
            segment_length = len(segment) - segment_offset
            if segment_index == last_segment_index:
                segment_handed_last = handed_last_digit

            suggestions, costs = self.generate_segment_advice(segment=segment, staff=staff,
                                                              offset=segment_offset,
                                                              handed_first_digit=segment_handed_first,
                                                              handed_last_digit=segment_handed_last, k=k)
            suggestions_for_segment.append(suggestions)
            costs_for_segment.append(costs)
            segment_lengths.append(segment_length)
            segment_index += 1

        suggestions, costs = self.combine_segments(suggestions_for_segment=suggestions_for_segment,
                                                   costs_for_segment=costs_for_segment,
                                                   segment_lengths=segment_lengths, k=k)
        return suggestions, costs

    def advise(self, score_index=0, staff="upper", offset=0, first_digit=None, last_digit=None):
        """
        Generate advice for the specified score. This method only supports segregated advice.
        :param score_index:
        :param staff: One of "upper," "lower," or "both."
        :param offset:
        :param first_digit: Constrain the advice to begin with this digit (1-5). Inconsistent with "both"
        staff parameter.
        :param last_digit: Constrain the advice to end with this digit (1-5). Inconsistent with "both"
        staff parameter.
        :param k: The number of suggestions and corresponding costs to return. (Up to this number may be
        returned.)
        :return: abcDF advice string.
        """
        suggestions, costs = self.generate_advice(score_index=score_index, staff=staff, offset=offset,
                                                  first_digit=first_digit, last_digit=last_digit)
        return suggestions[0]

    def load_corpus(self, d_corpus=None, path=None):
        """
        Load corpus to be processed by the Dactyler model.
        :param d_corpus: A DCorpus object.
        :param path: The path to a file that can be opened as a DCorpus object.
        :return:
        """
        if d_corpus:
            self._d_corpus = d_corpus
        elif path:
            self._d_corpus = DCorpus(path)
        else:
            raise Exception("No corpus specified for Dactyler.")

    @staticmethod
    def strike_distance_cost(gold_hand, gold_digit, test_hand, test_digit, method="hamming"):
        if method == "hamming":
            if test_digit != gold_digit or test_hand != gold_hand:
                return 1
            else:
                return 0

        one = str(gold_hand) + str(gold_digit)
        other = str(test_hand) + str(test_digit)
        if method == "natural":
            cost = Constant.NATURAL_EDIT_DISTANCES[(one, other)]
            return cost
        elif method == "pivot":
            cost = Constant.PIVOT_EDIT_DISTANCES[(one, other)]
            return cost
        else:
            raise Exception("Unsupported method: {0}".format(method))

    def score_note_count(self, score_index=0, staff="both"):
        d_score = self._d_corpus.d_score_by_index(score_index)
        note_count = d_score.note_count(staff=staff)
        return note_count

    @staticmethod
    def _distance_and_loc(method, staff, test_annot, gold_annot, gold_offset=0, zero_cost=False):
        current_gold_hand = ">" if staff == "upper" else "<"
        current_test_hand = ">" if staff == "upper" else "<"

        test_sf_count = test_annot.score_fingering_count(staff=staff)
        gold_sf_count = gold_annot.score_fingering_count(staff=staff)

        adjusted_gold_sf_count = gold_sf_count - gold_offset
        if test_sf_count != adjusted_gold_sf_count:
            raise Exception("Length mismatch: test: {0} gold: {1}".format(test_sf_count, adjusted_gold_sf_count))

        score = 0
        i = None
        gold_digit = None
        for i in range(test_sf_count):
            gold_i = i + gold_offset
            gold_sf = gold_annot.score_fingering_at_index(index=gold_i, staff=staff)
            gold_strike = gold_sf.pf.fingering.strike
            gold_hand = gold_strike.hand if gold_strike.hand else current_gold_hand
            gold_digit = int(gold_strike.digit)

            test_sf = test_annot.score_fingering_at_index(index=i, staff=staff)
            test_strike = test_sf.pf.fingering.strike
            test_hand = test_strike.hand if test_strike.hand else current_test_hand
            test_digit = int(test_strike.digit)

            current_gold_hand = gold_hand
            current_test_hand = test_hand

            cost = Dactyler.strike_distance_cost(method=method,
                                                 gold_hand=gold_hand,
                                                 gold_digit=gold_digit,
                                                 test_hand=test_hand,
                                                 test_digit=test_digit)
            if zero_cost and cost:
                return cost, gold_i, gold_digit
            score += cost

        return score, i, gold_digit

    @staticmethod
    def _eval_strike_distance(method, staff, test_annot, gold_annot):
        (cost, location, gold_digit) = Dactyler._distance_and_loc(method=method, staff=staff,
                                                                  test_annot=test_annot, gold_annot=gold_annot)
        return cost

    def evaluate_strike_distance(self, method="hamming", score_index=0, staff="upper", gold_indices=[]):
        """
        Evaluate the best solution for a given score reported by the model against each of the specified
        gold-standard annotations embedded in the DScore object using the specified method.
        :param method: The edit distance metric to apply in the evaluation. One of "hamming," "natural," or "pivot."
        :param score_index: The zero-based index of the DScore within the DCorpus currently loaded in this Dactyler.
        :param staff: The staff to include in the evaluation (one of "upper," "lower," or "both."
        :param gold_indices: The zero-based indices of the DAnnotation objects from the DScore's ABCDHeader to use
        as the gold standard advice.
        :return: An array of distance measures against each specified gold index, in sorted order.
        """
        d_score = self._d_corpus.d_score_by_index(score_index)
        if not d_score.is_fully_annotated():
            raise Exception("Only fully annotated scores can be evaluated.")

        if staff == "both":
            staves = ['upper', 'lower']
        else:
            staves = [staff]

        test_abcdf = self.advise(score_index=score_index, staff=staff)
        test_annot = DAnnotation(abcdf=test_abcdf)
        hdr = d_score.abcd_header()
        scores = []
        gold_index = 0
        for gold_annot in hdr.annotations():
            if gold_indices and gold_index not in gold_indices:
                gold_index += 1
                continue
            score = 0
            for staff in staves:
                score += Dactyler._eval_strike_distance(method=method, staff=staff,
                                                        test_annot=test_annot, gold_annot=gold_annot)
            scores.append(score)
            gold_index += 1

        return scores

    def evaluate_strike_distances(self, method="hamming", score_index=0, staff="upper", gold_indices=[], k=1):
        """
        Evaluate the k best solutions for a given score reported by the model against each of the specified
        gold-standard annotations embedded in the DScore object.
        :param method: The edit distance metric to apply in the evaluation. One of "hamming," "natural," or "pivot."
        :param score_index: The zero-based index of the DScore within the DCorpus currently loaded in this Dactyler.
        :param staff: The staff to include in the evaluation (one of "upper" or "lower.")
        :param gold_indices: The zero-based indices of the DAnnotation objects from the DScore's ABCDHeader to use
        as the gold standard advice.
        :param k: The number of "k-best" abcDF advice strings produced by the Dactyler model to evaluate.
        :return suggestions, costs, scores_for_gold_index: An array of the suggested abcDF strings. An array of
        the costs determined by the model, and a dictionary mapping gold-standard indices to an array of corresponding
        distance metric scores.
        """
        if staff != "upper" and staff != "lower":
            raise Exception("Strike distances must be evaluated one staff at a time.")

        d_score = self._d_corpus.d_score_by_index(score_index)
        if not d_score.is_fully_annotated():
            raise Exception("Only fully annotated scores can be evaluated.")

        suggestions, costs = self.generate_advice(score_index=score_index, staff=staff, k=k)
        hdr = d_score.abcd_header()
        scores_for_gold_index = dict()
        gold_index = 0

        for gold_annot in hdr.annotations():
            if gold_indices and gold_index not in gold_indices:
                gold_index += 1
                continue
            scores = []
            for i in range(len(suggestions)):
                test_annot = DAnnotation(abcdf=suggestions[i])
                score = Dactyler._eval_strike_distance(method=method, staff=staff,
                                                       test_annot=test_annot, gold_annot=gold_annot)
                scores.append(score)

            scores_for_gold_index[gold_index] = scores
            gold_index += 1

        return suggestions, costs, scores_for_gold_index

    def evaluate_strike_reentry(self, method="hamming", score_index=0, staff="upper", gold_indices=[]):
        d_score = self._d_corpus.d_score_by_index(score_index)
        if not d_score.is_fully_annotated():
            raise Exception("Only fully annotated scores can be evaluated.")

        if staff == "both":
            staves = ['upper', 'lower']
        else:
            staves = [staff]

        hdr = d_score.abcd_header()
        scores = {"upper": [], "lower": []}
        for staff in staves:
            current_gold_index = 0
            for gold_annot in hdr.annotations():
                score = 0
                if len(gold_indices) > 0 and current_gold_index not in gold_indices:
                    current_gold_index += 1
                    continue
                current_gold_index += 1

                test_abcdf = self.advise(score_index=score_index, staff=staff)
                self.squawk("COMPLETE ADVICE: {0}".format(test_abcdf))
                if staff == 'upper':
                    test_abcdf += '@'
                else:
                    test_abcdf = '@' + str(test_abcdf)
                test_annot = DAnnotation(abcdf=test_abcdf)
                (cost, loc, gold_digit) = self._distance_and_loc(zero_cost=True, method=method, staff=staff,
                                                                 test_annot=test_annot, gold_annot=gold_annot)
                score += cost
                self.squawk("GOLD: {0} staff for {1}".format(staff, gold_annot.abcdf(staff=staff, flat=True)))
                self.squawk("TEST: {0} staff for {1}".format(staff, test_abcdf))
                self.squawk("SCORE: {0} COST: {1} LOC: {2}".format(score, cost, loc))

                while cost > 0:
                    test_abcdf = self.advise(score_index=score_index, staff=staff,
                                             offset=loc, first_digit=gold_digit)
                    self.squawk("GOLD: {0} staff for {1}".format(staff, gold_annot.abcdf(staff=staff, flat=True)))
                    self.squawk("TRUNCATED ADVICE: {0}".format(test_abcdf))
                    if staff == 'upper':
                        test_abcdf += '@'
                    else:
                        test_abcdf = '@' + str(test_abcdf)
                    test_annot = DAnnotation(abcdf=test_abcdf)
                    (cost, loc, gold_digit) = self._distance_and_loc(zero_cost=True, gold_offset=loc,
                                                                     method=method, staff=staff,
                                                                     test_annot=test_annot, gold_annot=gold_annot)
                    score += cost
                    self.squawk("     score: {0} cost: {1} loc: {2}".format(score, cost, loc))

                scores[staff].append(score)

        total_scores = []
        if len(scores['upper']) > 0:
            for i in range(len(scores['upper'])):
                total_scores.append(scores['upper'][i])
                if len(scores['lower']) > 0:
                    total_scores[i] += scores['lower'][i]
        elif len(scores['lower']) > 0:
            for i in range(len(scores['lower'])):
                total_scores.append(scores['lower'][i])
        else:
            raise Exception("No scores found.")

        return total_scores

    @staticmethod
    def _pivot_flags(staff, d_notes, annot):
        current_hand = ">" if staff == "upper" else "<"

        sf_count = annot.score_fingering_count(staff=staff)

        pivot_flags = list()
        prior_ad_note = None
        for i in range(sf_count):
            d_note = d_notes[i]
            sf = annot.score_fingering_at_index(index=i, staff=staff)
            strike = sf.pf.fingering.strike
            hand = strike.hand if strike.hand else current_hand
            digit = int(strike.digit)
            ad_note = AnnotatedDNote(m21_note=d_note.m21_note(),
                                     prior_note=prior_ad_note,
                                     strike_hand=hand,
                                     strike_digit=digit)

            if ad_note.is_pivot():
                pivot_flags.append(True)
            else:
                pivot_flags.append(False)
            prior_ad_note = ad_note
            current_hand = hand

        return pivot_flags

    @staticmethod
    def _pivot_alignment_score(gold_pivot_flags, test_pivot_flags, cost=1):
        if len(gold_pivot_flags) != len(test_pivot_flags):
            raise Exception("Mismatched pivot flag lists.")

        score = 0
        for i in range(len(gold_pivot_flags)):
            if gold_pivot_flags[i] != test_pivot_flags[i]:
                score += cost
        return score

    def evaluate_pivot_alignment(self, score_index=0, staff="upper", gold_indices=[]):
        d_score = self._d_corpus.d_score_by_index(score_index)
        if not d_score.is_fully_annotated():
            raise Exception("Only fully annotated scores can be evaluated.")

        if staff == "both":
            staves = ['upper', 'lower']
        else:
            staves = [staff]

        test_abcdf = self.advise(score_index=score_index, staff=staff)
        test_annot = DAnnotation(abcdf=test_abcdf)
        hdr = d_score.abcd_header()
        scores = []
        gold_index = 0
        for gold_annot in hdr.annotations():
            if gold_indices and gold_index not in gold_indices:
                gold_index += 1
                continue
            score = 0
            for staff in staves:
                d_part = d_score.d_part(staff=staff)
                d_notes = d_part.orderly_d_notes()
                test_pivot_flags = Dactyler._pivot_flags(staff=staff, d_notes=d_notes, annot=test_annot)
                gold_pivot_flags = Dactyler._pivot_flags(staff=staff, d_notes=d_notes, annot=gold_annot)
                staff_score = self._pivot_alignment_score(gold_pivot_flags=gold_pivot_flags, test_pivot_flags=test_pivot_flags)
                score += staff_score
            scores.append(score)
            gold_index += 1

        return scores


class TrainedDactyler(Dactyler):
    def __init__(self):
        super().__init__()
        self._training = {}

    @abstractmethod
    def generate_segment_advice(self, segment, staff, offset, handed_first_digit, handed_last_digit, k=None):
        pass

    @abstractmethod
    def train(self, d_corpus, staff="both", segregate=True, segmenter=None, annotation_indices=[]):
        return

    def retain(self, pickle_path=None, to_db=False):
        if pickle_path:
            with open(pickle_path, 'wb') as pickle_file:
                pickle.dump(self._training, pickle_file, pickle.HIGHEST_PROTOCOL)
        elif to_db:
            raise Exception("Retaining pickled file to database not yet supported.")
        else:
            raise Exception("No retention destination specified.")

    def recall(self, pickle_path=None, pickle_db_id=None):
        if pickle_path:
            with open(pickle_path, 'rb') as pickle_file:
                self._training = pickle.load(pickle_file)
        elif pickle_db_id:
            raise Exception("Recalling pickled training from database not yet supported.")
        else:
            raise Exception("No source specified from which to recall training.")

    def training(self):
        return self._training

    def demonstrate(self):
        for k in self._training:
            print(k, ': ', self._training[k])

