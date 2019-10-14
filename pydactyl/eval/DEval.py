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
from krippendorff import alpha
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import seaborn as sns
import time
import math
from pydactyl.dcorpus.DCorpus import DCorpus
from pydactyl.dactyler import Constant
from pydactyl.dcorpus.DNote import DNote
from pydactyl.dcorpus.DAnnotation import DAnnotation


class DEval(ABC):
    def __init__(self):
        self._dactyler = None
        self._d_corpus = DCorpus()
        self._gold = dict()
        self._gold['upper'] = list()
        self._gold['lower'] = list()
        self._discord = dict()
        self._discord['upper'] = dict()
        self._discord['lower'] = dict()

    @staticmethod
    def _max_distance_for_method(method):
        if method == "natural":
            # d_max = Constant.MAX_NATURAL_EDIT_DISTANCE
            # FIXME: desegregated max turns DCPG murky and useless.
            d_max = 4
        elif method == "hamming":
            d_max = 1
        else:
            raise Exception("Not ready to measure {} gain".format(method))
        return d_max

    @staticmethod
    def _proximal_gain(test_abcdf, gold_handed_strikers, gold_vote_counts, staff, method="hamming"):
        testee = DAnnotation.abcdf_to_handed_strike_digits(test_abcdf, staff=staff)
        d_max = DEval._max_distance_for_method(method)
        nugget_index = 0
        total_gain = 0
        for nugget in gold_handed_strikers:
            nugget_gain = 0
            for i in range(len(testee)):
                distance = DAnnotation.strike_distance_cost(gold_handed_digit=nugget[i],
                                                            test_handed_digit=testee[i],
                                                            method=method)
                nugget_gain += (d_max - distance)
            nugget_gain *= gold_vote_counts[nugget_index]
            nugget_index += 1
            total_gain += nugget_gain
        return total_gain

    @staticmethod
    def _normalized_proximal_gain(test_abcdf, gold_handed_strikers, gold_vote_counts, staff, method="hamming"):
        p_gain = DEval._proximal_gain(test_abcdf, gold_handed_strikers, gold_vote_counts, staff, method)
        digits = DAnnotation.abcdf_to_handed_strike_digits(test_abcdf, staff=staff)
        n_total = len(digits)
        h_total = 0
        for count in gold_vote_counts:
            h_total += count
        d_max = DEval._max_distance_for_method(method)
        normalizer = h_total * n_total * d_max
        npg = 1.0 * p_gain / normalizer
        return npg

    @abstractmethod
    def load_data(self):
        return

    def sow_discord(self):
        return

    def gold_asts(self, score_index, staff="upper", last_digit=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        asts = list()
        for abcdf in score_gold:
            ast = DAnnotation.abcdf_to_ast(abcdf)
            asts.append(ast)
        return asts

    def gold_list_of_handed_strike_lists(self, score_index, staff="upper", last_digit=None):
        gold_asts = self.gold_asts(score_index, staff=staff, last_digit=last_digit)
        list_of_strike_lists = list()
        for gold_ast in gold_asts:
            nuggets = DAnnotation.ast_to_handed_strike_digits(gold_ast, staff=staff)
            list_of_strike_lists.append(nuggets)
        return list_of_strike_lists

    def gold_clusters(self, score_index, staff="upper", last_digit=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)

    @staticmethod
    def krippendorfs_alpha(upper_rh_advice, exercise_upper_gold):
        fingerings = list(upper_rh_advice)
        fingerings.pop(0)
        finger_ints = list(map(int, fingerings))
        exercise_upper_gold.append(finger_ints)
        krip = alpha(reliability_data=exercise_upper_gold, level_of_measurement='interval')
        exercise_upper_gold.pop()
        return krip

    def assert_good_gold(self, staff="upper"):
        score_count = self._d_corpus.score_count()
        gold_score_count = len(self._gold[staff])
        if score_count != gold_score_count:
            raise Exception("Gold count ({0}) does not match score count ({1}).".format(
                gold_score_count, score_count))

    def _random_sorted_gold(self, score_index, staff="upper", last_digit=None):
        """
        Returns a total-order list of gold fingerings for the given staff. Fingerings with
        the same number of votes (user counts) are randomized, but in a reproducible
        manner.
        :param score_index:
        :param staff:
        :return:
        """
        nuggets = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        distinct_nuggets = list(set(nuggets))
        nuggets_at = dict()
        for nugget in distinct_nuggets:
            vote_count = DEval.gold_suggestion_count(suggestion=nugget, score_gold=nuggets)
            if vote_count not in nuggets_at:
                nuggets_at[vote_count] = list()
            nuggets_at[vote_count].append(nugget)
        sorted_nuggets = list()
        random.seed(a=123654)
        for vote_count in sorted(nuggets_at):
            if len(nuggets_at[vote_count]) == 1:
                sorted_nuggets.append(nuggets_at[vote_count][0])
            else:
                nuggets = list(nuggets_at[vote_count])
                random.shuffle(nuggets)
                for nugget in nuggets:
                    sorted_nuggets.append(nugget)
        return sorted_nuggets

    def _all_gold_found(self, suggestions, score_index, staff="upper", last_digit=None):
        """
        FIXME: Only works for simple segregated fingerings with one hand marker at the beginning.
        :param suggestions:
        :return:
        """
        suggested = dict()
        for suggestion in suggestions:
            suggested[suggestion] = True

        gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        for nugget in gold:
            if nugget not in suggested:
                return False
        return True

    def _highest_cost_of_gold(self, suggestion_data_by_cost, score_index, staff="upper", last_digit=None):
        gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        highest_cost = -1
        for cost in suggestion_data_by_cost:
            if cost <= highest_cost:
                continue
            for data in suggestion_data_by_cost[cost]:
                sugg = data['suggestion']
                if sugg in gold:
                    highest_cost = cost
        return highest_cost

    def _trim_suggestions(self, suggestions, score_index, staff="upper", last_digit=None):
        """
        Remove suggestions after final gold suggestion is found in suggestions list.
        :param suggestions: A list of abcDF suggestions that includes all gold advice.
        :param score_index: ID for score being evaluated.
        :param staff: Upper or lower staff.
        :param last_digit: The final digit to require in all output abcDF.
        :return: trimmed_suggestions list of abcDF strings.
        """
        gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        nugget_count = 0
        for i in range(len(suggestions)):
            if suggestions[i] in gold:
                nugget_count += 1
            if nugget_count == len(gold):
                break
        trimmed_suggestions = suggestions[0:i+1]
        return trimmed_suggestions

    @staticmethod
    def suggestion_data_by_cost(suggestions, costs, details, cutoff_rank=None):
        """
        Return a dictionary of suggestion data keyed by total costs.
        :param suggestions: A list of abcDF suggestion strings, ordered by increasing cost.
        :param costs: A corresponding list of the particular suggestion costs.
        :param details: A corresponding list of cost details for each suggestion
        :param cutoff_rank: The final rank (as indicated by sorting by cost) to include in
        returned results.
        :return: data_by_cost: A dictionary of lists of dictionaries.
        """
        data_by_cost = dict()
        sugg_count = 0
        last_cost_to_include = None
        for i in range(len(suggestions)):
            sugg_count += 1
            sugg = suggestions[i]
            cost = costs[i]
            detail = details[i]

            if last_cost_to_include is not None and cost > last_cost_to_include:
                break

            if cutoff_rank and cutoff_rank >= sugg_count:
                last_cost_to_include = cost

            if cost not in data_by_cost:
                data_by_cost[cost] = list()
            data = {'suggestion': sugg, 'details': detail}
            data_by_cost[cost].append(data)
        return data_by_cost

    def score_advice(self, score_index, staff="upper", cycle=None, last_digit=None, k=10):
        # FIXME: I don't see what this method buys us. generate_advice() should do the right
        # default thing, no?
        if cycle:
            suggestions, costs, details = \
                self._dactyler.generate_advice(staff=staff, score_index=score_index, cycle=cycle, k=k)
        elif last_digit:
            suggestions, costs, details = \
                self._dactyler.generate_advice(staff=staff, score_index=score_index, last_digit=last_digit, k=k)
        else:
            suggestions, costs, details = \
                self._dactyler.generate_advice(staff=staff, score_index=score_index, k=k)
        return suggestions, costs, details

    def _score_gold(self, score_index, staff="upper", last_digit=None):
        all_gold = self._gold[staff][score_index]
        if last_digit:
            end_re = re.compile(str(last_digit) + '$')
            constrained_gold = list()
            for nugget in all_gold:
                if end_re.search(nugget):
                    constrained_gold.append(nugget)
            return constrained_gold
        return all_gold

    def _score_gold_votes(self, score_index, staff="upper", last_digit=None):
        all_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        gold_votes = {}
        for nugget in all_gold:
            if nugget in gold_votes:
                gold_votes[nugget] += 1
            else:
                gold_votes[nugget] = 1
        return gold_votes

    def _segmented_gold_strike_list(self, score_index, segment_lengths, staff="upper", last_digit=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        score_data = list()
        for seg_index in range(len(segment_lengths)):
            score_data.append(list())

        for score_annotation in score_gold:
            score_strikers = DAnnotation.abcdf_to_handed_strike_digits(score_annotation, staff=staff)
            seg_start = 0
            seg_end = 0
            seg_index = 0
            human_data = list()
            while True:
                segment_data = score_data[seg_index]
                if seg_start == len(score_strikers):
                    break
                seg_end += segment_lengths[seg_index]
                # phrase_data.append(score_strikers[seg_start:seg_end])
                segment_fingerings = list()
                for i in range(start=seg_start, stop=seg_end):
                    segment_fingerings.append(score_strikers[i])
                human_data.append(segment_fingerings)
                segment_data.append(human_data)
                seg_start = seg_end
        # score_data[segment_index][h][n]
        return score_data

    def _score_gold_distinct_count(self, score_index, staff="upper", last_digit=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        return len(set(score_gold))

    def _segment_gold_weighted_vote_totals(self, score_index, segment_lengths, staff="upper", last_digit=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        total = 0
        for fingering_str in score_gold:
            total += score_gold[fingering_str]
        return total

    def _score_gold_total_count(self, score_index, staff="upper", last_digit=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        total = 0
        for fingering_str in score_gold:
            total += 1
        return total

    def _score_gold_counts(self, score_index, staff="upper", last_digit=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        vote_counts = list()
        for fingering_str in score_gold:
            vote_counts.append(score_gold[fingering_str])
        return vote_counts

    def _recall_them_all(self, score_index, staff="upper", cycle=None, last_digit=None):
        k = 25
        last_count = None
        while True:
            suggestions, costs, details = self.score_advice(score_index=score_index, staff=staff,
                                                            last_digit=last_digit, cycle=cycle, k=k)
            suggestion_count = len(suggestions)
            if suggestion_count == last_count:
                break
            last_count = suggestion_count
            if self._all_gold_found(score_index=score_index, staff=staff,
                                    last_digit=last_digit, suggestions=suggestions):
                break
            k *= 2
        suggestion_count = len(suggestions)
        if suggestion_count <= 0:
            raise Exception("Bad suggestion count")
        suggestions = self._trim_suggestions(suggestions=suggestions, score_index=score_index,
                                             last_digit=last_digit, staff=staff)
        costs = costs[0:len(suggestions)]
        details = details[0:len(suggestions)]
        return suggestions, costs, details

    def score_avg_p(self, suggestions, score_index, staff="upper", last_digit=None, k=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        relevant_count = len(score_gold)
        if relevant_count == 0:
            raise Exception("No gold to search for.")
        tp_count = 0
        fp_count = 0
        p_sum = 0
        precision_at_rank = list()
        recall_at_rank = list()
        suggestion_id = 0
        for suggestion in suggestions:
            if k is not None and suggestion_id >= k:
                break
            if suggestion in score_gold:
                tp_count += 1
            else:
                fp_count += 1
            p = tp_count/(tp_count + fp_count)
            precision_at_rank.append(p)
            if suggestion in score_gold:
                p_sum += p  # Recall goes up, so we include this p in the sum to average.
            r = tp_count/relevant_count
            recall_at_rank.append(r)
            suggestion_id += 1
        avg_p = 1.0 * p_sum / relevant_count
        result = {
            'relevant': relevant_count,
            'p_at_rank': precision_at_rank,
            'r_at_rank': recall_at_rank,
            'avg_p': avg_p}
        return result

    def score_avg_p_at_perfect_recall(self, score_index, staff="upper", cycle=None, last_digit=None):
        self.assert_good_gold(staff=staff)
        suggestions, costs, details = self._recall_them_all(score_index=score_index, staff=staff,
                                                            last_digit=last_digit, cycle=cycle)
        result = self.score_avg_p(suggestions=suggestions, score_index=score_index,
                                  last_digit=last_digit, staff=staff)
        return result

    def score_avg_p_at_k(self, score_index, staff="upper", cycle=None, last_digit=None, k=None):
        self.assert_good_gold(staff=staff)
        suggestions, costs, details = self.score_advice(score_index=score_index, staff=staff,
                                                        last_digit=last_digit, cycle=cycle, k=k)
        result = self.score_avg_p(suggestions=suggestions, score_index=score_index,
                                  last_digit=last_digit, staff=staff, k=k)
        return result

    def score_p_r_at_k(self, score_index, staff="upper", cycle=None, last_digit=None, k=5):
        """
        Precision and recall at cutoff k. Ignores ties in model output and in gold standard.
        :param score_index: Identifies the score in the loaded corpus to consider.
        :param staff: The staff to use.
        :param cycle: The cycle constraint to place on the model. Incompatible with last_digit
        parameter.
        :param last_digit: Constrain the last digit output by the model. Incompatible with
        cycle parameter.
        :param k: The cutoff rank to enforce.
        :return:
        """
        self.assert_good_gold(staff=staff)
        avg_data = self.score_avg_p_at_k(score_index=score_index, staff=staff, cycle=cycle, last_digit=last_digit, k=k)
        precision = avg_data['p_at_rank'][-1]
        relevant_count = avg_data['relevant']
        total_relevant_count = self._score_gold_distinct_count(score_index=score_index, staff=staff)
        recall = relevant_count/total_relevant_count
        result = {
            'relevant': relevant_count,
            'total_relevant': total_relevant_count,
            'precision': precision,
            'recall': recall
        }
        return result

    def _relevancy_asfarray(self, suggestions, score_index, staff="upper", last_digit=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        vote_total = self._score_gold_total_count(score_index=score_index, staff=staff, last_digit=last_digit)
        if vote_total == 0:
            raise Exception("No gold found.")

        relevancy = list()
        for fingering in suggestions:
            vote_count = DEval.gold_suggestion_count(suggestion=fingering, score_gold=score_gold)
            relevancy.append(vote_count)

        relevancy_asf = np.asfarray(a=relevancy)
        relevancy_asf /= vote_total
        return relevancy_asf

    def _proximal_gains(self, suggestions, score_index, staff="upper", last_digit=None,
                        method="hamming", normalize=False):
        vote_total = self._score_gold_total_count(score_index=score_index, staff=staff, last_digit=last_digit)
        if vote_total == 0:
            raise Exception("No gold found.")

        gold_striker_list = self.gold_list_of_handed_strike_lists(score_index, staff=staff, last_digit=last_digit)
        gold_vote_counts = self._score_gold_counts(score_index, staff=staff, last_digit=last_digit)

        pg_values = list()
        for fingering in suggestions:
            if normalize:
                pg = DEval._normalized_proximal_gain(test_abcdf=fingering, gold_handed_strikers=gold_striker_list,
                                                     gold_vote_counts=gold_vote_counts, staff=staff, method=method)
            else:
                pg = DEval._proximal_gain(test_abcdf=fingering, gold_handed_strikers=gold_striker_list,
                                          gold_vote_counts=gold_vote_counts, staff=staff, method=method)
            pg_values.append(pg)

        return pg_values

    @staticmethod
    def _cpgs(proximal_gains):
        r = len(proximal_gains)
        cumulative_gains = list()
        total = 0
        for i in range(r):
            total += proximal_gains[0]
            cumulative_gains.append(total)
        return cumulative_gains

    @staticmethod
    def _dcpgs(cpgs, phi=None, p=None):
        r = len(cpgs)
        gains = list()
        for i in range(r):
            if phi:
                discount_factor = phi(r=i+1, p=p)
            else:
                discount_factor = 1.0/math.log2(i+2)
            gain = discount_factor * cpgs[i]
            gains.append(gain)
        return gains

    def _dcpg_normalizer(self, score_index, staff="upper", method="hamming", last_digit=None, k=10):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        nuggets = list(score_gold.keys())
        sample_abcdf = nuggets[0]
        digits = DAnnotation.abcdf_to_handed_strike_digits(sample_abcdf, staff=staff)
        n_total = len(digits)
        h_total = self._score_gold_total_count(score_index=score_index, staff=staff, last_digit=last_digit)
        d_max = DEval._max_distance_for_method(method)
        r_total = k
        normalizer = r_total * h_total * n_total * d_max
        return normalizer

    @staticmethod
    def gold_suggestion_count(suggestion, score_gold):
        sugg_count = 0
        for i in range(len(score_gold)):
            if score_gold[i] == suggestion:
                sugg_count += 1
        return sugg_count

    @staticmethod
    def estimated_prob_user_happy(suggestion, score_gold, h_total):
        if h_total == 0:
            return 0
        suggestion_count = DEval.gold_suggestion_count(suggestion=suggestion, score_gold=score_gold)
        prob = 1.0 * suggestion_count/h_total
        return prob

    @staticmethod
    def p_prime(suggestion, segment_gold, staff="upper"):
        suggested_strikers = DAnnotation.abcdf_to_handed_strike_digits(suggestion, staff=staff)
        big_h = len(segment_gold)
        if big_h == 0:
            return 0

        sum_total = 0
        for human_data in segment_gold:
            big_w = 0
            match = 1
            for note_index in range(len(human_data)):
                if human_data[note_index] == 'x':
                    big_w += 1
                elif human_data[note_index] != suggested_strikers[note_index]:
                    match = 0
                    break
            sum_total += match/(big_w + 1)
        p_prime = (1/big_h)*sum_total
        return p_prime

    @staticmethod
    def segment_discord(segment_gold):
        discordant_with = dict()
        for j in range(len(segment_gold)):
            outer_annotations = segment_gold[j]
            discordants = dict()
            for i in range(len(segment_gold)):
                inner_annotations = segment_gold[i]
                if i == j:
                    continue
                clashes = 0
                for n in range(len(outer_annotations)):
                    if not (outer_annotations[n] == inner_annotations[n] or
                            outer_annotations[n] == 'x' or
                            inner_annotations[n] == 'x'):
                        clashes += 1
                if clashes > 0:
                    discordants[i] = clashes
            discordant_with[j] = discordants
        return discordant_with

    @staticmethod
    def p_double_prime(suggestion, segment_gold, segment_discord, staff="upper"):
        suggested_strikers = DAnnotation.abcdf_to_handed_strike_digits(suggestion, staff=staff)
        big_h = len(segment_gold)
        if big_h == 0:
            return 0

        sum_total = 0
        human_index = 0
        for human_data in segment_gold:
            big_w = 0
            clashes = 0
            for note_index in range(len(human_data)):
                if human_data[note_index] == 'x':
                    big_w += 1
                elif human_data[note_index] != suggested_strikers[note_index]:
                    clashes += 1
            discordants = segment_discord[human_index]

            sum_total += (1/(2**clashes))/(big_w + 1)  # + DEval.discord()
            human_index += 1
        p_prime = (1/big_h)*sum_total

        return p_prime

    def segmented_wildcard_rank_at_k(self, score_index, method="match",
                                     staff="upper", cycle=None, last_digit=None, phi=None, p=None, k=10):
        self.assert_good_gold(staff=staff)
        if k is None:
            raise Exception("Cannot yet recall all for phrases.")
        segment_suggestions, segment_costs, segment_details, segment_lengths = \
            self._dactyler.generate_segmented_advice(score_index=score_index, staff=staff, cycle=cycle,
                                                     offset=0, last_digit=last_digit, k=k)
        segment_gold_data = self._segmented_gold_strike_list(score_index=score_index, segment_lengths=segment_lengths,
                                                             staff=staff, last_digit=last_digit)
        err_for_segment = list()
        # segment_gold[segment_index][h][n]
        for seg_index in range(len(segment_gold_data)):
            seg_gold = segment_gold_data[seg_index]
            seg_discord = DEval.segment_discord(seg_gold)
            err = 0
            prob_still_going = 1
            for r in range(1, k+1):
                if phi:
                    discount_factor = phi(r=r, p=p)
                else:
                    discount_factor = 1.0/r
                if method == "match":
                    prob_found = DEval.p_prime(suggestion=segment_suggestions[r-1],
                                               segment_gold=seg_gold, staff=staff)
                elif method == "similarity":
                    prob_found = DEval.p_double_prime(suggestion=segment_suggestions[r-1], segment_gold=seg_gold,
                                                      segment_discord=seg_discord, staff=staff)
                elif method == "equity":
                    prob_found = DEval.p_double_prime(suggestion=segment_suggestions[r-1], segment_gold=seg_gold,
                                                      segment_discord=seg_discord, staff=staff)
                else:
                    raise Exception("Wildcard method {} not supported.".format(method))

                err += (discount_factor * prob_still_going * prob_found)
                prob_still_going *= (1 - prob_found)
            err_for_segment.append(err)
        return err_for_segment

    def score_err_at_k(self, score_index, staff="upper", cycle=None, last_digit=None, phi=None, p=None, k=10):
        self.assert_good_gold(staff=staff)
        if k is None:
            suggestions, costs, details = self._recall_them_all(score_index=score_index, staff=staff,
                                                                last_digit=last_digit, cycle=cycle)
        else:
            suggestions, costs, details = self.score_advice(score_index=score_index, staff=staff,
                                                            last_digit=last_digit, cycle=cycle, k=k)
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        h_total = self._score_gold_total_count(score_index=score_index, staff=staff, last_digit=last_digit)
        err = 0
        prob_still_going = 1
        for r in range(1, k+1):
            if phi:
                discount_factor = phi(r=r, p=p)
            else:
                discount_factor = 1.0/r
            prob_found = DEval.estimated_prob_user_happy(suggestion=suggestions[r-1],
                                                         score_gold=score_gold, h_total=h_total)
            err += (discount_factor * prob_still_going * prob_found)
            prob_still_going *= (1 - prob_found)
        return err

    def score_epr_at_k(self, score_index, staff="upper", cycle=None, last_digit=None,
                       method="hamming", phi=None, p=None, k=10):
        self.assert_good_gold(staff=staff)
        if k is None:
            suggestions, costs, details = self._recall_them_all(score_index=score_index, staff=staff,
                                                                last_digit=last_digit, cycle=cycle)
        else:
            suggestions, costs, details = self.score_advice(score_index=score_index, staff=staff,
                                                            last_digit=last_digit, cycle=cycle, k=k)
        norm_prox_gains = self._proximal_gains(suggestions=suggestions, score_index=score_index, method=method,
                                               staff=staff, last_digit=last_digit, normalize=True)
        epr = 0
        prob_still_going = 1
        for r in range(1, k+1):
            if phi:
                discount_factor = phi(r=r, p=p)
            else:
                discount_factor = 1.0/r
            prob_found = norm_prox_gains[r-1]
            epr += (discount_factor * prob_still_going * prob_found)
            prob_still_going *= (1 - prob_found)

        return epr

    def score_dcpg_at_k(self, score_index, staff="upper", cycle=None, last_digit=None,
                        method="hamming", phi=None, p=None, k=10):
        self.assert_good_gold(staff=staff)
        if k is None:
            suggestions, costs, details = self._recall_them_all(score_index=score_index, staff=staff,
                                                                last_digit=last_digit, cycle=cycle)
        else:
            suggestions, costs, details = self.score_advice(score_index=score_index, staff=staff,
                                                            last_digit=last_digit, cycle=cycle, k=k)
        prox_gains = self._proximal_gains(suggestions=suggestions, score_index=score_index,
                                          staff=staff, last_digit=last_digit, method=method)
        cpgs = DEval._cpgs(proximal_gains=prox_gains)
        dcpgs = DEval._dcpgs(cpgs=cpgs, phi=phi, p=p)
        dcpg_at_k = dcpgs[k-1]
        return dcpg_at_k

    def score_ndcpg_at_k(self, score_index, staff="upper", cycle=None, last_digit=None,
                         method="hamming", phi=None, p=None, k=10):
        dcpg = self.score_dcpg_at_k(score_index=score_index, staff=staff, cycle=cycle,
                                    last_digit=last_digit, phi=phi, p=p, k=k)
        normalizer = self._dcpg_normalizer(score_index=score_index, staff=staff, method=method, last_digit=last_digit)
        ndcpg = dcpg / normalizer
        return ndcpg

    def _ideal_relevancy_asfarray(self, score_index, staff="upper", last_digit=None, k=10):
        score_gold = self._score_gold_votes(score_index=score_index, staff=staff, last_digit=last_digit)
        sorted_gold = self._random_sorted_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        decreasing_gold = list(reversed(sorted_gold))
        vote_total = self._score_gold_total_count(score_index=score_index, staff=staff, last_digit=last_digit)

        relevancy = list()
        for i in range(k):
            if i < len(decreasing_gold):
                next_fingering = decreasing_gold[i]
                relevancy.append(score_gold[next_fingering])
            else:
                relevancy.append(0)

        relevancy_asf = np.asfarray(a=relevancy)
        relevancy_asf /= vote_total
        return relevancy_asf

    @staticmethod
    def _dcg(relevancy, base="2"):
        if base == "e":
            dcg = relevancy[0] + np.sum(relevancy[1:] / np.log(np.arange(2, relevancy.size + 1)))
        elif base == "2":
            dcg = relevancy[0] + np.sum(relevancy[1:] / np.log2(np.arange(2, relevancy.size + 1)))
        else:
            b = int(base)
            dcg = relevancy[0]
            for i in range(1, relevancy.size + 1):
                dcg += 1.0*relevancy[i]/math.log(i + 1, base=b)
        return dcg

    def score_dcg_at_k(self, score_index, staff="upper", cycle=None, last_digit=None, base="2", k=10):
        self.assert_good_gold(staff=staff)
        if k is None:
            suggestions, costs, details = self._recall_them_all(score_index=score_index, staff=staff,
                                                                last_digit=last_digit, cycle=cycle)
        else:
            suggestions, costs, details = self.score_advice(score_index=score_index, staff=staff,
                                                            last_digit=last_digit, cycle=cycle, k=k)
        relevancy = self._relevancy_asfarray(suggestions=suggestions, score_index=score_index,
                                             staff=staff, last_digit=last_digit)
        dcg_at_k = DEval._dcg(relevancy=relevancy, base=base)
        return dcg_at_k

    def score_ndcg_at_k(self, score_index, staff="upper", cycle=None, last_digit=None, base="2", k=10):
        self.assert_good_gold(staff=staff)
        if k is None:
            suggestions, costs, details = self._recall_them_all(score_index=score_index, staff=staff,
                                                                last_digit=last_digit, cycle=cycle)
        else:
            suggestions, costs, details = self.score_advice(score_index=score_index, staff=staff,
                                                            last_digit=last_digit, cycle=cycle, k=k)
        relevancy = self._relevancy_asfarray(suggestions=suggestions, score_index=score_index,
                                             staff=staff, last_digit=last_digit)
        ideal_relevancy = self._ideal_relevancy_asfarray(score_index=score_index, staff=staff,
                                                         last_digit=last_digit, k=len(suggestions))
        dcg_at_k = DEval._dcg(relevancy=relevancy, base=base)
        dcg_ideal = DEval._dcg(relevancy=ideal_relevancy, base=base)
        if not dcg_ideal:
            return 0.0
        ndcg_at_k = dcg_at_k/dcg_ideal
        return ndcg_at_k

    def score_alpha_at_k(self, score_index, staff="upper", cycle=None, last_digit=None, base="2", k=10):
        self.assert_good_gold(staff=staff)
        if k is None:
            suggestions, costs, details = self._recall_them_all(score_index=score_index, staff=staff,
                                                                last_digit=last_digit, cycle=cycle)
        else:
            suggestions, costs, details = self.score_advice(score_index=score_index, staff=staff,
                                                            last_digit=last_digit, cycle=cycle, k=k)
        relevancy = self._relevancy_asfarray(suggestions=suggestions, score_index=score_index,
                                             staff=staff, last_digit=last_digit)
        ideal_relevancy = self._ideal_relevancy_asfarray(score_index=score_index, staff=staff,
                                                         last_digit=last_digit, k=len(suggestions))
        dcg_at_k = DEval._dcg(relevancy=relevancy, base=base)
        dcg_ideal = DEval._dcg(relevancy=ideal_relevancy, base=base)
        if not dcg_ideal:
            return 0.0
        ndcg_at_k = dcg_at_k/dcg_ideal
        return ndcg_at_k

    @abstractmethod
    def map_at_perfect_recall(self, staff="upper"):
        return

    @abstractmethod
    def map_at_k(self, staff="upper", k=10):
        return

    @abstractmethod
    def p_r_at_k(self, staff="upper", k=10):
        return

    @abstractmethod
    def dcg_at_k(self, staff="upper", base="2", k=10):
        return

    @abstractmethod
    def ndcg_at_k(self, staff="upper", base="2", k=10):
        return

    @abstractmethod
    def dcpg_at_k(self, staff="upper", phi=None, p=None, k=10):
        return

    @abstractmethod
    def ndcpg_at_k(self, staff="upper", phi=None, p=None, k=10):
        return

    @abstractmethod
    def err_at_k(self, staff="upper", phi=None, p=None, k=10):
        return

    @abstractmethod
    def epr_at_k(self, staff="upper", phi=None, p=None, method="hamming", k=10):
        return

