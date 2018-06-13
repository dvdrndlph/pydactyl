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

from abc import abstractmethod
import networkx as nx
import copy
import re
import music21
from . import Dactyler as D
from didactyl.dcorpus.DNote import DNote

TEST_CORPUS = '/Users/dave/tb2/didactyl/dd/corpora/small.abc'
THUMB = 1
INDEX = 2
MIDDLE = 3
RING = 4
LITTLE = 5

NO_MIDI = -1

WEIGHT = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1
}

finger_span = {
    ('>1', '>2'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': 1, 'MaxRel': 5, 'MaxComf': 8, 'MaxPrac': 10},
    ('>1', '>3'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': 3, 'MaxRel': 7, 'MaxComf': 10, 'MaxPrac': 12},
    ('>1', '>4'): {'MinPrac': -3, 'MinComf': -1, 'MinRel': 5, 'MaxRel': 9, 'MaxComf': 12, 'MaxPrac': 14},
    ('>1', '>5'): {'MinPrac': -1, 'MinComf': 1, 'MinRel': 7, 'MaxRel': 10, 'MaxComf': 13, 'MaxPrac': 15},
    ('>2', '>3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},
    ('>2', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('>2', '>5'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 8, 'MaxPrac': 10},
    ('>3', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('>3', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('>4', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},
    ('>2', '>1'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -5, 'MaxRel': -1, 'MaxComf': 3, 'MaxPrac': 5},
    ('>3', '>1'): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -7, 'MaxRel': -3, 'MaxComf': 2, 'MaxPrac': 4},
    ('>4', '>1'): {'MinPrac': -14, 'MinComf': -12, 'MinRel': -9, 'MaxRel': -5, 'MaxComf': 1, 'MaxPrac': 3},
    ('>5', '>1'): {'MinPrac': -15, 'MinComf': -13, 'MinRel': -10, 'MaxRel': -7, 'MaxComf': -1, 'MaxPrac': 1},
    ('>3', '>2'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>4', '>2'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>2'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('>4', '>3'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>3'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>4'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<1'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': 1, 'MaxRel': 5, 'MaxComf': 8, 'MaxPrac': 10},
    ('<3', '<1'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': 3, 'MaxRel': 7, 'MaxComf': 10, 'MaxPrac': 12},
    ('<4', '<1'): {'MinPrac': -3, 'MinComf': -1, 'MinRel': 5, 'MaxRel': 9, 'MaxComf': 12, 'MaxPrac': 14},
    ('<5', '<1'): {'MinPrac': -1, 'MinComf': 1, 'MinRel': 7, 'MaxRel': 10, 'MaxComf': 13, 'MaxPrac': 15},
    ('<3', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},
    ('<4', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('<5', '<2'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 8, 'MaxPrac': 10},
    ('<4', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('<5', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('<5', '<4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},
    ('<1', '<2'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -5, 'MaxRel': -1, 'MaxComf': 3, 'MaxPrac': 5},
    ('<1', '<3'): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -7, 'MaxRel': -3, 'MaxComf': 2, 'MaxPrac': 4},
    ('<1', '<4'): {'MinPrac': -14, 'MinComf': -12, 'MinRel': -9, 'MaxRel': -5, 'MaxComf': 1, 'MaxPrac': 3},
    ('<1', '<5'): {'MinPrac': -15, 'MinComf': -13, 'MinRel': -10, 'MaxRel': -7, 'MaxComf': -1, 'MaxPrac': 1},
    ('<2', '<3'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<4'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<5'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('<3', '<4'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<3', '<5'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<4', '<5'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},

    #(1, 1): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    #(2, 2): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    #(3, 3): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    #(4, 4): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    #(5, 5): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0}
}

NOTE_CLASS_IS_BLACK = {
    0: False,
    1: True,
    2: False,
    3: True,
    4: False,
    5: False,
    6: True,
    7: False,
    8: True,
    9: False,
    10: True,
    11: False
}


def is_black(midi_number):
    modulo_number = midi_number % 12
    return NOTE_CLASS_IS_BLACK[modulo_number]


def is_white(midi_number):
    return not is_black(midi_number=midi_number)


def is_between(midi, midi_left, midi_right):
    if not midi or not midi_left or not midi_right:
        return False

    if midi_right > midi > midi_left:
        return True
    if midi_right < midi < midi_left:
        return True

    return False


class Parncutt(D.Dactyler):
    def init_component_costs(self):
        self._costs = {
            'str': 0,
            'sma': 0,
            'lar': 0,
            'pcc': 0,
            'pcs': 0,
            'wea': 0,
            '345': 0,
            '3t4': 0,
            'bl4': 0,
            'bl1': 0,
            'bl5': 0,
            'pa1': 0,
        }

    def __init__(self):
        super().__init__()
        self._costs = {}
        self.init_component_costs()

    @staticmethod
    def transition_allowed(from_midi, from_digit, to_midi, to_digit):
        required_span = to_midi - from_midi

        # Repeated notes are always played with the same finger.
        if required_span == 0:
            if from_digit == to_digit:
                print("Good {0} to {1} trans of span {2}".format(from_digit, to_digit, required_span))
                return True
            else:
                print("BAD {0} to {1} trans of span {2}".format(from_digit, to_digit, required_span))
                return False

        if (from_digit, to_digit) not in finger_span:
            print("BAD {0} to {1} trans of span {2}".format(from_digit, to_digit, required_span))
            return False

        max_prac = finger_span[(from_digit, to_digit)]['MaxPrac']
        min_prac = finger_span[(from_digit, to_digit)]['MinPrac']
        if min_prac <= required_span <= max_prac:
            print("Good {0} to {1} trans of span {2} (between {3} and {4})".format(from_digit,
                                                                                 to_digit,
                                                                                 required_span,
                                                                                 min_prac,
                                                                                 max_prac))
            return True

        print("BAD {0} to {1} trans of span {2} (between {3} and {4})".format(from_digit,
                                                                            to_digit,
                                                                            required_span,
                                                                            min_prac,
                                                                            max_prac))
        return False

    @staticmethod
    def prune_dead_end(g, node_id):
        if node_id == 0:
            return
        if len(list(g.successors(node_id))) > 0:
            return
        predecessor_node_ids = g.predecessors(node_id)
        g.remove_node(node_id)
        for predecessor_id in predecessor_node_ids:
            Parncutt.prune_dead_end(g=g, node_id=predecessor_id)

    @staticmethod
    def fingered_note_nx_graph(segment, hand, handed_first_digit, handed_last_digit):
        g = nx.DiGraph()
        g.add_node(0, start=1, midi=0, digit="-")
        prior_slice_node_ids = list()
        prior_slice_node_ids.append(0)
        last_note_in_segment_index = len(segment) - 1
        note_in_segment_index = 0
        node_id = 1
        on_last_prefingered_note = False
        for note in segment:
            on_first_prefingered_note = False
            slice_node_ids = list()

            if note_in_segment_index == 0 and handed_first_digit:
                on_first_prefingered_note = True

            if note_in_segment_index == last_note_in_segment_index and handed_last_digit:
                on_last_prefingered_note = True

            viable_prior_node_ids = dict()
            for digit in (THUMB, INDEX, MIDDLE, RING, LITTLE):
                handed_digit = hand + str(digit)
                if on_last_prefingered_note and handed_digit != handed_last_digit:
                    continue
                if on_first_prefingered_note and handed_digit != handed_first_digit:
                    continue
                g.add_node(node_id, midi=note.pitch.midi, digit=handed_digit)
                slice_node_ids.append(node_id)
                if 0 in prior_slice_node_ids:
                    g.add_edge(0, node_id)
                else:
                    incoming_count = 0
                    for prior_node_id in prior_slice_node_ids:
                        prior_node = g.nodes[prior_node_id]
                        prior_midi = prior_node["midi"]
                        prior_handed_digit = prior_node["digit"]
                        if Parncutt.transition_allowed(from_midi=prior_midi, from_digit=prior_handed_digit,
                                                       to_midi=note.pitch.midi, to_digit=handed_digit):
                            g.add_edge(prior_node_id, node_id)
                            incoming_count += 1
                            viable_prior_node_ids[prior_node_id] = True
                    if incoming_count == 0:
                        g.remove_node(node_id)
                        slice_node_ids.remove(node_id)
                node_id += 1

            for pni in prior_slice_node_ids:
                if pni not in viable_prior_node_ids:
                    Parncutt.prune_dead_end(g, pni)

            if len(slice_node_ids) > 0:
                prior_slice_node_ids = copy.copy(slice_node_ids)
            note_in_segment_index += 1

        g.add_node(node_id, end=1, midi=0, digit="-")
        for prior_node_id in prior_slice_node_ids:
            g.add_edge(prior_node_id, node_id)

        return g

    def trigram_node_cost(self, midi_1, handed_digit_1, midi_2, handed_digit_2, midi_3, handed_digit_3):
        cost = 0

        digit_1 = re.sub(r'[<>]', '', handed_digit_1)
        digit_2 = re.sub(r'[<>]', '', handed_digit_2)
        digit_3 = re.sub(r'[<>]', '', handed_digit_3)

        if digit_1 != '-':
            semitone_diff_12 = midi_2 - midi_1
            max_comf_12 = finger_span[(handed_digit_1, handed_digit_2)]['MaxComf']
            min_comf_12 = finger_span[(handed_digit_1, handed_digit_2)]['MinComf']
            min_rel_12 = finger_span[(handed_digit_1, handed_digit_2)]['MinRel']
            max_rel_12 = finger_span[(handed_digit_1, handed_digit_2)]['MaxRel']

            # Rule 1 ("Stretch")
            if semitone_diff_12 > max_comf_12:
                self._costs['str'] = 2 * (semitone_diff_12 - max_comf_12) * WEIGHT[1]
            elif semitone_diff_12 < min_comf_12:
                self._costs['str'] = 2 * (min_comf_12 - semitone_diff_12) * WEIGHT[1]

            span_penalty = 2
            if digit_1 == THUMB or digit_2 == THUMB:
                span_penalty = 1

            # Rule 2 ("Small-Span")
            if digit_1 and semitone_diff_12 < min_rel_12:
                self._costs['sma'] = span_penalty * (min_rel_12 - semitone_diff_12) * WEIGHT[2]

            # Rule 3 ("Large-Span")
            if digit_1 and semitone_diff_12 > max_rel_12:
                self._costs['lar'] = span_penalty * (semitone_diff_12 - min_rel_12) * WEIGHT[3]

            # Rule 6 ("Weak-Finger")
            if digit_1 == RING or digit_1 == LITTLE:
                self._costs['wea'] = WEIGHT[6]

            # Rule 8 ("Three-to-Four")
            if digit_1 == MIDDLE and digit_2 == RING:
                self._costs['3t4'] = WEIGHT[8]

            # Rule 9 ("Four-on-Black")
            if (digit_1 == RING and is_black(midi_1) and digit_2 == MIDDLE and is_white(midi_2)) or \
                    (digit_1 == MIDDLE and is_white(midi_1) and digit_2 == RING and is_black(midi_2)):
                self._costs['bl4'] = WEIGHT[9]

            # Rule 12 ("Thumb-Passing")
            thumb_passing_cost = 1
            if is_black(midi_1) != is_black(midi_2):
                thumb_passing_cost = 3
            if (midi_1 < midi_2 and digit_2 == THUMB) or (midi_2 < midi_1 and digit_1 == THUMB):
                self._costs['pa1'] = thumb_passing_cost * WEIGHT[12]

        if digit_1 != '-' and digit_3 != '-' and digit_1 != digit_3:
            semitone_diff_13 = midi_3 - midi_1
            max_comf_13 = finger_span[(handed_digit_1, handed_digit_3)]['MaxComf']
            min_comf_13 = finger_span[(handed_digit_1, handed_digit_3)]['MinComf']
            max_prac_13 = finger_span[(handed_digit_1, handed_digit_3)]['MaxPrac']
            min_prac_13 = finger_span[(handed_digit_1, handed_digit_3)]['MinPrac']

            # Rule 4 ("Position-Change-Count)"
            if semitone_diff_13 > max_comf_13:
                if digit_2 == THUMB and \
                        is_between(midi_2, midi_1, midi_3) and semitone_diff_13 > max_prac_13:
                    self._costs['pcc'] = 2 * WEIGHT[4]  # A "full change"
                else:
                    self._costs['pcc'] = 1 * WEIGHT[4]  # A "half change"
            elif semitone_diff_13 < min_comf_13:
                if digit_2 == THUMB and is_between(midi_2, midi_1, midi_3) and semitone_diff_13 < min_prac_13:
                    self._costs['pcc'] = 2 * WEIGHT[4]  # A "full change"
                else:
                    self._costs['pcc'] = 1 * WEIGHT[4]  # A "half change"

            # Rule 5 ("Position-Change-Size")
            if semitone_diff_13 < min_comf_13:
                self._costs['pcs'] = (min_comf_13 - semitone_diff_13) * WEIGHT[5]
            elif semitone_diff_13 > max_comf_13:
                self._costs['pcs'] = (semitone_diff_13 - max_comf_13) * WEIGHT[5]

            # Rule 7 ("Three-Four-Five")
            hard_sequence = True
            hard_finger = MIDDLE  # That is, 3.
            for finger in sorted((digit_1, digit_2, digit_3)):
                if finger != hard_finger:
                    hard_sequence = False
                hard_finger += 1
            if hard_sequence:
                self._costs['345'] = WEIGHT[7]

        black_key_cost = 1
        if digit_1 and digit_2 and digit_3:
            # Rule 10 ("Thumb-on-Black")
            if digit_1 and is_white(midi_1):
                black_key_cost += 2
            if digit_3 and is_white(midi_3):
                black_key_cost += 2
            if digit_2 == THUMB and is_black(midi_2):
                self._costs['bl1'] += black_key_cost * WEIGHT[10]

        # Rule 11 ("Five-on-Black")
        if digit_2 == LITTLE and is_black(midi_2):
            self._costs['bl5'] += black_key_cost * WEIGHT[11]

        for cost_key in self._costs:
            cost += self._costs[cost_key]
        return cost

    def trigram_nx_graph(self, fn_graph):
        g = nx.DiGraph()
        g.add_node(0, start=1)
        level_1_slice = [0]
        prior_trigram_slice = [0]
        next_trigram_node_id = 1
        done = False
        while not done:
            slice_trigram_id_for_key = dict()
            next_level_1_slice = list()
            for level_1_node_id in level_1_slice:
                level_2_nodes = list(fn_graph.successors(level_1_node_id))
                for level_2_node_id in level_2_nodes:
                    next_level_1_slice.append(level_2_node_id)
                    level_3_nodes = list(fn_graph.successors(level_2_node_id))
                    for level_3_node_id in level_3_nodes:
                        node_1 = fn_graph.nodes[level_1_node_id]
                        node_2 = fn_graph.nodes[level_2_node_id]
                        node_3 = fn_graph.nodes[level_3_node_id]
                        if node_3['digit'] == '-':
                            done = True
                        digit_1 = node_1['digit']
                        digit_2 = node_2['digit']
                        digit_3 = node_3['digit']
                        midi_1 = node_1['midi']
                        midi_2 = node_2['midi']
                        midi_3 = node_3['midi']
                        slice_trigram_key = (digit_1, digit_2, digit_3)
                        if slice_trigram_key not in slice_trigram_id_for_key:
                            g.add_node(next_trigram_node_id,
                                       midi_1=midi_1, digit_1=digit_1,
                                       midi_2=midi_2, digit_2=digit_2,
                                       midi_3=midi_3, digit_3=digit_3)
                            slice_trigram_id_for_key[slice_trigram_key] = next_trigram_node_id
                            trigram_node_id = next_trigram_node_id
                            next_trigram_node_id += 1
                        else:
                            trigram_node_id = slice_trigram_id_for_key[slice_trigram_key]
                        for prior_trigram_node_id in prior_trigram_slice:
                            if 'start' in g.nodes[prior_trigram_node_id] or \
                                (g.nodes[prior_trigram_node_id]['digit_2'] == digit_1 and
                                 g.nodes[prior_trigram_node_id]['digit_3'] == digit_2):
                                edge_weight = self.trigram_node_cost(midi_1=midi_1, handed_digit_1=digit_1,
                                                                     midi_2=midi_2, handed_digit_2=digit_2,
                                                                     midi_3=midi_3, handed_digit_3=digit_3)
                                g.add_edge(prior_trigram_node_id, trigram_node_id, weight=edge_weight)
            level_1_slice = next_level_1_slice
            prior_trigram_slice = []
            for node_key, node_id in slice_trigram_id_for_key.items():
                prior_trigram_slice.append(node_id)

        g.add_node(next_trigram_node_id, end=1)
        for prior_trigram_node_id in prior_trigram_slice:
            g.add_edge(prior_trigram_node_id, next_trigram_node_id)

        return g, next_trigram_node_id

    def k_best_advice(g, target_id, k):
        """
        Apply standard shortest path algorithms to determine set of optimal fingerings based on
        a standardized networkx graph.
        :param g: The weighted trinode graph. Weights must be specified via a "weight" edge parameter. Fingerings
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

    def generate_segment_advice(self, segment, staff, offset, handed_first_digit=None, handed_last_digit=None, k=None):
        """
        Generate a set of k ranked fingering suggestions for the given segment.
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
        if len(segment) == 1:
            note_list = DNote.note_list(segment)
            abcdf = D.Dactyler.one_note_advise(note_list[0], staff=staff,
                                               first_digit=handed_first_digit,
                                               last_digit=handed_last_digit)
            return [abcdf], [0]

        hand = ">"
        if staff == "lower":
            hand = "<"

        fn_graph = Parncutt.fingered_note_nx_graph(segment=segment, hand=hand,
                                                   handed_first_digit=handed_first_digit,
                                                   handed_last_digit=handed_last_digit)
        # nx.write_graphml(fn_graph, "/Users/dave/goo.graphml")

        trigram_graph, target_node_id = self.trigram_nx_graph(fn_graph=fn_graph)
        # nx.write_graphml(trigram_graph, "/Users/dave/gootri.graphml")
        return D.Dactyler.generate_standard_graph_advice(g=trigram_graph, target_id=target_node_id, k=k)

