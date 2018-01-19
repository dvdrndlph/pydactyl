__author__ = 'David Randolph'
# Copyright (c) 2014-2015 David A. Randolph.
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

import music21
import util
from nltk import corpus
# from node import GraphNode
import networkx as nx
import copy
import matplotlib.pyplot as plt

import pprint
import time
import re

#TEST_CORPUS = '/Users/timgut/Projects/node.js/node-web-service/dd/corpora/parncutt.abc'
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
    (1, 2): {'MinPrac': -5, 'MinComf': -3, 'MinRel': 1, 'MaxRel': 5, 'MaxComf': 8, 'MaxPrac': 10},
    (1, 3): {'MinPrac': -4, 'MinComf': -2, 'MinRel': 3, 'MaxRel': 7, 'MaxComf': 10, 'MaxPrac': 12},
    (1, 4): {'MinPrac': -3, 'MinComf': -1, 'MinRel': 5, 'MaxRel': 9, 'MaxComf': 12, 'MaxPrac': 14},
    (1, 5): {'MinPrac': -1, 'MinComf': 1, 'MinRel': 7, 'MaxRel': 10, 'MaxComf': 13, 'MaxPrac': 15},
    (2, 3): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},
    (2, 4): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    (2, 5): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 8, 'MaxPrac': 10},
    (3, 4): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    (3, 5): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    (4, 5): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},
    (2, 1): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -5, 'MaxRel': -1, 'MaxComf': 3, 'MaxPrac': 5},
    (3, 1): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -7, 'MaxRel': -3, 'MaxComf': 2, 'MaxPrac': 4},
    (4, 1): {'MinPrac': -14, 'MinComf': -12, 'MinRel': -9, 'MaxRel': -5, 'MaxComf': 1, 'MaxPrac': 3},
    (5, 1): {'MinPrac': -15, 'MinComf': -13, 'MinRel': -10, 'MaxRel': -7, 'MaxComf': -1, 'MaxPrac': 1},
    (3, 2): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    (4, 2): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    (5, 2): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    (4, 3): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    (5, 3): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    (5, 4): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    #(1, 1): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    #(2, 2): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    #(3, 3): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    #(4, 4): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    #(5, 5): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0}
}

class FingeredNote:
    def __init__(self, m21note=None, finger=None):
        self.m21note = m21note
        self.finger = finger

    def get_midi(self):
        if self.m21note:
            return self.m21note.midi
        return None

    def get_finger(self):
        if self.finger:
            return self.finger
        return None

    def get_tuple_str(self):
        return "%s:%s" % (self.get_midi(), self.get_finger())

    def is_black_key(self):
        return util.is_black(self.m21note)
        # if not self.m21note:
            # return False
        # if self.note21note.accidental:
            # acc = self.note21note.accidental
            # if int(acc.alter) % 2 == 1:
                # return True
        # return False

    def is_white_key(self):
        if not self.note21note:
            return False
        return not self.is_black_key()

    def is_between(self, note_x, note_y):
        if not note_x or not note_y:
            return False

        midi = self.get_midi()
        midi_x = note_x.get_midi()
        midi_y = note_y.get_midi()

        if not midi or not midi_x or not midi_y:
            return False

        if midi > midi_x and midi < midi_y:
            return True
        if midi < midi_x and midi > midi_y:
            return True

        return False


class FingeredNoteNode(GraphNode):
    def __init__(self, fingered_note=None, terminal=None):
        GraphNode.__init__(self, terminal=terminal)
        self.fingered_note = fingered_note if fingered_note else FingeredNote()

    def dump(self, tab_count=1, recurse=True):
        #time.sleep(1)

        tab_str = ''
        for i in range(tab_count):
            tab_str += "\t"

        if self.is_end():
            print tab_str + " END"
            return

        if self.is_start():
            print tab_str + " START" +\
                " Kids: " + str(len(self.next_nodes))
            tab_count += 1
            for node in self.next_nodes:
                node.dump(tab_count)
            return

        print tab_str +\
            " MIDI: " + str(self.fingered_note.get_midi()) +\
            " Finger: " + str(self.fingered_note.get_finger()) +\
            " Kids: " + str(len(self.next_nodes))
        tab_count += 1

        if recurse:
            for node in self.next_nodes:
                node.dump(tab_count)

    def get_finger(self):
        if self.fingered_note:
            return self.fingered_note.get_finger()
        return None

    def get_midi(self):
        if self.fingered_note:
            return self.fingered_note.get_midi()
        return None

    def get_tuple(self):
        return self.get_midi(), self.get_finger()

    def get_tuple_str(self):
        return "%s:%s" % (self.get_midi(), self.get_finger())

    def get_paths_as_str(self):
        str = self.get_tuple_str()
        if self.is_end():
            str += "\n"
        str += ","
        for kid in self.next_nodes:
            str += kid.get_paths_as_str()
        return str

    def get_paths(self, paths=[], path_input=[]):
        path = list(path_input)
        path.append(self.get_tuple())

        for node in self.next_nodes:
            if node.is_end():
                path_to_append = list(path)
                path_to_append.append(node.get_tuple())
                paths.append(path_to_append)
            else:
                node.get_paths(paths, path)
        return paths

    def can_transition_to(self, next_node):
        if self.is_start() or next_node.is_end():
            return True
        if self.is_end():
            return False

        if next_node.get_finger() == self.get_finger():
            return False  # Limitation of model.

        required_span = next_node.get_midi() - self.get_midi()
        max_prac = finger_span[(self.get_finger(), next_node.get_finger())]['MaxPrac']
        min_prac = finger_span[(self.get_finger(), next_node.get_finger())]['MinPrac']
        if min_prac <= required_span <= max_prac:
            print "Good {0}->{1} trans: {2} (between {3} and {4})".format(self.get_finger(),
                                                                         next_node.get_finger(),
                                                                         required_span,
                                                                         min_prac,
                                                                         max_prac)
            return True

        print "BAD {0}->{1} trans: {2} (between {3} and {4})".format(self.get_finger(),
                                                                    next_node.get_finger(),
                                                                    required_span,
                                                                    min_prac,
                                                                    max_prac)
        return False

    @staticmethod
    def build_from_score(score):
        print score
        start_node = FingeredNoteNode(terminal=GraphNode.START)
        parent_nodes = [start_node]
        for n in score[1].getElementsByClass(music21.note.Note):
            print str(n.midi)
            trellis_nodes = []
            is_in_next_column = {}
            for f in (THUMB, INDEX, MIDDLE, RING, LITTLE):
                fn = FingeredNote(n, f)
                fnn = FingeredNoteNode(fn)
                trellis_nodes.append(fnn)
                is_in_next_column[f] = False

            for parent_node in parent_nodes:
                childless = True
                for trellis_node in trellis_nodes:
                    if parent_node.connect_to(trellis_node):
                        is_in_next_column[trellis_node.fingered_note.finger] = True
                        childless = False
                if childless:
                    parent_node.disconnect()

            new_parent_nodes = []
            for trellis_node in trellis_nodes:
                if is_in_next_column[trellis_node.fingered_note.finger]:
                    new_parent_nodes.append(trellis_node)
            parent_nodes = new_parent_nodes

        end_node = FingeredNoteNode(terminal=GraphNode.END)
        for parent_node in parent_nodes:
            parent_node.connect_to(end_node)

        return start_node


class TrigramNode(GraphNode):
    end_node = None

    def __init__(self, note_1=None, note_2=None, note_3=None, terminal=None, layer_index=None):
        GraphNode.__init__(self, terminal=terminal)
        if note_1:
            self.note_1 = note_1
        else:
            self.note_1 = FingeredNote()
        if note_2:
            self.note_2 = note_2
        else:
            self.note_2 = FingeredNote()
        if note_3:
            self.note_3 = note_3
        else:
            self.note_3 = FingeredNote()

        assert isinstance(self.note_1, FingeredNote)
        assert isinstance(self.note_2, FingeredNote)
        assert isinstance(self.note_3, FingeredNote)

        if self.is_end():
            TrigramNode.end_node = self
        if self.is_start():
            self.layer_index = 0
        else:
            self.layer_index = layer_index

        self.costs = {
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

        self.cost = 0
        self.cost = self.calculate_node_cost()
        print self

    def __str__(self):
        finger_1 = self.note_1.get_finger() if self.note_1.get_finger() else '-'
        finger_2 = self.note_2.get_finger() if self.note_2.get_finger() else '-'
        finger_3 = self.note_3.get_finger() if self.note_3.get_finger() else '-'
        if self.get_midi():
            midi = self.get_midi()
        else:
            midi = self.terminal
        my_str = "{0}:{1}{2}{3} {4} {5}".format(midi,
                                                finger_1,
                                                finger_2,
                                                finger_3,
                                                self.cost,
                                                self.costs)
        return my_str

    def get_note_1(self):
        return self.note_1

    def get_note_2(self):
        return self.note_2

    def get_note_3(self):
        return self.note_3

    def get_finger(self):
        return self.note_2.get_finger()

    def get_midi(self):
        return self.note_2.get_midi()

    def dump(self, tab_count=0, recurse=True):
        # time.sleep(1)
        midi = self.note_2.get_midi()
        tab_str = ''
        for i in range(tab_count):
            tab_str += "  "
        str = "%s%s Note: %s Kids: %s" % (tab_str, self.get_display_key(), midi, len(self.next_nodes))
        print str

        tab_count += 1

        if recurse:
            for node in self.next_nodes:
                node.dump(tab_count)

    def add_to_nx_graph(self, nx_graph, trigram_node_from=None):
        # time.sleep(1)
        assert isinstance(nx_graph, nx.DiGraph)
        nx_graph.add_node(self.get_hashable_key())
        if trigram_node_from:
            nx_node_from = trigram_node_from.get_hashable_key()
            nx_node_to = self.get_hashable_key()
            nx_graph.add_edge(nx_node_from, nx_node_to, cost=self.cost)

        for node in self.next_nodes:
            node.add_to_nx_graph(nx_graph, trigram_node_from=self)

    def get_nx_graph(self):
        nx_graph = nx.DiGraph()
        self.add_to_nx_graph(nx_graph)
        return nx_graph

    def add_to_yen_graph(self, yen_graph, trigram_node_from=None):
        # time.sleep(1)
        if trigram_node_from:
            yen_node_from = trigram_node_from.get_hashable_key()
            yen_node_to = self.get_hashable_key()
            yen_graph.add_edge(yen_node_from, yen_node_to, cost=self.cost)

        for node in self.next_nodes:
            node.add_to_yen_graph(yen_graph, trigram_node_from=self)

    def get_yen_graph(self):
        yen_graph = YenKSP.DiGraph()
        self.add_to_yen_graph(yen_graph)
        return yen_graph

    def calculate_node_cost(self):
        if self.is_start() or self.is_end():
            return 0

        note_1 = self.note_1
        note_2 = self.note_2
        note_3 = self.note_3
        finger_1 = note_1.get_finger()
        finger_2 = note_2.get_finger()
        finger_3 = note_3.get_finger()
        midi_1 = int(note_1.get_midi()) if note_1.get_midi() else NO_MIDI
        midi_2 = int(note_2.get_midi())
        midi_3 = int(note_3.get_midi()) if note_3.get_midi() else NO_MIDI

        cost = 0

        if finger_1:
            semitone_diff_12 = midi_2 - midi_1
            max_comf_12 = finger_span[(finger_1, finger_2)]['MaxComf']
            min_comf_12 = finger_span[(finger_1, finger_2)]['MinComf']
            min_rel_12 = finger_span[(finger_1, finger_2)]['MinRel']
            max_rel_12 = finger_span[(finger_1, finger_2)]['MaxRel']

            # Rule 1 ("Stretch")
            if semitone_diff_12 > max_comf_12:
                self.costs['str'] = 2 * (semitone_diff_12 - max_comf_12) * WEIGHT[1]
            elif semitone_diff_12 < min_comf_12:
                self.costs['str'] = 2 * (min_comf_12 - semitone_diff_12) * WEIGHT[1]

            span_penalty = 2
            if finger_1 == THUMB or finger_2 == THUMB:
                span_penalty = 1

            # Rule 2 ("Small-Span")
            if finger_1 and semitone_diff_12 < min_rel_12:
                self.costs['sma'] = span_penalty * (min_rel_12 - semitone_diff_12) * WEIGHT[2]

            # Rule 3 ("Large-Span")
            if finger_1 and semitone_diff_12 > max_rel_12:
                self.costs['lar'] = span_penalty * (semitone_diff_12 - min_rel_12) * WEIGHT[3]

            # Rule 6 ("Weak-Finger")
            if finger_1 == RING or finger_1 == LITTLE:
                self.costs['wea'] = WEIGHT[6]

            # Rule 8 ("Three-to-Four")
            if finger_1 == MIDDLE and finger_2 == RING:
                self.costs['3t4'] = WEIGHT[8]

            # Rule 9 ("Four-on-Black")
            if (finger_1 == RING and note_1.is_black_key() and finger_2 == MIDDLE and note_2.is_white_key) or \
                    (finger_1 == MIDDLE and note_1.is_white_key and finger_2 == RING and note_2.is_black_key):
                self.costs['bl4'] = WEIGHT[9]

            # Rule 12 ("Thumb-Passing")
            thumb_passing_cost = 1
            if note_1.is_black_key() != note_2.is_black_key():
                thumb_passing_cost = 3
            if (midi_1 < midi_2 and finger_2 == THUMB) or (midi_2 < midi_1 and finger_1 == THUMB):
                self.costs['pa1'] = thumb_passing_cost * WEIGHT[12]

        if finger_1 and finger_3 and finger_1 != finger_3:
            semitone_diff_13 = midi_3 - midi_1
            max_comf_13 = finger_span[(finger_1, finger_3)]['MaxComf']
            min_comf_13 = finger_span[(finger_1, finger_3)]['MinComf']
            max_prac_13 = finger_span[(finger_1, finger_3)]['MaxPrac']
            min_prac_13 = finger_span[(finger_1, finger_3)]['MinPrac']

            # Rule 4 ("Position-Change-Count)"
            if semitone_diff_13 > max_comf_13:
                if finger_2 == THUMB and \
                        note_2.is_between(note_1, note_3) and \
                                semitone_diff_13 > max_prac_13:
                    self.costs['pcc'] = 2 * WEIGHT[4]  # A "full change"
                else:
                    self.costs['pcc'] = 1 * WEIGHT[4]  # A "half change"
            elif semitone_diff_13 < min_comf_13:
                if finger_2 == THUMB and note_2.is_between(note_1, note_3) and \
                        semitone_diff_13 < min_prac_13:
                    self.costs['pcc'] = 2 * WEIGHT[4]  # A "full change"
                else:
                    self.costs['pcc'] = 1 * WEIGHT[4]  # A "half change"

            # Rule 5 ("Position-Change-Size")
            if semitone_diff_13 < min_comf_13:
                self.costs['pcs'] = (min_comf_13 - semitone_diff_13) * WEIGHT[5]
            elif semitone_diff_13 > max_comf_13:
                self.costs['pcs'] = (semitone_diff_13 - max_comf_13) * WEIGHT[5]

            # Rule 7 ("Three-Four-Five")
            hard_sequence = True
            hard_finger = MIDDLE  # That is, 3.
            for finger in sorted((finger_1, finger_2, finger_3)):
                if finger != hard_finger:
                    hard_sequence = False
                hard_finger += 1
            if hard_sequence:
                self.costs['345'] = WEIGHT[7]

        black_key_cost = 1
        if finger_1 and finger_2 and finger_3:
            # Rule 10 ("Thumb-on-Black")
            if finger_1 and note_1.is_white_key:
                black_key_cost += 2
            if finger_3 and note_3.is_white_key:
                black_key_cost += 2
            if finger_2 == THUMB and note_2.is_black_key():
                self.costs['bl1'] += black_key_cost * WEIGHT[10]

        # Rule 11 ("Five-on-Black")
        if finger_2 == LITTLE and note_2.is_black_key():
            self.costs['bl5'] += black_key_cost * WEIGHT[11]

        for cost_key in self.costs:
            cost += self.costs[cost_key]
        return cost

    def get_cost(self):
        return self.cost

    def get_three_note_key(self, note_1, note_2, note_3):
        str_1 = note_1.get_tuple_str()
        str_2 = note_2.get_tuple_str()
        str_3 = note_3.get_tuple_str()
        key = "%s,%s,%s" % (str_1, str_2, str_3)
        return key

    def get_key(self):
        return self.get_three_note_key(self.note_1, self.note_2, self.note_3)

    def get_display_key(self):
        n1 = self.note_1.get_finger() if self.note_1.get_finger() else '-'
        n2 = self.note_2.get_finger() if self.note_2.get_finger() else '-'
        n3 = self.note_3.get_finger() if self.note_3.get_finger() else '-'
        key = "%s/%s%s%s/%s" % (self.layer_index, n1, n2, n3, self.get_cost())
        return key

    def get_hashable_key(self):
        key = str(self.layer_index) + '/' + self.get_key() + '=' + str(self.get_cost())
        return key

    def build(self, fnn, fnn_path_input=[], trigram_layers=[], trigram_layer_index=0):
        assert isinstance(fnn, FingeredNoteNode)
        if not trigram_layers:
            parent_key = self.get_key()
            trigram_layers.append({parent_key: self})

        # print "Trigram layers: %s" % trigram_layer_index
        # pprint.pprint(trigram_layers[trigram_layer_index])
        # time.sleep(1)
        fnn_path = list(fnn_path_input)
        fnn_path.append(fnn)
        trigram_node = self
        if len(fnn_path) > 2:
            note_1 = fnn_path[-3].fingered_note
            note_2 = fnn_path[-2].fingered_note
            note_3 = fnn_path[-1].fingered_note
            trigram_key = self.get_three_note_key(note_1, note_2, note_3)
            trigram_layer_index += 1
            if trigram_layer_index < len(trigram_layers) and trigram_key in trigram_layers[trigram_layer_index]:
                trigram_node = trigram_layers[trigram_layer_index][trigram_key]
                nodes_in_layer = len(trigram_layers[trigram_layer_index])
                print "INDEX: %s(%s) REUSE node: %s" % (trigram_layer_index, nodes_in_layer, trigram_key)
            else:
                trigram_node = TrigramNode(note_1, note_2, note_3, layer_index=trigram_layer_index)
                if trigram_layer_index < len(trigram_layers):
                    trigram_layers[trigram_layer_index][trigram_key] = trigram_node
                    nodes_in_layer = len(trigram_layers[trigram_layer_index])
                    print "INDEX: %s(%s) NEW  node: %s" % (trigram_layer_index, nodes_in_layer, trigram_key)
                else:
                    trigram_layers.append({trigram_key: trigram_node})
                    nodes_in_layer = len(trigram_layers[trigram_layer_index])
                    print "INDEX: %s(%s) NEW  node: %s on new layer" % (trigram_layer_index, nodes_in_layer, trigram_key)
            self.connect_to(trigram_node)

        if fnn.next_nodes:
            # print "KIDS: %s" % len(fnn.next_nodes)
            for kid in fnn.next_nodes:
                trigram_node.build(kid, fnn_path, trigram_layers, trigram_layer_index)
        elif not self.is_start():
            trigram_layer_index += 1
            terminal_node = TrigramNode(terminal=GraphNode.END, layer_index=trigram_layer_index)
            terminal_key = terminal_node.get_key()
            if trigram_layer_index < len(trigram_layers) and terminal_key in trigram_layers[trigram_layer_index]:
                terminal_node = trigram_layers[trigram_layer_index][terminal_key]
                nodes_in_layer = len(trigram_layers[trigram_layer_index])
                print "INDEX: %s(%s) REUSE node: %s" % (trigram_layer_index, nodes_in_layer, terminal_key)
            else:
                if trigram_layer_index < len(trigram_layers):
                    trigram_layers[trigram_layer_index][terminal_key] = terminal_node
                    nodes_in_layer = len(trigram_layers[trigram_layer_index])
                    print "INDEX: %s(%s) Reuse node: %s" % (trigram_layer_index, nodes_in_layer, terminal_key)
                else:
                    trigram_layers.append({terminal_key: terminal_node})
                    nodes_in_layer = len(trigram_layers[trigram_layer_index])
                    print "INDEX: %s(%s) NEW   node: %s on new layer" % (trigram_layer_index, nodes_in_layer, terminal_key)
            trigram_node.connect_to(terminal_node)

    @staticmethod
    def build_from_fnn(fnn):
        trigram_graph = TrigramNode(terminal=GraphNode.START)
        trigram_graph.build(fnn)
        return trigram_graph

    @staticmethod
    def _sort_by_cost(path_list):
        cost_pattern = re.compile('.*=(\d+)$')
        cost_hash = {}
        for path in path_list:
            cost_search = re.search(cost_pattern, path[1])
            cost = cost_search.group(1)
            if not cost in cost_hash:
                cost_hash[cost] = []
            cost_hash[cost].append(path)
        sorted_list = []
        for cost in sorted(cost_hash):
            for path in sorted(cost_hash[cost]):
                sorted_list.append(path)
        return sorted_list

    @staticmethod
    def _yen_ksp(graph, source, sink, K):
        graph_copy = copy.deepcopy(graph)
        assert isinstance(graph, nx.DiGraph)
        A = []
        answer = nx.shortest_path(graph, source, sink, 'cost')
        A.append(answer)
        B = []

        print "NODES: %s EDGES: %s" % (len(graph.nodes()), len(graph.edges()))
        cost_pattern = re.compile('.*=(\d+)$')
        for k in range(1, K):
            end_index = len(A[k - 1]) - 1
            for i in range(0, end_index):
                print "k: %s i: %s end: %s" % (k, i, end_index)
                spur_node = A[k - 1][i]
                root_path = A[k - 1][0:i + 1]
                print "Spur: " + spur_node
                print "Path: " + str(root_path)

                paths_removed = []
                nodes_removed = []
                # pprint.pprint(A)
                for path in A:
                    if cmp(root_path, path[0:i + 1]) == 0:
                        from_node = path[i]
                        to_node = path[i + 1]
                        print "Remove edge: %s -> %s" % (from_node, to_node)
                        if not (from_node, to_node) in paths_removed:
                            paths_removed.append((from_node, to_node))
                            graph.remove_edge(from_node, to_node)

                for node in root_path:
                    if not node == spur_node:
                        print "Remove node: %s" % node
                        graph.remove_node(node)
                        nodes_removed.append(node)

                print "NODES: %s EDGES: %s" % (len(graph.nodes()), len(graph.edges()))
                print(graph.edges())
                try:
                    spur_path = nx.shortest_path(graph, spur_node, sink, 'cost')
                    total_path = root_path + spur_path[1:]
                    print "Root path: %s Spur path: %s" % (root_path, spur_path)
                    print "Total path: %s" % total_path
                    B.append(total_path)
                except nx.NetworkXNoPath as e:
                    print "Ain't got no t-bone: {0}".format(e.message)

                # FIXME: The surgical approach commented out does not work, as
                # links are blasted when nodes are removed. We need to save more state
                # to be able to recover. For now, we just restore from a copy of the graph.
                # Restore edges and nodes removed previously
                # for node in nodes_removed:
                    # print "Add node: %s" % node
                    # graph.add_node(node)
                # for path in paths_removed:
                    # print "Add edge: %s -> %s" % path
                    # cost_search = re.search('.*=(\d+)$', path[1])
                    # cost = cost_search.group(1)
                    # graph.add_edge(path[0], path[1], cost=cost)
                # print "NODES: %s EDGES: %s" % (len(graph.nodes()), len(graph.edges()))
                # print(graph.edges())
                graph = copy.deepcopy(graph_copy)

            if not B:
                break
            B = TrigramNode._sort_by_cost(B)
            A.append(B.pop(0))
        return A

    def get_k_best_fingerings(self, k=1):
        nx_graph = self.get_nx_graph()
        # nx.draw(nx_graph)
        # plt.show()
        # print str(TrigramNode.end_node)

        start_key = self.get_hashable_key()
        end_key = TrigramNode.end_node.get_hashable_key()
        k_best = TrigramNode._yen_ksp(nx_graph, start_key, end_key, k)
        fingerings = []
        finger_re = re.compile("\d+/\w+:\w+,\d+:(\d),.*")
        for shorty in k_best:
            finger_numbers = []
            for node_str in shorty:
                finger_match = finger_re.search(node_str)
                if finger_match:
                    finger_number = finger_match.group(1)
                    finger_numbers.append(finger_number)
            fingerings.append("".join(finger_numbers))
        return fingerings


fnn_graphs = {}
corp = corpus.Corpus(TEST_CORPUS)
scores = corp.get_score_list()
for score in scores:
    meta = score[0]
    fnn_graphs[meta.title] = FingeredNoteNode.build_from_score(score)
trigram_graphs = []
for fnn_graph in sorted(fnn_graphs):
    trigram_graphs.append(TrigramNode.build_from_fnn(fnn_graphs[fnn_graph]))
for trigrammer in trigram_graphs:
    k_best_fingerings = trigrammer.get_k_best_fingerings(1)
    pprint.pprint(k_best_fingerings)

exit(0)
