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
import matplotlib
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


def is_black(m21_note):
    if not m21_note:
        return False
    return NOTE_CLASS_IS_BLACK[m21_note.pitch.pitchClass]


def is_white(m21_note):
    if not m21_note:
        return False
    return not is_black()


class FingeredNote:
    def __init__(self, m21_note=None, finger=None):
        self._m21_note = m21_note
        self._finger = finger

    def midi(self):
        if self._m21_note:
            return self._m21_note.pitch.midi
        return None

    def finger(self):
        if self._finger:
            return self._finger
        return None

    def tuple_str(self):
        return "%s:%s" % (self.midi(), self.finger())

    def is_black_key(self):
        return is_black(self._m21_note)
        # if not self.m21note:
            # return False
        # if self.note21note.accidental:
            # acc = self.note21note.accidental
            # if int(acc.alter) % 2 == 1:
                # return True
        # return False

    def is_white_key(self):
        if not self._m21_note:
            return False
        return not self.is_black_key()

    def is_between(self, note_x, note_y):
        if not note_x or not note_y:
            return False

        midi = self.midi()
        midi_x = note_x.midi()
        midi_y = note_y.midi()

        if not midi or not midi_x or not midi_y:
            return False

        if midi_y > midi > midi_x:
            return True
        if midi_y < midi < midi_x:
            return True

        return False


class GraphNode:
    START = 'Start'
    END = 'End'

    def __init__(self, terminal=None):
        if terminal and terminal != GraphNode.START and terminal != GraphNode.END:
            raise Exception('Bad terminal setting for GraphNode.')
        self._terminal = terminal
        self._next_nodes = []
        self._prior_nodes = []

    def terminal(self):
        return self._terminal;

    def next_nodes(self):
        return self._next_nodes;

    def prior_nodes(self):
        return self._prior_nodes;

    def is_start(self):
        if self._terminal and self._terminal == GraphNode.START:
            return True
        return False

    def is_end(self):
        if self._terminal and self._terminal == GraphNode.END:
            return True
        return False

    def connect_from(self, prior_node):
        self._prior_nodes.append(prior_node)

    def disconnect(self, next_node=None):
        if self.is_start():
            # raise Exception('Cannot disconnect: child nodes present.')
            return

        if self._next_nodes:
            self._next_nodes.remove(next_node)

        if not self._next_nodes:
            for prior_node in self._prior_nodes:
                prior_node.disconnect(next_node=self)
            self._prior_nodes = []

    # Override in derived class to enforce restrictions.
    @abstractmethod
    def can_transition_to(self, next_node):
        return True

    def connect_to(self, next_node):
        if self.can_transition_to(next_node):
            if not next_node in self._next_nodes:
                self._next_nodes.append(next_node)
                next_node.connect_from(self)
                return True
            return False
        return False

    def dump(self):
        print(str(self))


class FingeredNoteNode(GraphNode):
    def __init__(self, fingered_note=None, terminal=None):
        GraphNode.__init__(self, terminal=terminal)
        self._fingered_note = fingered_note

    def fingered_note(self):
        return self._fingered_note;

    def dump(self, tab_count=1, recurse=True):
        # time.sleep(1)

        tab_str = ''
        for i in range(tab_count):
            tab_str += "\t"

        if self.is_end():
            print(tab_str + " END")
            return

        if self.is_start():
            print(tab_str + " START" + " Kids: " + str(len(self._next_nodes)))
            tab_count += 1
            for node in self._next_nodes:
                node.dump(tab_count)
            return

        print(tab_str +
            " MIDI: " + str(self.fingered_note().midi()) +
            " Finger: " + str(self.fingered_note().finger()) +
            " Kids: " + str(len(self._next_nodes)))
        tab_count += 1

        if recurse:
            for node in self._next_nodes:
                node.dump(tab_count)

    def finger(self):
        if self._fingered_note:
            return self._fingered_note.finger()
        return None

    def midi(self):
        if self._fingered_note:
            return self._fingered_note.midi()
        return None

    def tuple(self):
        return self.midi(), self.finger()

    def tuple_str(self):
        return "%s:%s" % (self.midi(), self.finger())

    def paths_as_str(self):
        str = self.tuple_str()
        if self.is_end():
            str += "\n"
        str += ","
        for kid in self._next_nodes:
            str += kid.paths_as_str()
        return str

    def paths(self, paths=[], path_input=[]):
        path = list(path_input)
        path.append(self.tuple())

        for node in self._next_nodes:
            if node.is_end():
                path_to_append = list(path)
                path_to_append.append(node.tuple())
                paths.append(path_to_append)
            else:
                node.paths(paths, path)
        return paths

    def can_transition_to(self, next_node):
        if self.is_start() or next_node.is_end():
            return True
        if self.is_end():
            return False

        if next_node.finger() == self.finger():
            return False  # Limitation of model.

        required_span = next_node.midi() - self.midi()
        max_prac = finger_span[(self.finger(), next_node.finger())]['MaxPrac']
        min_prac = finger_span[(self.finger(), next_node.finger())]['MinPrac']
        if min_prac <= required_span <= max_prac:
            print("Good {0}->{1} trans: {2} (between {3} and {4})".format(self.finger(),
                                                                          next_node.finger(),
                                                                          required_span,
                                                                          min_prac,
                                                                          max_prac))
            return True

        print("BAD {0}->{1} trans: {2} (between {3} and {4})".format(self.finger(),
                                                                     next_node.finger(),
                                                                     required_span,
                                                                     min_prac,
                                                                     max_prac))
        return False

    @staticmethod
    def build_from_score(score):
        print(score)
        start_node = FingeredNoteNode(terminal=GraphNode.START)
        parent_nodes = [start_node]
        for n in score.getElementsByClass(music21.note.Note):
            print(str(n.pitch.midi))
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
                        is_in_next_column[trellis_node.fingered_note().finger()] = True
                        childless = False
                if childless:
                    parent_node.disconnect()

            new_parent_nodes = []
            for trellis_node in trellis_nodes:
                if is_in_next_column[trellis_node.fingered_note().finger()]:
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
            self._note_1 = note_1
        else:
            self._note_1 = FingeredNote()
        if note_2:
            self._note_2 = note_2
        else:
            self._note_2 = FingeredNote()
        if note_3:
            self._note_3 = note_3
        else:
            self._note_3 = FingeredNote()

        assert isinstance(self._note_1, FingeredNote)
        assert isinstance(self._note_2, FingeredNote)
        assert isinstance(self._note_3, FingeredNote)

        if self.is_start():
            self._layer_index = 0
        else:
            self._layer_index = layer_index

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

        self._cost = 0
        self._cost = self.calculate_node_cost()
        print(self)

    def __str__(self):
        finger_1 = self._note_1.finger() if self._note_1.finger() else '-'
        finger_2 = self._note_2.finger() if self._note_2.finger() else '-'
        finger_3 = self._note_3.finger() if self._note_3.finger() else '-'
        if self.midi():
            midi = self.midi()
        else:
            midi = self._terminal
        my_str = "{0}:{1}{2}{3} {4} {5}".format(midi,
                                                finger_1,
                                                finger_2,
                                                finger_3,
                                                self._cost,
                                                self._costs)
        return my_str

    def note_1(self):
        return self._note_1

    def note_2(self):
        return self._note_2

    def note_3(self):
        return self._note_3

    def finger(self):
        return self._note_2.finger()

    def midi(self):
        return self._note_2.midi()

    def dump(self, tab_count=0, recurse=True):
        # time.sleep(1)
        midi = self._note_2.midi()
        tab_str = ''
        for i in range(tab_count):
            tab_str += "  "
        st = "%s%s Note: %s Kids: %s" % (tab_str, self.display_key(), midi, len(self._next_nodes))
        print(st)

        tab_count += 1

        if recurse:
            for node in self._next_nodes:
                node.dump(tab_count)

    def add_to_nx_graph(self, nx_graph, trigram_node_from=None):
        # time.sleep(1)
        assert isinstance(nx_graph, nx.DiGraph)
        nx_graph.add_node(self.hashable_key())
        if trigram_node_from:
            nx_node_from = trigram_node_from.hashable_key()
            nx_node_to = self.hashable_key()
            nx_graph.add_edge(nx_node_from, nx_node_to, cost=self._cost)

        for node in self._next_nodes:
            node.add_to_nx_graph(nx_graph, trigram_node_from=self)

    def nx_graph(self):
        nx_graph = nx.DiGraph()
        self.add_to_nx_graph(nx_graph)
        return nx_graph

    def calculate_node_cost(self):
        if self.is_start() or self.is_end():
            return 0

        note_1 = self._note_1
        note_2 = self._note_2
        note_3 = self._note_3
        finger_1 = note_1.finger()
        finger_2 = note_2.finger()
        finger_3 = note_3.finger()
        midi_1 = int(note_1.midi()) if note_1.midi() else NO_MIDI
        midi_2 = int(note_2.midi())
        midi_3 = int(note_3.midi()) if note_3.midi() else NO_MIDI

        cost = 0

        if finger_1:
            semitone_diff_12 = midi_2 - midi_1
            max_comf_12 = finger_span[(finger_1, finger_2)]['MaxComf']
            min_comf_12 = finger_span[(finger_1, finger_2)]['MinComf']
            min_rel_12 = finger_span[(finger_1, finger_2)]['MinRel']
            max_rel_12 = finger_span[(finger_1, finger_2)]['MaxRel']

            # Rule 1 ("Stretch")
            if semitone_diff_12 > max_comf_12:
                self._costs['str'] = 2 * (semitone_diff_12 - max_comf_12) * WEIGHT[1]
            elif semitone_diff_12 < min_comf_12:
                self._costs['str'] = 2 * (min_comf_12 - semitone_diff_12) * WEIGHT[1]

            span_penalty = 2
            if finger_1 == THUMB or finger_2 == THUMB:
                span_penalty = 1

            # Rule 2 ("Small-Span")
            if finger_1 and semitone_diff_12 < min_rel_12:
                self._costs['sma'] = span_penalty * (min_rel_12 - semitone_diff_12) * WEIGHT[2]

            # Rule 3 ("Large-Span")
            if finger_1 and semitone_diff_12 > max_rel_12:
                self._costs['lar'] = span_penalty * (semitone_diff_12 - min_rel_12) * WEIGHT[3]

            # Rule 6 ("Weak-Finger")
            if finger_1 == RING or finger_1 == LITTLE:
                self._costs['wea'] = WEIGHT[6]

            # Rule 8 ("Three-to-Four")
            if finger_1 == MIDDLE and finger_2 == RING:
                self._costs['3t4'] = WEIGHT[8]

            # Rule 9 ("Four-on-Black")
            if (finger_1 == RING and note_1.is_black_key() and finger_2 == MIDDLE and note_2.is_white_key) or \
                    (finger_1 == MIDDLE and note_1.is_white_key and finger_2 == RING and note_2.is_black_key):
                self._costs['bl4'] = WEIGHT[9]

            # Rule 12 ("Thumb-Passing")
            thumb_passing_cost = 1
            if note_1.is_black_key() != note_2.is_black_key():
                thumb_passing_cost = 3
            if (midi_1 < midi_2 and finger_2 == THUMB) or (midi_2 < midi_1 and finger_1 == THUMB):
                self._costs['pa1'] = thumb_passing_cost * WEIGHT[12]

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
                    self._costs['pcc'] = 2 * WEIGHT[4]  # A "full change"
                else:
                    self._costs['pcc'] = 1 * WEIGHT[4]  # A "half change"
            elif semitone_diff_13 < min_comf_13:
                if finger_2 == THUMB and note_2.is_between(note_1, note_3) and \
                        semitone_diff_13 < min_prac_13:
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
            for finger in sorted((finger_1, finger_2, finger_3)):
                if finger != hard_finger:
                    hard_sequence = False
                hard_finger += 1
            if hard_sequence:
                self._costs['345'] = WEIGHT[7]

        black_key_cost = 1
        if finger_1 and finger_2 and finger_3:
            # Rule 10 ("Thumb-on-Black")
            if finger_1 and note_1.is_white_key:
                black_key_cost += 2
            if finger_3 and note_3.is_white_key:
                black_key_cost += 2
            if finger_2 == THUMB and note_2.is_black_key():
                self._costs['bl1'] += black_key_cost * WEIGHT[10]

        # Rule 11 ("Five-on-Black")
        if finger_2 == LITTLE and note_2.is_black_key():
            self._costs['bl5'] += black_key_cost * WEIGHT[11]

        for cost_key in self._costs:
            cost += self._costs[cost_key]
        return cost

    def cost(self):
        return self._cost

    @staticmethod
    def three_note_key(note_1, note_2, note_3):
        str_1 = note_1.tuple_str() if note_1 else '-'
        str_2 = note_2.tuple_str()
        str_3 = note_3.tuple_str() if note_3 else '-'
        key = "%s,%s,%s" % (str_1, str_2, str_3)
        return key

    def key(self):
        return TrigramNode.three_note_key(self._note_1, self._note_2, self._note_3)

    def display_key(self):
        n1 = self._note_1.finger() if self._note_1.finger() else '-'
        n2 = self._note_2.finger() if self._note_2.finger() else '-'
        n3 = self._note_3.finger() if self._note_3.finger() else '-'
        key = "%s/%s%s%s/%s" % (self._layer_index, n1, n2, n3, self.cost())
        return key

    def hashable_key(self):
        key = str(self._layer_index) + '/' + self.key() + '=' + str(self.cost())
        return key

    def build(self, fnn, fnn_path_input=[], trigram_layers=[], trigram_layer_index=0):
        assert isinstance(fnn, FingeredNoteNode)
        if not trigram_layers:
            parent_key = self.key()
            trigram_layers.append({parent_key: self})

        # print "Trigram layers: %s" % trigram_layer_index
        # pprint.pprint(trigram_layers[trigram_layer_index])
        # time.sleep(1)
        fnn_path = list(fnn_path_input)
        fnn_path.append(fnn)
        trigram_node = self
        if len(fnn_path) > 2:
            note_1 = fnn_path[-3].fingered_note()
            note_2 = fnn_path[-2].fingered_note()
            note_3 = fnn_path[-1].fingered_note()
            trigram_key = TrigramNode.three_note_key(note_1, note_2, note_3)
            trigram_layer_index += 1
            if trigram_layer_index < len(trigram_layers) and trigram_key in trigram_layers[trigram_layer_index]:
                trigram_node = trigram_layers[trigram_layer_index][trigram_key]
                nodes_in_layer = len(trigram_layers[trigram_layer_index])
                print("INDEX: %s(%s) REUSE node: %s" % (trigram_layer_index, nodes_in_layer, trigram_key))
            else:
                trigram_node = TrigramNode(note_1, note_2, note_3, layer_index=trigram_layer_index)
                if trigram_layer_index < len(trigram_layers):
                    trigram_layers[trigram_layer_index][trigram_key] = trigram_node
                    nodes_in_layer = len(trigram_layers[trigram_layer_index])
                    print("INDEX: %s(%s) NEW  node: %s" % (trigram_layer_index, nodes_in_layer, trigram_key))
                else:
                    trigram_layers.append({trigram_key: trigram_node})
                    nodes_in_layer = len(trigram_layers[trigram_layer_index])
                    print("INDEX: %s(%s) NEW  node: %s on new layer" % (trigram_layer_index, nodes_in_layer, trigram_key))
            self.connect_to(trigram_node)

        if fnn.next_nodes():
            # print("KIDS: %s" % len(fnn.next_nodes))
            for kid in fnn.next_nodes():
                trigram_node.build(kid, fnn_path, trigram_layers, trigram_layer_index)
        elif not self.is_start():
            trigram_layer_index += 1
            terminal_node = TrigramNode(terminal=GraphNode.END, layer_index=trigram_layer_index)
            terminal_key = terminal_node.key()
            if trigram_layer_index < len(trigram_layers) and terminal_key in trigram_layers[trigram_layer_index]:
                terminal_node = trigram_layers[trigram_layer_index][terminal_key]
                nodes_in_layer = len(trigram_layers[trigram_layer_index])
                print("INDEX: %s(%s) REUSE node: %s" % (trigram_layer_index, nodes_in_layer, terminal_key))
            else:
                if trigram_layer_index < len(trigram_layers):
                    trigram_layers[trigram_layer_index][terminal_key] = terminal_node
                    nodes_in_layer = len(trigram_layers[trigram_layer_index])
                    print("INDEX: %s(%s) Reuse node: %s" % (trigram_layer_index, nodes_in_layer, terminal_key))
                else:
                    trigram_layers.append({terminal_key: terminal_node})
                    nodes_in_layer = len(trigram_layers[trigram_layer_index])
                    print("INDEX: %s(%s) NEW   node: %s on new layer" % (trigram_layer_index, nodes_in_layer, terminal_key))
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
            if cost not in cost_hash:
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

        print("NODES: %s EDGES: %s" % (len(graph.nodes()), len(graph.edges())))
        cost_pattern = re.compile('.*=(\d+)$')
        for k in range(1, K):
            end_index = len(A[k - 1]) - 1
            for i in range(0, end_index):
                print("k: %s i: %s end: %s" % (k, i, end_index))
                spur_node = A[k - 1][i]
                root_path = A[k - 1][0:i + 1]
                print("Spur: " + spur_node)
                print("Path: " + str(root_path))

                paths_removed = []
                nodes_removed = []
                # pprint.pprint(A)
                for path in A:
                    if root_path == path[0:i + 1]:
                        from_node = path[i]
                        to_node = path[i + 1]
                        print("Remove edge: %s -> %s" % (from_node, to_node))
                        if not (from_node, to_node) in paths_removed:
                            paths_removed.append((from_node, to_node))
                            graph.remove_edge(from_node, to_node)

                for node in root_path:
                    if not node == spur_node:
                        print("Remove node: %s" % node)
                        graph.remove_node(node)
                        nodes_removed.append(node)

                print("NODES: %s EDGES: %s" % (len(graph.nodes()), len(graph.edges())))
                print(graph.edges())
                try:
                    spur_path = nx.shortest_path(graph, spur_node, sink, 'cost')
                    total_path = root_path + spur_path[1:]
                    print("Root path: %s Spur path: %s" % (root_path, spur_path))
                    print("Total path: %s" % total_path)
                    B.append(total_path)
                except nx.NetworkXNoPath as e:
                    print("Ain't got no t-bone: {0}".format(e.message))

                # FIXME: The surgical approach commented out does not work, as
                # links are blasted when nodes are removed. We need to save more state
                # to be able to recover. For now, we just restore from a copy of the graph.
                # Restore edges and nodes removed previously
                # for node in nodes_removed:
                    # print("Add node: %s" % node)
                    # graph.add_node(node)
                # for path in paths_removed:
                    # print("Add edge: %s -> %s" % path)
                    # cost_search = re.search('.*=(\d+)$', path[1])
                    # cost = cost_search.group(1)
                    # graph.add_edge(path[0], path[1], cost=cost)
                # print("NODES: %s EDGES: %s" % (len(graph.nodes()), len(graph.edges())))
                # print(graph.edges())
                graph = copy.deepcopy(graph_copy)

            if not B:
                break
            B = TrigramNode._sort_by_cost(B)
            A.append(B.pop(0))
        return A

    def k_best_fingerings(self, k=1):
        nx_graph = self.nx_graph()
        # nx.draw(nx_graph)
        # plt.show()
        # print(str(TrigramNode.end_node))

        start_key = self.hashable_key()
        end_key = TrigramNode.end_node.hashable_key()
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


class Parncutt(D.Dactyler):
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
        pass

    @staticmethod
    def fingered_note_nx_graph(segment, hand, handed_first_digit, handed_last_digit):
        g = nx.DiGraph()
        g.add_node(0, midi=0, digit="0")
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

        g.add_node(node_id, midi=0, digit="0")
        for prior_node_id in prior_slice_node_ids:
            g.add_edge(prior_node_id, node_id)

        return g

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

        nx.write_graphml(fn_graph, "/Users/dave/goo.graphml")
        # nx.draw(fn_graph)
        # trigram_graph = TrigramNode.build_from_fnn(fnn_graph)
        # k_best_fingerings = trigram_graph.k_best_fingerings(k=k)

