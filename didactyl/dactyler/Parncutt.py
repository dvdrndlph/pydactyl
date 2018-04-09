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

from .Dactyler import Dactyler


class GraphNode:
    START = 'Start'
    END = 'End'

    def __init__(self, terminal=None):
        if terminal and terminal != GraphNode.START and terminal != GraphNode.END:
            raise Exception('Bad terminal setting for GraphNode.')
        self.terminal = terminal
        self.next_nodes = []
        self.prior_nodes = []

    def is_start(self):
        if self.terminal and self.terminal == GraphNode.START:
            return True
        return False

    def is_end(self):
        if self.terminal and self.terminal == GraphNode.END:
            return True
        return False

    # Should be overridden by derived class
    def can_transition_to(self, next_node):
        if self.is_start() or next_node.is_end():
            return True
        if self.is_end():
            return False

    def connect_from(self, prior_node):
        self.prior_nodes.append(prior_node)

    def disconnect(self, next_node=None):
        if self.is_start():
            # raise Exception('Cannot disconnect: child nodes present.')
            return

        if self.next_nodes:
            self.next_nodes.remove(next_node)

        if not self.next_nodes:
            for prior_node in self.prior_nodes:
                prior_node.disconnect(next_node=self)
            self.prior_nodes = []

    # Override in derived class to enforce restrictions.
    def can_transition_to(self, next_node):
        return True

    def connect_to(self, next_node):
        if self.can_transition_to(next_node):
            if not next_node in self.next_nodes:
                self.next_nodes.append(next_node)
                next_node.connect_from(self)
                return True
            return False
        return False

    def dump(self):
        print(str(self))


class FingeredNoteNode(GraphNode):
    def __init__(self, ad_note=None, terminal=None):
        GraphNode.__init__(self, terminal=terminal)
        self.fingered_note = ad_note

    def dump(self, tab_count=1, recurse=True):
        # time.sleep(1)

        tab_str = ''
        for i in range(tab_count):
            tab_str += "\t"

        if self.is_end():
            print(tab_str + " END")
            return

        if self.is_start():
            print(tab_str + " START" + " Kids: " + str(len(self.next_nodes)))
            tab_count += 1
            for node in self.next_nodes:
                node.dump(tab_count)
            return

        print(tab_str +
            " MIDI: " + str(self.fingered_note.midi()) +
            " Finger: " + str(self.fingered_note.strike_digit()) +
            " Kids: " + str(len(self.next_nodes)))
        tab_count += 1

        if recurse:
            for node in self.next_nodes:
                node.dump(tab_count)

    def finger(self):
        if self.fingered_note:
            return self.fingered_note.strike_digit()
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
            print("Good {0}->{1} trans: {2} (between {3} and {4})".format(self.get_finger(),
                                                                          next_node.get_finger(),
                                                                          required_span,
                                                                          min_prac,
                                                                          max_prac))
            return True

        print("BAD {0}->{1} trans: {2} (between {3} and {4})".format(self.get_finger(),
                                                                     next_node.get_finger(),
                                                                     required_span,
                                                                     min_prac,
                                                                     max_prac))
        return False


class Parncutt(Dactyler.Dactyler):
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

    DEFAULT_FINGER_SPANS = {
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
        # (1, 1): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
        # (2, 2): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
        # (3, 3): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
        # (4, 4): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
        # (5, 5): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0}
    }

    def advise(self, score_index=0, staff="upper", offset=0, first_digit=None, last_digit=None):
        d_scores = self._d_corpus.d_score_list()
        if score_index >= len(d_scores):
            raise Exception("Score index out of range")

        d_score = d_scores[score_index]
        if staff == "both":
            upper_advice = self.advise(score_index=score_index, staff="upper")
            abcdf = upper_advice + "@"
            if d_score.part_count() > 1:
                lower_advice = self.advise(score_index=score_index, staff="lower")
                abcdf += lower_advice
            return abcdf

        if staff != "upper" and staff != "lower":
            raise Exception("Segregated advice is only dispensed one staff at a time.")

        if d_score.part_count() == 1:
            d_part = d_score.combined_d_part()
        else:
            # We support (segregated) left hand fingerings. By segregated, we
            # mean the right hand is dedicated to the upper staff, and the
            # left hand is dedicated to the lower staff.
            d_part = d_score.d_part(staff=staff)

        m21_stream = d_part.orderly_note_stream()