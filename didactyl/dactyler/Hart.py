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
""" Implements and enhances method described in the following paper:

        M. Hart, R. Bosch, and E. Tsai, “Finding Optimal Piano Fingerings,”
            UMAP J., vol. 21, no. 2, pp. 167–177, 2000.
            
    We enhance the method to handle repeated pitches, two staffs,
    and segregated two-hand fingering.
"""

import numpy
import re
import copy
import networkx as nx

from didactyl.dactyler import Constant
from . import Dactyler as D
from didactyl.dcorpus.DNote import DNote


class Interval:
    def __init__(self, low_color, high_color, low_finger, high_finger, semitone_delta):
        self._l_color = int(low_color)
        self._h_color = int(high_color)
        self._l_finger = int(low_finger)
        self._h_finger = int(high_finger)
        self._s = int(semitone_delta)

    def l_color(self):
        return self._l_color

    def h_color(self):
        return self._h_color

    def l_finger(self):
        return self._l_finger

    def h_finger(self):
        return self._h_finger

    def s(self):
        return self._s

    def __hash__(self):
        val = self._h_finger + 10*self._l_finger + 100*self._h_color + 1000*self._l_color + 10000*self._s
        return val

    def __eq__(self, other):
        if self.l_color() == other.l_color() and self.h_color() == other.h_color() and \
                self.l_finger() == other.l_finger() and self.h_finger() == other.h_finger() and \
                self.s() == other.s():
            return True
        return False

    def __str__(self):
        my_str = "finger: {0}->{1}, color: {2}->{3}, interval: {4}"
        my_str = my_str.format(str(self._l_finger),
                               str(self._h_finger),
                               str(self._l_color),
                               str(self._h_color),
                               str(self._s))
        return my_str

    def __repr__(self):
        return self.__str__()


class Hart(D.Dactyler):
    BIG_NUM = 999
    MAX_INTERVAL_SIZE = 12
    COST_FILE = '/Users/dave/tb2/didactyl/dd/data/tables_0.dat'

    def _define_costs(self):
        costs = {}
        for l_color in range(2):
            for h_color in range(2):
                for l_finger in range(1, 6):
                    for h_finger in range(1, 6):
                        for s in range(0, self._max_interval_size + 1):
                            # FIXME: What about 0 interval?
                            ic = Interval(l_color, h_color, l_finger, h_finger, s)
                            costs[ic] = Hart.BIG_NUM

        interval_size_finalized = False
        max_interval_size = self._max_interval_size
        max_interval_re = re.compile(r"^Max_Interval:\s+(\d+)")
        heading_re = re.compile(r"^([^_]+)_to_([^_]+)")
        cost_re_str = "^(\d)\s+(\d)"
        cost_re = None
        l_color = None
        h_color = None

        f = open(self._cost_path, 'r')
        for line in f:
            line = line.rstrip()
            if not interval_size_finalized:
                matches = re.search(max_interval_re, line)
                if matches:
                    max_interval_size = int(matches.group(1))
                interval_size_finalized = True

                for i in range(max_interval_size + 1):
                    cost_re_str += "\s+(\d+)"
                cost_re_str += "\s*$"
                cost_re = re.compile(cost_re_str)
                continue

            matches = re.search(heading_re, line)
            if matches:
                l_color = Constant.BLACK if matches.group(1) == 'Black' else Constant.WHITE
                h_color = Constant.BLACK if matches.group(2) == 'Black' else Constant.WHITE
                continue

            matches = re.search(cost_re, line)
            if matches:
                l_finger = matches.group(1)
                h_finger = matches.group(2)
                group_num = 3
                for s in range(0, max_interval_size + 1):
                    cost = matches.group(group_num)
                    interval = Interval(low_color=l_color,
                                        high_color=h_color,
                                        low_finger=l_finger,
                                        high_finger=h_finger,
                                        semitone_delta=s)
                    costs[interval] = int(cost)
                    group_num += 1
        f.close()
        return costs

    def __init__(self, cost_path=None, max_interval_size=MAX_INTERVAL_SIZE):
        super().__init__()
        self._cost_path = Hart.COST_FILE
        if cost_path:
            self._cost_path = cost_path
        self._max_interval_size = max_interval_size
        self._costs = self._define_costs()

    def generate_segment_advice(self, segment, staff, offset, handed_first_digit, handed_last_digit, k=None):
        """
        Generate a set of k ranked fingering suggestions for the given segment. Note that the original
        Hart implementation only returns one best fingering.
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
        if k is not None and k != 1:
            raise Exception("Original Hart does not support k best solutions. Try HartK.")
        opt_cost = Hart.BIG_NUM

        first_digit = int(handed_first_digit[1:]) if handed_first_digit else None
        last_digit = int(handed_last_digit[1:]) if handed_last_digit else None

        segment_note_count = len(segment)
        note_list = DNote.note_list(segment)
        if len(segment) == 1:
            abcdf = D.Dactyler.one_note_advise(note_list[0], staff=staff,
                                               first_digit=handed_first_digit,
                                               last_digit=handed_last_digit)
            return [abcdf], [0]

        m = segment_note_count - 1
        fs = numpy.zeros([segment_note_count, 6], dtype=int)
        for n in reversed(range(1, segment_note_count)):
            for s in range(1, 6):
                fs[n, s] = Hart.BIG_NUM

        fsx = numpy.zeros([segment_note_count, 6, 6], dtype=int)
        num_opt = numpy.zeros([segment_note_count, 6], dtype=int)
        xstar = numpy.zeros([segment_note_count, 6, 5], dtype=int)

        # Stage m
        mth_note = note_list[m]
        mth_color = mth_note.color()
        prior_color = mth_note.prior_color()
        mth_interval = mth_note.semitone_delta()
        penultimate_digit_options = range(1, 6)
        if first_digit and segment_note_count == 2:
            penultimate_digit_options = [first_digit]
        for s in penultimate_digit_options:
            if s == 1:
                self.squawk("Stage {0}: color {1}->{2}, delta {3}".format(m, prior_color, mth_color, mth_interval))
            self.squeak("{0:4d}:".format(s))
            for x in range(1, 6):
                if (staff == "upper" and mth_note.is_ascending()) or (staff == "lower" and not mth_note.is_ascending):
                    interval = Interval(prior_color, mth_color, s, x, mth_interval)
                else:
                    interval = Interval(mth_color, prior_color, x, s, mth_interval)
                cost = self._costs[interval]
                if last_digit:
                    # Last digit in fingering sequence is constrained, so we force all paths to
                    # lead to it by making other paths look less attractive. But we retain preference
                    # for arcs with known reasonable (non-"infinite") costs.
                    if x == last_digit:
                        if cost == Hart.BIG_NUM:
                            cost -= 1  # Prefer these arcs over any that lead to the wrong place
                        # Otherwise, prefer plausible arcs (those with non-"infinite" cost) according
                        # to their costs.
                    else:
                        cost = Hart.BIG_NUM  # Paths leading to the wrong last digit are "infinitely" expensive.

                fsx[m, s, x] = cost
                self.squeak("{0:4d}  ".format(fsx[m, s, x]))

            for x in range(1, 6):
                if fsx[m, s, x] < fs[m, s]:
                    fs[m, s] = fsx[m, s, x]

            num_opt[m, s] = 0
            for x in range(1, 6):
                if fsx[m, s, x] == fs[m, s]:
                    xstar[m, s, num_opt[m, s]] = x
                    num_opt[m, s] += 1

            for x in range(num_opt[m, s]):
                self.squeak(str(xstar[m, s, x]))
                if x < num_opt[m, s] - 1:
                    self.squeak(",")
                else:
                    self.squawk("")

        # Stages m-1 through 1
        start_index = 1
        for n in reversed(range(1, m)):
            nth_note = note_list[n]
            nth_color = nth_note.color()
            prior_color = nth_note.prior_color()
            nth_interval = nth_note.semitone_delta()
            for s in range(1, 6):
                if s == 1:
                    self.squawk("Stage {0}: color {1}->{2}, delta {3}".format(n, prior_color, nth_color, nth_interval))
                self.squeak("{0:4d}:".format(s))

                for x in range(1, 6):
                    if (staff == "upper" and nth_note.is_ascending()) or \
                            (staff == "lower" and not nth_note.is_ascending):
                        interval = Interval(prior_color, nth_color, s, x, nth_interval)
                    else:
                        interval = Interval(nth_color, prior_color, x, s, nth_interval)
                    cost = self._costs[interval]
                    # First digit in fingering sequence is constrained, so we force all paths to
                    # start from it by making other paths look less attractive. But we retain preference
                    # for arcs with known reasonable (non-"infinite") costs.
                    if first_digit and n == start_index:
                        if s == first_digit:
                            if cost == Hart.BIG_NUM:
                                cost -= 1  # Prefer these arcs over any that come from the wrong place
                                # Otherwise, prefer plausible arcs (those with non-"infinite" cost)
                                # according to their costs.
                        else:
                            cost = Hart.BIG_NUM  # Paths leading to the wrong last digit are "infinitely" expensive.
                    fsx[n, s, x] = cost + fs[n + 1, x]
                    self.squeak("{0:4d}+{1:d}={2:4d}  ".format(cost, fs[n + 1, x], fsx[n, s, x]))

                for x in range(1, 6):
                    if fsx[n, s, x] < fs[n, s]:
                        fs[n, s] = fsx[n, s, x]

                num_opt[n, s] = 0
                for x in range(1, 6):
                    if fsx[n, s, x] == fs[n, s]:
                        xstar[n, s, num_opt[n, s]] = x
                        num_opt[n, s] += 1

                if num_opt[n, s] == 0:
                    self.squawk("")

                for x in range(num_opt[n, s]):
                    self.squeak(str(xstar[n, s, x]))
                    if x < num_opt[n, s] - 1:
                        self.squeak(",")
                    else:
                        self.squeak("\n")

        fingers = [0]
        for s in range(1, 6):
            fs_cost = fs[1, s]
            if fs_cost < opt_cost:
                opt_cost = fs_cost
                fingers[0] = s
        for n in range(1, m + 1):
            fingers.append(xstar[n, fingers[n - 1], 0])

        self.squawk("The optimal cost is {0}".format(opt_cost))
        self.squawk("Here is an optimal fingering:")
        self.squawk(fingers)

        hand = ">"
        if staff == "lower":
            hand = "<"
        abcdf = hand + "".join(str(f) for f in fingers)
        return [abcdf], [0]


class HartK(Hart):
    def generate_segment_advice(self, segment, staff, offset, handed_first_digit, handed_last_digit, k=None):
        segment_note_count = len(segment)
        note_list = DNote.note_list(segment)
        if len(segment) == 1:
            abcdf = D.Dactyler.one_note_advise(note_list[0], staff=staff,
                                               first_digit=handed_first_digit,
                                               last_digit=handed_last_digit)
            return [abcdf], [0]

        hand = ">"
        if staff == "lower":
            hand = "<"

        g = nx.MultiDiGraph()
        g.add_node(0, midi=None, handed_digit=None)
        prior_slice_node_ids = list()
        prior_slice_node_ids.append(0)
        last_note_in_segment_index = len(segment) - 1
        node_id = 1
        on_last_prefingered_note = False
        for note_in_segment_index in range(segment_note_count):
            d_note = note_list[note_in_segment_index]
            on_first_prefingered_note = False
            slice_node_ids = list()

            if note_in_segment_index == 0 and handed_first_digit:
                on_first_prefingered_note = True

            if note_in_segment_index == last_note_in_segment_index and handed_last_digit:
                on_last_prefingered_note = True

            for digit in range(1, 6):
                handed_digit = hand + str(digit)
                if on_last_prefingered_note and handed_digit != handed_last_digit:
                    continue
                if on_first_prefingered_note and handed_digit != handed_first_digit:
                    continue

                g.add_node(node_id, midi=d_note.midi(), handed_digit=handed_digit)
                slice_node_ids.append(node_id)
                if 0 in prior_slice_node_ids:
                    g.add_edge(0, node_id, weight=1)
                else:
                    for prior_node_id in prior_slice_node_ids:
                        prior_node = g.nodes[prior_node_id]
                        prior_hd = prior_node["handed_digit"]
                        prior_digit = int(prior_hd[1:])

                        if (staff == "upper" and d_note.is_ascending()) or (staff == "lower" and not d_note.is_ascending):
                            interval = Interval(low_color=d_note.prior_color(),
                                                high_color=d_note.color(),
                                                low_finger=prior_digit,
                                                high_finger=digit,
                                                semitone_delta=d_note.semitone_delta())
                        else:
                            interval = Interval(low_color=d_note.color(),
                                                high_color=d_note.prior_color(),
                                                low_finger=digit,
                                                high_finger=prior_digit,
                                                semitone_delta=d_note.semitone_delta())
                        cost = self._costs[interval]

                        g.add_edge(prior_node_id, node_id, weight=cost)
                node_id += 1
            if len(slice_node_ids) > 0:
                prior_slice_node_ids = copy.copy(slice_node_ids)

        g.add_node(node_id, midi=None, handed_digit=None)
        for prior_node_id in prior_slice_node_ids:
            g.add_edge(prior_node_id, node_id, weight=1)

        return D.Dactyler.standard_graph_advise(g=g, target_id=node_id, k=k)
