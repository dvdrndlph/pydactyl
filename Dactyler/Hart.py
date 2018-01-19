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

import numpy
import re
import music21
from Dactyler import Dactyler
from Corpus import Corpus


class Interval:
    def __init__(self, l_color, h_color, l_finger, h_finger, s):
        self.l_color = int(l_color)
        self.h_color = int(h_color)
        self.l_finger = int(l_finger)
        self.h_finger = int(h_finger)
        self.s = int(s)

    def __hash__(self):
        val = self.h_finger + 10*self.l_finger + 100*self.h_color + 1000*self.l_color + 10000*self.s
        return val

    def __eq__(self, other):
        if self.l_color == other.l_color and self.h_color == other.h_color and self.l_finger == other.l_finger and\
                self.h_finger == other.h_finger and self.s == other.s:
            return True
        return False

    def __str__(self):
        my_str = "finger: {0}->{1}, color: {2}->{3}, interval: {4}"
        my_str = my_str.format(str(self.l_finger),
                               str(self.h_finger),
                               str(self.l_color),
                               str(self.h_color),
                               str(self.s))
        return my_str

    def __repr__(self):
        return self.__str__()


class MyNote:
    BLACK = 1
    WHITE = 0

    def __init__(self, m21_note, prior_note=None):
        self.m21_note = m21_note
        self.prior_note = prior_note

    def __str__(self):
        my_str = "MIDI {0}".format(self.m21_note.midi)
        return my_str

    note_class_is_black = {
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

    @staticmethod
    def get_note_list(m21_score):
        prior_note = None
        notes = []
        for n in m21_score[1].getElementsByClass(music21.note.Note):
            new_note = None

            if not prior_note:
                new_note = MyNote(n)
            else:
                new_note = MyNote(n, prior_note=prior_note)

            notes.append(new_note)
            prior_note = new_note
        return notes

    def is_black(self):
        if not self.m21_note:
            return False
        return MyNote.note_class_is_black[self.m21_note.pitchClass]

    def is_white(self):
        if not self.m21_note:
            return False
        return not self.is_black()

    def get_color(self):
        if not self.m21_note:
            return None
        if self.is_black():
            return MyNote.BLACK
        return MyNote.WHITE

    def get_prior_color(self):
        if self.prior_note:
            return self.prior_note.get_color()
        return None

    def get_midi(self):
        if self.m21_note:
            return self.m21_note.midi
        return None

    def get_prior_midi(self):
        if self.prior_note:
            return self.prior_note.get_midi()
        return None

    def is_ascending(self):
        if not self.get_prior_midi() or not self.get_midi():
            return False
        if self.get_midi() > self.get_prior_midi():
            return True
        return False

    def get_semitone_delta(self):
        delta = self.m21_note.midi - self.prior_note.m21_note.midi
        delta = -1 * delta if delta < 0 else delta
        return delta


class Hart(Dactyler.Dactyler):
    BIG_NUM = 99999
    MAX_INTERVAL_SIZE = 12
    COST_FILE = '/Users/dave/tb2/didactyl/dd/data/tables.dat'
    # TEST_CORPUS = "/tmp/%s" % sys.argv[1]
    TEST_CORPUS = "/Users/dave/tb2/didactyl/dd/corpora/beringer/broken_chords.abc"
    LOG_FILE_PATH = '/tmp/didactyl.log'
    Log = open(LOG_FILE_PATH, 'a')

    def _define_costs(self):
        costs = {}
        for l_color in range(2):
            for h_color in range(2):
                for l_finger in range(1, 6):
                    for h_finger in range(1, 6):
                        for s in range(1, self.max_interval_size + 1):
                            # FIXME: What about 0 interval?
                            ic = Interval(l_color, h_color, l_finger, h_finger, s)
                            costs[ic] = Hart.BIG_NUM

        interval_size_finalized = False
        max_interval_size = self.max_interval_size
        max_interval_re = re.compile(r"^Max_Interval:\s+(\d+)")
        heading_re = re.compile(r"^([^_]+)_to_([^_]+)")
        cost_re_str = "^(\d)\s+(\d)"
        cost_re = None
        l_color = None
        h_color = None
        l_finger = None
        h_finger = None

        f = open(self.cost_path, 'r')
        for line in f:
            line = line.rstrip()
            if not interval_size_finalized:
                matches = re.search(max_interval_re, line)
                if matches:
                    max_interval_size = int(matches.group(1))
                interval_size_finalized = True

                for i in range(max_interval_size):
                    cost_re_str += "\s+(\d+)"
                cost_re = re.compile(cost_re_str)
                continue

            matches = re.search(heading_re, line)
            if matches:
                l_color = MyNote.BLACK if matches.group(1) == 'Black' else MyNote.WHITE
                h_color = MyNote.BLACK if matches.group(2) == 'Black' else MyNote.WHITE
                continue

            matches = re.search(cost_re, line)
            if matches:
                l_finger = matches.group(1)
                h_finger = matches.group(2)
                first_group_num = 2
                for s in range(1, max_interval_size + 1):
                    group_num = first_group_num + s
                    cost = matches.group(group_num)
                    interval = Interval(l_color=l_color, h_color=h_color, l_finger=l_finger, h_finger=h_finger, s=s)
                    costs[interval] = int(cost)
        return costs

    def __init__(self, cost_path=None, max_interval_size=MAX_INTERVAL_SIZE):
        super().__init__(hands=Dactyler.Constant.HANDS_RIGHT, chords=False)
        self.cost_path = Hart.COST_FILE
        if cost_path:
            self.cost_path = cost_path
        self.max_interval_size = max_interval_size
        self.costs = self._define_costs();
        self.scores = None

    def load_corpus(self, path=None, query=None, corpus_type=Dactyler.Constant.CORPUS_ABC):
        if corpus_type != Dactyler.Constant.CORPUS_ABC and \
                corpus_type != Dactyler.Constant.CORPUS_ABCD:
            raise Exception("Only ABC is currently supported")

        if not path:
            path = Hart.TEST_CORPUS;
        if query:
            raise Exception("MySQL query not implemented yet")
        corp = Corpus.Corpus(path)
        self.scores = corp.get_score_list()

    def advise(self, offset=0, first_finger=None):
        if offset or first_finger:
            raise Exception("Offset start not implemented yet")

        for score in self.scores:
            opt_cost = Hart.BIG_NUM
            note_list = MyNote.get_note_list(score)
            m = len(note_list) - 1
            fs = numpy.zeros([len(note_list), 6], dtype=int)
            for n in reversed(range(1, len(note_list))):
                for s in range(1, 6):
                    fs[n, s] = Hart.BIG_NUM

            fsx = numpy.zeros([len(note_list), 6, 6], dtype=int)
            num_opt = numpy.zeros([len(note_list), 6], dtype=int)
            xstar = numpy.zeros([len(note_list), 6, 5], dtype=int)

            # Stage m
            mth_note = note_list[m]
            mth_color = mth_note.get_color()
            prior_color = mth_note.get_prior_color()
            mth_interval = mth_note.get_semitone_delta()
            for s in range(1, 6):
                if s == 1:
                    self.squawk("Stage {0}: color {1}->{2}, delta {3}".format(m, prior_color, mth_color, mth_interval))
                self.squeak("{0:4d}:".format(s))
                for x in range(1, 6):
                    interval = None
                    if mth_note.is_ascending():
                        interval = Interval(prior_color, mth_color, s, x, mth_interval)
                    else:
                        interval = Interval(mth_color, prior_color, x, s, mth_interval)
                    print(interval)
                    cost = self.costs[interval]
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
            for n in reversed(range(1, m)):
                nth_note = note_list[n]
                nth_color = nth_note.get_color()
                prior_color = nth_note.get_prior_color()
                nth_interval = nth_note.get_semitone_delta()
                for s in range(1, 6):
                    if s == 1:
                        self.squawk("Stage {0}: color {1}->{2}, delta {3}".format(n, prior_color, nth_color, nth_interval))
                    self.squeak("{0:4d}:".format(s))

                    if nth_interval != 0:
                        for x in range(1, 6):
                            interval = None
                            if nth_note.is_ascending():
                                interval = Interval(prior_color, nth_color, s, x, nth_interval)
                            else:
                                interval = Interval(nth_color, prior_color, x, s, nth_interval)
                            cost = self.costs[interval]
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
                    else:
                        self.squawk("UNCHARTED WATERS")
                        for x in range(1, 6):
                            fsx[n, s, x] = fsx[n + 1, s, x]
                            fs[n, s] = fs[n + 1, s]

                    for x in range(num_opt[n, s]):
                        self.squeak(str(xstar[n, s, x]))
                        if x < num_opt[n, s] - 1:
                            self.squeak(",")
                        else:
                            self.squeak("\n")

            fingers = [0]
            for s in range(1, 6):
                if fs[1, s] < opt_cost:
                    opt_cost = fs[1, s]
                    fingers[0] = s
            for n in range(1, m + 1):
                fingers.append(xstar[n, fingers[n - 1], 0])

            self.squawk("The optimal cost is {0}".format(opt_cost))
            self.squawk("Here is an optimal fingering:")
            self.squawk(fingers)
    
            # print format(opt_cost)
            # print fingers

            title = score[0].title

            print('{ "title" : "' + title + '", "optimal_fingering": "' + str(fingers) + '" }')
