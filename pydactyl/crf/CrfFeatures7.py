__author__ = 'David Randolph'
# Copyright (c) 2020-2023 David A. Randolph.
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
from music21 import note
from pydactyl.dcorpus.DNotesData import DNotesData
from pydactyl.dcorpus.DScore import DScore
import pydactyl.crf.CrfUtil as c

CRF_VERSION = "7"
REVERSE_NOTES = False
MAX_LEAP = 15


def my_note2features(notes_data: DNotesData, i, staff):
    notes = notes_data.notes
    d_score: DScore = notes_data.d_score
    features = dict()

    # features['composer'] = d_score.composer()
    # Period of piece: Baroque, Classical, Romantic, Modern, Contemporary, Other.
    # features['primary_period'] = d_score.periods()[0]
    # features['period_str'] = d_score.period_str()
    # IDEA: Composer year/decade/century of birth.

    # is_human_performance = d_score.via_human_performance()
    # is_midi = d_score.via_midi()
    # features['is_human'] = is_human_performance
    # features['is_midi'] = is_midi

    # IDEA: Inferred key of note sequence, staff, note windows.

    features['staff'] = staff

    # Chord features. Approximate with 30 ms offset deltas a la Nakamura.
    # These features are determined entirely from note strikes.
    chording_cat = c.chording_categories(notes=notes, middle_i=i, staff=staff)
    # if chording_cat not in ('upper00', 'lower00'):
    #     print("Chord!")
    features['chord_cat'] = chording_cat
    chord_border = c.chord_border(chording_cat)
    features['chord_border'] = chord_border

    concurrent_cat = notes_data.concurrent_count_feature_str(i)
    features['concurrent_cat'] = concurrent_cat

    features['BOP'] = "0"
    if i == 0:
        features['BOP'] = "1"
    features['EOP'] = "0"
    if i >= len(notes) - 1:
        features['EOP'] = "1"

    x_d = dict()
    y_d = dict()
    x_d[-4], y_d[-4] = c.lattice_distance(notes=notes, from_i=i-4, to_i=i, max_leap=MAX_LEAP)
    x_d[-3], y_d[-3] = c.lattice_distance(notes=notes, from_i=i-3, to_i=i, max_leap=MAX_LEAP)
    x_d[-2], y_d[-2] = c.lattice_distance(notes=notes, from_i=i-2, to_i=i, max_leap=MAX_LEAP)
    x_d[-1], y_d[-1] = c.lattice_distance(notes=notes, from_i=i-1, to_i=i, max_leap=MAX_LEAP)
    x_d[+1], y_d[+1] = c.lattice_distance(notes=notes, from_i=i, to_i=i+1, max_leap=MAX_LEAP)
    x_d[+2], y_d[+2] = c.lattice_distance(notes=notes, from_i=i, to_i=i+2, max_leap=MAX_LEAP)
    x_d[+3], y_d[+3] = c.lattice_distance(notes=notes, from_i=i, to_i=i+3, max_leap=MAX_LEAP)
    x_d[+4], y_d[+4] = c.lattice_distance(notes=notes, from_i=i, to_i=i+4, max_leap=MAX_LEAP)
    # x_d['-4-3'], y_d['-4-3'] = c.lattice_distance(notes=notes, from_i=i-4, to_i=i-3, max_leap=MAX_LEAP)
    # x_d['-3-2'], y_d['-3-2'] = c.lattice_distance(notes=notes, from_i=i-3, to_i=i-2, max_leap=MAX_LEAP)
    # x_d['-2-1'], y_d['-2-1'] = c.lattice_distance(notes=notes, from_i=i-2, to_i=i-1, max_leap=MAX_LEAP)
    # x_d['-10'], y_d['-10'] = c.lattice_distance(notes=notes, from_i=i-1, to_i=i, max_leap=MAX_LEAP)
    # x_d['0+1'], y_d['0+1'] = c.lattice_distance(notes=notes, from_i=i, to_i=i+1, max_leap=MAX_LEAP)
    # x_d['+1+2'], y_d['+1+2'] = c.lattice_distance(notes=notes, from_i=i+1, to_i=i+2, max_leap=MAX_LEAP)
    # x_d['+2+3'], y_d['+2+3'] = c.lattice_distance(notes=notes, from_i=i+2, to_i=i+3, max_leap=MAX_LEAP)
    # x_d['+3+4'], y_d['+3+4'] = c.lattice_distance(notes=notes, from_i=i+3, to_i=i+4, max_leap=MAX_LEAP)

    features['x_distance:-3'] = x_d[-3]
    features['x_distance:-2'] = x_d[-2]
    features['x_distance:-1'] = x_d[-1]
    features['x_distance:+1'] = x_d[+1]
    features['x_distance:+2'] = x_d[+2]
    features['x_distance:+3'] = x_d[+3]

    features['y_distance:-3'] = y_d[-3]
    features['y_distance:-2'] = y_d[-2]
    features['y_distance:-1'] = y_d[-1]
    features['y_distance:+1'] = y_d[+1]
    features['y_distance:+2'] = y_d[+2]
    features['y_distance:+3'] = y_d[+3]

    features['dxgram_-1|+1'] = "{}|{}".format(x_d[-1], x_d[1])
    # features['dygram_-1|+1'] = "{}|{}".format(y_d[-1], y_d[1])
    features['dxgram_-2|-1|+1|+2'] = "{}|{}|{}|{}".format(x_d[-2], x_d[-1], x_d[1], x_d[2])
    features['dxgram_-3|-2|-1|+1|+2|+3'] = "{}|{}|{}|{}|{}|{}".format(x_d[-3], x_d[-2], x_d[-1], x_d[1], x_d[2], x_d[3])

    black = dict()
    black[-1] = str(c.black_key(notes, i-1))
    black[0] = str(c.black_key(notes, i))
    black[1] = str(c.black_key(notes, i+1))

    features['black:-1'] = black[-1]
    features['black'] = black[0]
    features['black:+1'] = black[1]
    features['black3gram'] = "{}|{}|{}".format(black[-1], black[0], black[1])

    features['returning'] = "0"
    if x_d[-2] == 0:
        features['returning'] = "1"  # .5486
    features['will_return'] = "0"
    if x_d[+2] == 0:
        features['will_return'] = "1"  # .5562

    features['ascending'] = "0"
    if x_d[-1] < 0 and x_d[+1] > 0:
        features['ascending'] = "1"
    features['descending'] = "0"
    if x_d[-1] > 0 and x_d[+1] < 0:
        features['descending'] = "1"

    # The n-grams might be at a disadvantage against distances, as they provide fewer opportunities to
    # learn from isomorphic situations.
    # pit = dict()
    # pit[-3], pit[-2], pit[-1], pit[0], pit[1], pit[2], pit[3] = c.get_pit_strings(notes, i, range=3)
    # features['pit'] = pit[0]
    # features['pit_-1|0'] = pit[-1] + '|' + pit[0]
    # features['pit_0|+1'] = pit[0] + '|' + pit[1]
    # features['pit_-1|0|+1'] = pit[-1] + '|' + pit[0] + '|' + pit[1]
    # features['pit_-2|-1|0|+1|+2'] = "{}|{}|{}|{}|{}".format(pit[-2], pit[-1], pit[0], pit[1], pit[2])

    # Impact of large leaps? Costs max out, no? Maybe not.
    features['leap'] = "0"
    if c.leap_is_excessive(notes, i, max_leap=MAX_LEAP):
        features['leap'] = "1"

    oon = notes[i]
    m21_note: note.Note = oon['note']
    on_velocity = m21_note.volume.velocity
    if on_velocity is None:
        on_velocity = 64
    features['velocity'] = on_velocity

    tempi = c.tempo_features(notes=notes, middle_i=i)
    for k in tempi:
        features[k] = tempi[k]

    arts = c.articulation_features(notes=notes, middle_i=i)
    for k in arts:
        features[k] = arts[k]

    reps_before, reps_after = c.repeat_features(notes=notes, middle_i=i)
    features['odd_repeats_before'] = str(bool(reps_before % 2))
    # features['odd_repeats_after'] = str(bool(reps_after % 2))

    return features
