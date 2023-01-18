__author__ = 'David Randolph'
# Copyright (c) 2021, 2022 David A. Randolph.
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
import os
import pickle
import dill
import re
from pathlib import Path
from music21 import note
from pydactyl.dactyler.Parncutt import TrigramNode, is_black, ImaginaryBlackKeyRuler
from pydactyl.eval.DExperimentOpts import DExperimentOpts

PICKLE_BASE_DIR = '/tmp/pickle/'
MAX_LEAP = 16
CHORD_MS_THRESHOLD = 30
CHORD_TAG_LIST = {
    'str': True,
    'sma': True,
    'impw': True,
    'nrfw': True
}
SIX_SIX_RULER = ImaginaryBlackKeyRuler()


def pickle_directory(obj_type):
    my_dir = PICKLE_BASE_DIR + obj_type + '/'
    return my_dir


def pickle_it(obj, obj_type, file_name, use_dill=False):
    pickle_dir = pickle_directory(obj_type)
    path = Path(pickle_dir)
    if not path.is_dir():
        os.makedirs(pickle_dir)
    pickle_path = pickle_dir + file_name
    print("Pickling {} to path {}.".format(obj_type, pickle_path))
    pickle_fh = open(pickle_path, 'wb')
    if use_dill:
        dill.dump(obj, pickle_fh, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        pickle.dump(obj, pickle_fh, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_fh.close()


def get_experiment_name(opts: DExperimentOpts):
    corpora_str = "-".join(opts.corpus_names)
    name = corpora_str + '__' + opts.test_method + '__' + opts.model_version
    return name


def unpickle_it(obj_type, clean_list, file_name=None, opts: DExperimentOpts = None, use_dill=False):
    pickle_dir = pickle_directory(obj_type)
    if file_name is None:
        file_name = get_experiment_name(opts=opts)
    pickle_path = pickle_dir + file_name

    path = Path(pickle_path)
    if path.is_file():
        if obj_type in clean_list:
            os.remove(pickle_path)
            print("Pickle file {} removed because {} is on the clean_list.".format(pickle_path, obj_type))
            return None
        pickle_fh = open(pickle_path, 'rb')
        if use_dill:
            unpickled_obj = dill.load(pickle_fh)
        else:
            unpickled_obj = pickle.load(pickle_fh)
        pickle_fh.close()
        print("Unpickled {} from path {}.".format(obj_type, pickle_path))
        return unpickled_obj
    return None


def has_preset_evaluation_defined(corpus_name):
    if corpus_name in ('pig_seg', 'pig_indy', 'pig', 'complete_layer_one'):
        return True
    return False


def is_in_test_set(title: str, corpus_name='pig_indy'):
    if corpus_name in ('pig_seg', 'pig_indy', 'pig'):
        example, annotator_id = title.split('-')
        example_int = int(example)
        if example_int <= 30:
            return True
    elif corpus_name == 'complete_layer_one':
        if title.startswith('Sonatina 6.1'):
            return True
    else:
        raise Exception("Not implemented yet.")
    return False


def get_pit_strings(notes, i, range=4):
    target_pit_count = range * 2 + 1
    index = i
    pit_index = index - range
    pit_list = []
    while pit_index < 0:
        pit_list.append("-1")
        pit_index += 1
    note_count = len(notes)
    pit_count = len(pit_list)
    while pit_index < note_count and pit_count < target_pit_count:
        pit_list.append(str(notes[pit_index]['note'].pitch.midi))
        pit_index += 1
        pit_count += 1
    while pit_count < target_pit_count:
        pit_list.append("-1")
        pit_count += 1
    return pit_list


def get_trigram_node(notes, annotations, i):
    midi_1 = None
    handed_digit_1 = '-'
    if i > 0:
        midi_1 = notes[i-1]['note'].pitch.midi
        handed_digit_1 = annotations[i-1]
    midi_2 = notes[i]['note'].pitch.midi
    handed_digit_2 = annotations[i]

    midi_3 = None
    handed_digit_3 = '-'
    if i < len(notes) - 1:
        midi_3 = notes[i+1]['note'].pitch.midi
        handed_digit_3 = annotations[i+1]
    trigram_node = TrigramNode(midi_1=midi_1, handed_digit_1=handed_digit_1,
                               midi_2=midi_2, handed_digit_2=handed_digit_2,
                               midi_3=midi_3, handed_digit_3=handed_digit_3)
    return trigram_node


def get_trigram_nodes(notes, i, staff):
    trigram_nodes = []
    hand = '>'
    if staff == "lower":
        hand = "<"
    for left_digit in range(1, 6, 1):
        if i < 1:
            midi_1 = None
            handed_digit_1 = '-'
        else:
            midi_1 = notes[i-1]['note'].pitch.midi
            handed_digit_1 = hand + str(left_digit)
        for middle_digit in range(1, 6, 1):
            midi_2 = notes[i]['note'].pitch.midi
            handed_digit_2 = hand + str(middle_digit)
            for right_digit in range(1, 6, 1):
                if i + 1 not in notes:
                    midi_3 = None
                    handed_digit_3 = '-'
                else:
                    midi_3 = notes[i+1]['note'].pitch.midi
                    handed_digit_3 = hand + str(right_digit)
                if handed_digit_1 != handed_digit_2 and handed_digit_2 != handed_digit_3:
                    trigram_node = TrigramNode(midi_1=midi_1, handed_digit_1=handed_digit_1,
                                               midi_2=midi_2, handed_digit_2=handed_digit_2,
                                               midi_3=midi_3, handed_digit_3=handed_digit_3)
                    trigram_nodes.append(trigram_node)
    return trigram_nodes


def judgments(judge, notes, middle_i, staff):
    trigram_nodes = get_trigram_nodes(notes=notes, i=middle_i, staff=staff)
    nodes_by_cost = {}
    lowest_cost = 999999
    for node in trigram_nodes:
        cost, _ = judge.trigram_node_cost(trigram_node=node)
        if cost < lowest_cost:
            lowest_cost = cost
        if cost not in nodes_by_cost:
            nodes_by_cost[cost] = []
        nodes_by_cost[cost].append(node)
    bad_fingers = dict()
    bad_fingers['-1'] = dict.fromkeys(['1', '2', '3', '4', '5'], 1)
    bad_fingers['0'] = dict.fromkeys(['1', '2', '3', '4', '5'], 1)
    bad_fingers['+1'] = dict.fromkeys(['1', '2', '3', '4', '5'], 1)
    for good_trigram in nodes_by_cost[lowest_cost]:
        if good_trigram.midi_1 is not None:
            digit_1 = good_trigram.handed_digit_1[1]
            bad_fingers['-1'][digit_1] = 0
        digit_2 = good_trigram.handed_digit_2[1]
        bad_fingers['0'][digit_2] = 0
        if good_trigram.midi_3 is not None:
            digit_3 = good_trigram.handed_digit_3[1]
            bad_fingers['+1'][digit_3] = 0
    return bad_fingers


def leap_is_excessive(notes, middle_i, max_leap=MAX_LEAP):
    left_i = middle_i - 1
    if left_i in notes:
        leap = notes[middle_i].pitch.midi - notes[left_i].pitch.midi
        if abs(leap) > max_leap:
            return True
    else:
        return True  # That first step is a doozy. Infinite leap.
    return False


def chording_notes(notes, middle_i):
    middle_offset_ms = notes[middle_i]['second_offset'] * 1000
    min_left_offset_ms = middle_offset_ms - CHORD_MS_THRESHOLD
    max_right_offset_ms = middle_offset_ms + CHORD_MS_THRESHOLD
    left_chord_notes = list()
    for i in range(middle_i - 1, middle_i - 5, -1):
        if i < 0:
            break
        i_offet_ms = notes[i]['second_offset'] * 1000
        if i_offet_ms > min_left_offset_ms:
            left_chord_notes.append(notes[i])
    right_chord_notes = list()
    for i in range(middle_i + 1, middle_i + 5, 1):
        if i >= len(notes):
            break
        i_offet_ms = notes[i]['second_offset'] * 1000
        if i_offet_ms < max_right_offset_ms:
            right_chord_notes.append(notes[i])
    # if left_chord_notes or right_chord_notes:
        # print("We see chords at position {}.".format(middle_i))
    return left_chord_notes, right_chord_notes


def chordings(notes, middle_i):
    left_chord_notes, right_chord_notes = chording_notes(notes, middle_i)
    left_chord_note_count = len(left_chord_notes)
    right_chord_note_count = len(right_chord_notes)
    return left_chord_note_count, right_chord_note_count


def chordings_by_pitch(notes, middle_i):
    left_notes, right_notes = chording_notes(notes=notes, middle_i=middle_i)
    chord_notes = left_notes + right_notes
    lower_note_count = 0
    upper_note_count = 0
    middle_pit = notes[middle_i]['note'].pitch.midi
    for knot in chord_notes:
        knot_pit = knot['note'].pitch.midi
        if knot_pit < middle_pit:
            lower_note_count += 1
        elif knot_pit > middle_pit:
            upper_note_count += 1
    return lower_note_count, upper_note_count


def chording_categories(notes, middle_i, staff=None):
    lower_note_count, upper_note_count = chordings_by_pitch(notes, middle_i)
    cat_str = "{}{}{}".format(staff, lower_note_count, upper_note_count)
    return cat_str


def chord_border(chord_category):
    mats = re.search('.*(\d)(\d)$', chord_category)
    if mats.group(0) == '0' and mats.group(1) != 0:
        return 'left'
    if mats.group(1) == '0' and mats.group(0) != 0:
        return 'right'
    return 'none'


def chording_category_ngrams(notes, middle_i, pre_count=1, post_count=1, staff=None):
    start_i = middle_i + pre_count
    stop_i = middle_i + post_count + 1
    cat_str = ''
    if staff is not None:
        cat_str = staff + '_'
    for i in range(start_i, stop_i):
        lower_note_count, upper_note_count = chordings_by_pitch(notes, middle_i)
        cat_str += "{}{}".format(lower_note_count, upper_note_count)
    return cat_str


def complex_chording(notes, annotations, middle_i):
    lower_note_count, higher_note_count = chordings(notes=notes, middle_i=middle_i)
    prior_left_digit = None
    prior_right_digit = None
    penalty = 0
    for i in range(middle_i - lower_note_count, middle_i + higher_note_count + 1, 1):
        hand = annotations[i][0]
        digit = int(annotations[i][1])
        if hand == ">":
            if prior_right_digit is not None:
                if prior_right_digit >= digit:
                    penalty += 1
            else:
                prior_right_digit = digit
        else:
            if prior_left_digit is not None:
                if prior_left_digit <= digit:
                    penalty += 1
            else:
                prior_left_digit = digit
    return penalty


def tempo_features(notes, middle_i):
    """
    Calculate notes per second left of middle_i, right of middle_i, and throughout the
    9-note window.
    """
    middle_offset_s = notes[middle_i]['second_offset']
    left_offset_s = 0
    left_note_count = 0
    for i in range(middle_i, middle_i - 5, -1):
        if i < 0:
            break
        left_offset_s = notes[i]['second_offset']
        left_note_count += 1
    left_seconds = middle_offset_s - left_offset_s
    right_offset_s = 0
    right_note_count = 0
    for i in range(middle_i, middle_i + 5, 1):
        if i >= len(notes):
            break
        right_offset_s = notes[i]['second_offset']
        right_note_count += 1
    right_seconds = right_offset_s - middle_offset_s
    window_seconds = left_seconds + right_seconds
    note_count = left_note_count + right_note_count - 1
    if window_seconds == 0:
        window_nps = 0
    else:
        window_nps = note_count / window_seconds

    if left_seconds == 0:
        left_nps = 0
    else:
        left_nps = left_note_count / left_seconds

    if right_seconds == 0:
        right_nps = 0
    else:
        right_nps = right_note_count / right_seconds

    result = {
        'window_nps': window_nps,
        'left_nps': left_nps,
        'right_nps': right_nps
    }
    return result


def articulation_features(notes, middle_i):
    """
    From a segregated fingering standpoint at least, staccato is a purely melodic phenomenon.
    We are looking for a feature to explain why finger order does not align with note order
    (where pivoting occurs without the thumb). When chords are sounding, the hand is anchored
    and is not free to perform such feats. Therefore, we look for a melodic window around the
    note and calculate measures of separation between notes within the window.
    :param notes: List of fingered ordered offset notes.
    :param middle_i: Index of note being evaluated.
    :return: Dictionary of separation measures with following keys:
        staccato_count: Number of notes in the 9-note window surrounding that are followed by
                        silence at least half as long as the note itself.
        normalized_silence: The sum of silence between notes surrounding the note at middle_i
                            divided by the total duration of the 9-note window, measured
                            from the leftmost note onset to the rightmost onset.
        separated_count: The number of notes surrounding the middle_i note that are separated
                         by at least 60ms.
    """
    result = {
        'staccato_count': 0,
        'normalized_silence': 0.0,
        'separated_count': 0
    }

    left_consonant_count, right_consonant_count = chordings(notes, middle_i=middle_i)
    if left_consonant_count or right_consonant_count:
        return result

    total_silence = 0.0
    left_off_time = None
    left_dur = None
    window_start_time = None
    window_end_time = None
    for i in range(middle_i - 4, middle_i + 5, 1):
        if i < 0:
            continue
        if i >= len(notes):
            break
        on_time = notes[i]['second_offset']
        if window_start_time is None:
            window_start_time = on_time
        window_end_time = on_time
        dur = notes[i]['second_duration']
        off_time = on_time + dur
        if left_off_time is not None:
            silence = on_time - left_off_time
            if silence * 1000 > 2 * CHORD_MS_THRESHOLD:
                result['separated_count'] += 1
            if silence >= left_dur / 2:
                result['staccato_count'] += 1
            if silence > 0:
                total_silence += silence
        left_off_time = off_time
        left_dur = dur
    window_dur = window_end_time - window_start_time
    if window_dur > 0:
        result['normalized_silence'] = total_silence / window_dur
    return result


def repeat_features(notes, middle_i):
    """
    Count notes prior to and after the note at middle_i with same pitch as the note at middle_i.
    :param notes: List of fingered ordered offset notes.
    :param middle_i: Index of note being evaluated.
    :return: Tuple of (repeats_before, repeats_after).
    """
    middle_midi = notes[middle_i]['note'].pitch.midi
    repeat_before = 0
    i = middle_i - 1
    while i >= 0:
        previous_midi = notes[i]['note'].pitch.midi
        if previous_midi == middle_midi:
            repeat_before += 1
        else:
            break
        i -= 1

    repeat_after = 0
    i = middle_i + 1
    while i < len(notes):
        next_midi = notes[i]['note'].pitch.midi
        if next_midi == middle_midi:
            repeat_after += 1
        else:
            break
        i += 1

    return repeat_before, repeat_after


def black_key(notes, i):
    """
    Return True if the key sounding the note at index i is black.
    Return False otherwise.
    """
    if i < 0:
        return False
    if i >= len(notes):
        return False
    midi = notes[i]['note'].pitch.midi
    is_black_key = is_black(midi_number=midi)
    return is_black_key


def integral_distance(notes, from_i, to_i, absolute=False, max_leap: int = None):
    if from_i < 0 or to_i >= len(notes):
        return 0
    from_midi = notes[from_i]['note'].pitch.midi
    to_midi = notes[to_i]['note'].pitch.midi
    diff = to_midi - from_midi
    if max_leap is not None and abs(diff) > max_leap:
        multiplier = 1 if diff > 0 else -1
        diff = max_leap * multiplier
    if absolute:
        diff = abs(diff)
    return diff


def lattice_distance(notes, from_i, to_i, absolute=False, max_leap: int = None):
    if from_i < 0 or to_i >= len(notes):
        return 0, 0
    from_midi = notes[from_i]['note'].pitch.midi
    to_midi = notes[to_i]['note'].pitch.midi
    x_distance = SIX_SIX_RULER.distance(from_midi=from_midi, to_midi=to_midi)

    from_is_black = is_black(midi_number=from_midi)
    to_is_black = is_black(midi_number=to_midi)
    if from_is_black == to_is_black:
        y_distance = 0
    elif to_is_black:
        y_distance = 1
    else:
        y_distance = -1
    abs_x_distance = abs(x_distance)
    abs_y_distance = abs(y_distance)
    if max_leap is not None and abs_x_distance > max_leap:
        multiplier = 1 if x_distance > 0 else -1
        x_distance = max_leap * multiplier
        abs_x_distance = abs(x_distance)
    if absolute:
        return abs_x_distance, abs_y_distance
    return x_distance, y_distance


def get_settings():
    settings = {
        'judge': 'Parncutt',
        'judge_chords': False,
        'bop': False,
        'eop': False,
        'distance': 'lattice',
        'staff': True,
        'black': True,
        'simple_chording': True,
        'leap': False,
        'articulation': False,
        'tempo': False,
        'velocity': False,
        'repeat': False
    }
    return settings


def note2attrs(notes, i, staff):
    settings = get_settings()
    attrs = {}

    attrs['note'] = notes[i]

    if settings['distance'] == 'integral':
        attrs['distance:-4'] = integral_distance(notes=notes, from_i=i-4, to_i=i)
        attrs['distance:-3'] = integral_distance(notes=notes, from_i=i-3, to_i=i)
        attrs['distance:-2'] = integral_distance(notes=notes, from_i=i-2, to_i=i)
        attrs['distance:-1'] = integral_distance(notes=notes, from_i=i-1, to_i=i)
        attrs['distance:+1'] = integral_distance(notes=notes, from_i=i, to_i=i+1)
        attrs['distance:+2'] = integral_distance(notes=notes, from_i=i, to_i=i+2)
        attrs['distance:+3'] = integral_distance(notes=notes, from_i=i, to_i=i+3)
        attrs['distance:+4'] = integral_distance(notes=notes, from_i=i, to_i=i+4)
    elif settings['distance'] == 'lattice':
        attrs['x_distance:-4'], attrs['y_distance:-4'] = lattice_distance(notes=notes, from_i=i-4, to_i=i)
        attrs['x_distance:-3'], attrs['y_distance:-3'] = lattice_distance(notes=notes, from_i=i-3, to_i=i)
        attrs['x_distance:-2'], attrs['y_distance:-2'] = lattice_distance(notes=notes, from_i=i-2, to_i=i)
        attrs['x_distance:-1'], attrs['y_distance:-1'] = lattice_distance(notes=notes, from_i=i-1, to_i=i)
        attrs['x_distance:+1'], attrs['y_distance:+1'] = lattice_distance(notes=notes, from_i=i, to_i=i+1)
        attrs['x_distance:+2'], attrs['y_distance:+2'] = lattice_distance(notes=notes, from_i=i, to_i=i+2)
        attrs['x_distance:+3'], attrs['y_distance:+3'] = lattice_distance(notes=notes, from_i=i, to_i=i+3)
        attrs['x_distance:+4'], attrs['y_distance:+4'] = lattice_distance(notes=notes, from_i=i, to_i=i+4)

    if settings['simple_chording']:
        # Chord features. Approximate with 30 ms offset deltas a la Nakamura.
        left_chord_notes, right_chord_notes = chordings(notes=notes, middle_i=i)
        attrs['left_chord'] = left_chord_notes
        attrs['right_chord'] = right_chord_notes

    if settings['staff']:
        attrs['staff'] = 0
        if staff == "upper":
            attrs['staff'] = 1
            # @100: [0.54495717 0.81059147 0.81998371 0.68739401 0.73993751]
            # @1:   [0.54408935 0.80563961 0.82079826 0.6941775  0.73534277]

    if settings['black']:
        attrs['black_key']: black_key(notes, i)

    if settings['leap']:
        # Impact of large leaps? Costs max out, no? Maybe not.
        attrs['leap'] = 0
        if leap_is_excessive(notes, i):
            attrs['leap'] = 1

    if settings['velocity']:
        oon = notes[i]
        m21_note: note.Note = oon['note']
        on_velocity = m21_note.volume.velocity
        if on_velocity is None:
            on_velocity = 64
        attrs['velocity'] = on_velocity

    if settings['tempo']:
        tempi = tempo_features(notes=notes, middle_i=i)
        for k in tempi:
            attrs[k] = tempi[k]

    if settings['articulation']:
        arts = articulation_features(notes=notes, middle_i=i)
        for k in arts:
            attrs[k] = arts[k]

    if settings['repeat']:
        reps_before, reps_after = repeat_features(notes=notes, middle_i=i)
        attrs['repeats_before'] = reps_before
        attrs['repeats_after'] = reps_after

    return attrs


def note2features(notes, i, staff, categorical=False):
    settings = get_settings()
    features = {}

    if settings['bop']:
        features['BOP'] = "0"
        if i == 0:
            features['BOP'] = "1"
    if settings['eop']:
        features['EOP'] = "0"
        if i >= len(notes) - 1:
            features['EOP'] = "1"

    # if settings['distance'] != 'none':
    #     for index_offset in range(1, settings['distance_window'] + 1):
    #         left_i = i - index_offset
    #         right_i = i + index_offset
    #         if settings['distance'] == 'integral':
    #             left_tag = "distance:{}".format(left_i)
    #             right_tag = "distance:+{}".format(right_i)
    #             features[left_tag] = integral_distance(notes=notes, from_i=left_i, to_i=i)
    #             features[right_tag] = integral_distance(notes=notes, from_i=i, to_i=right_i)
    #         elif settings['distance'] == 'lattice':
    #             left_x_tag = "x_distance:{}".format(left_i)
    #             right_x_tag = "x_distance:+{}".format(right_i)
    #             left_y_tag = "y_distance:{}".format(left_i)
    #             right_y_tag = "y_distance:+{}".format(right_i)
    #             features[left_x_tag], features[left_y_tag] = lattice_distance(notes=notes, from_i=left_i, to_i=i)
    #             features[right_x_tag], features[right_y_tag] = lattice_distance(notes=notes, from_i=i, to_i=right_i)

    if settings['distance'] == 'integral':
        features['distance:-4'] = integral_distance(notes=notes, from_i=i-4, to_i=i)
        features['distance:-3'] = integral_distance(notes=notes, from_i=i-3, to_i=i)
        features['distance:-2'] = integral_distance(notes=notes, from_i=i-2, to_i=i)
        features['distance:-1'] = integral_distance(notes=notes, from_i=i-1, to_i=i)
        features['distance:+1'] = integral_distance(notes=notes, from_i=i, to_i=i+1)
        features['distance:+2'] = integral_distance(notes=notes, from_i=i, to_i=i+2)
        features['distance:+3'] = integral_distance(notes=notes, from_i=i, to_i=i+3)
        features['distance:+4'] = integral_distance(notes=notes, from_i=i, to_i=i+4)
    elif settings['distance'] == 'lattice':
        features['x_distance:-4'], features['y_distance:-4'] = lattice_distance(notes=notes, from_i=i-4, to_i=i)
        features['x_distance:-3'], features['y_distance:-3'] = lattice_distance(notes=notes, from_i=i-3, to_i=i)
        features['x_distance:-2'], features['y_distance:-2'] = lattice_distance(notes=notes, from_i=i-2, to_i=i)
        features['x_distance:-1'], features['y_distance:-1'] = lattice_distance(notes=notes, from_i=i-1, to_i=i)
        features['x_distance:+1'], features['y_distance:+1'] = lattice_distance(notes=notes, from_i=i, to_i=i+1)
        features['x_distance:+2'], features['y_distance:+2'] = lattice_distance(notes=notes, from_i=i, to_i=i+2)
        features['x_distance:+3'], features['y_distance:+3'] = lattice_distance(notes=notes, from_i=i, to_i=i+3)
        features['x_distance:+4'], features['y_distance:+4'] = lattice_distance(notes=notes, from_i=i, to_i=i+4)

    if settings['simple_chording']:
        # Chord features. Approximate with 30 ms offset deltas a la Nakamura.
        left_chord_notes, right_chord_notes = chordings(notes=notes, middle_i=i)
        features['left_chord'] = left_chord_notes
        features['right_chord'] = right_chord_notes

    if settings['staff']:
        features['staff'] = 0
        if staff == "upper":
            features['staff'] = 1
            # @100: [0.54495717 0.81059147 0.81998371 0.68739401 0.73993751]
            # @1:   [0.54408935 0.80563961 0.82079826 0.6941775  0.73534277]

    if settings['black']:
        features['black_key']: black_key(notes, i)

    # if settings['complex_chording']:
        # features['complex_chord'] = complex_chording(notes=notes, annotations=annotations, middle_i=i)

    if settings['leap']:
        # Impact of large leaps? Costs max out, no? Maybe not.
        features['leap'] = 0
        if leap_is_excessive(notes, i):
            features['leap'] = 1

    if settings['velocity']:
        oon = notes[i]
        m21_note: note.Note = oon['note']
        on_velocity = m21_note.volume.velocity
        if on_velocity is None:
            on_velocity = 64
        features['velocity'] = on_velocity

    if settings['tempo']:
        tempi = tempo_features(notes=notes, middle_i=i)
        for k in tempi:
            features[k] = tempi[k]

    if settings['articulation']:
        arts = articulation_features(notes=notes, middle_i=i)
        for k in arts:
            features[k] = arts[k]

    if settings['repeat']:
        reps_before, reps_after = repeat_features(notes=notes, middle_i=i)
        features['repeats_before'] = reps_before
        features['repeats_after'] = reps_after

    # if settings['judge'] != 'none':
    #     bad_fingers = judgments(judge=settings['judge'], notes=notes, middle_i=i, staff=staff)
    #     for position in bad_fingers:
    #         for digit in bad_fingers[position]:
    #             k = "judge_{}:{}".format(digit, position)
    #             features[k] = bad_fingers[position][digit]
    # FIXME: Lattice distance in Parncutt rules? Approximated by Jacobs.
    #        Mitigated by Balliauw (which just makes the x-distance more
    #        accurate between same-colored keys).

    if categorical:
        for feature in features:
            features[feature] = str(features[feature])
    return features


def nondefault_hand_count(hsd_seq, staff="upper"):
    nondefault_hand = '<'
    if staff == 'lower':
        nondefault_hand = '>'
    bad_hand_cnt = 0
    for fingering in hsd_seq:
        if fingering[0] == nondefault_hand:
            bad_hand_cnt += 1
    return bad_hand_cnt


def has_wildcard(hsd_seq):
    for fingering in hsd_seq:
        if fingering[0] == 'x':
            return True
    return False
