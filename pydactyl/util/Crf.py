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
from pathlib import Path
from pydactyl.dactyler.Parncutt import TrigramNode, is_black, ImaginaryBlackKeyRuler, PhysicalRuler

VERSION_FEATURES = {
    '0000': {
        'judge': 'none',
        'judge_chords': False,
        'bop': False,
        'eop': False,
        'distance': 'none',
        # 'distance_window': 4,
        'staff': True,
        'black': False,
        'simple_chording': True,
        'leap': False,
        'articulation': False,
        'tempo': False,
        'velocity': False,
        'repeat': False
    },
    '0001': {
        'judge': 'none',
        'judge_chords': False,
        'bop': False,
        'eop': False,
        'distance': 'lattice',
        # 'distance_window': 4,
        'staff': True,
        'black': True,
        'simple_chording': True,
        'leap': False,
        'articulation': True,
        'tempo': False,
        'velocity': False,
        'repeat': False
    },
    '0002': {
        'judge': 'parncutt',
        'judge_chords': False,
        'bop': False,
        'eop': False,
        'distance': 'lattice',
        # 'distance_window': 4,
        'staff': True,
        'black': True,
        'simple_chording': True,
        'leap': False,
        'articulation': True,
        'tempo': False,
        'velocity': False,
        'repeat': False
    },
    '0003': {
        'judge': 'parncutt',
        'judge_chords': False,
        'bop': True,
        'eop': True,
        'distance': 'lattice',
        # 'distance_window': 4,
        'staff': True,
        'black': True,
        'simple_chording': True,
        'leap': False,
        'articulation': True,
        'tempo': False,
        'velocity': False,
        'repeat': False
    },
}

PICKLE_BASE_DIR = '/tmp/pickle/'
MAX_LEAP = 16
CHORD_MS_THRESHOLD = 30
# CLEAN_LIST = {}  # Reuse all pickled results.
# CLEAN_LIST = {'crf': True}
# CLEAN_LIST = {'DCorpus': True}
CLEAN_LIST = {'crf': True, 'DExperiment': True}  # Pickles to discard (and regenerate).
# CLEAN_LIST = {'crf': True, 'DCorpus': True, 'DExperiment': True}  # Pickles to discard (and regenerate).
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


def unpickle_it(obj_type, file_name, use_dill=False):
    pickle_dir = pickle_directory(obj_type)
    pickle_path = pickle_dir + file_name

    path = Path(pickle_path)
    if path.is_file():
        if obj_type in CLEAN_LIST:
            os.remove(pickle_path)
            print("Pickle file {} removed because {} is on the CLEAN_LIST.".format(pickle_path, obj_type))
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


def leap_is_excessive(notes, middle_i):
    left_i = middle_i - 1
    if left_i in notes:
        leap = notes[middle_i].pitch.midi - notes[left_i].pitch.midi
        if abs(leap) > MAX_LEAP:
            return True
    else:
        return True  # That first step is a doozy. Infinite leap.
    return False


def chordings(notes, middle_i):
    middle_offset_ms = notes[middle_i]['second_offset'] * 1000
    min_left_offset_ms = middle_offset_ms - CHORD_MS_THRESHOLD
    max_right_offset_ms = middle_offset_ms + CHORD_MS_THRESHOLD
    left_chord_notes = 0
    for i in range(middle_i - 1, middle_i - 5, -1):
        if i < 0:
            break
        i_offet_ms = notes[i]['second_offset'] * 1000
        if i_offet_ms > min_left_offset_ms:
            left_chord_notes += 1
    right_chord_notes = 0
    for i in range(middle_i + 1, middle_i + 5, 1):
        if i >= len(notes):
            break
        i_offet_ms = notes[i]['second_offset'] * 1000
        if i_offet_ms < max_right_offset_ms:
            right_chord_notes += 1
    return left_chord_notes, right_chord_notes


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
    midi = notes[i]['note'].pitch.midi
    is_black_key = is_black(midi_number=midi)
    return is_black_key


def integral_distance(notes, from_i, to_i, absolute=False):
    if from_i < 0 or to_i >= len(notes):
        return 0
    from_midi = notes[from_i]['note'].pitch.midi
    to_midi = notes[to_i]['note'].pitch.midi
    diff = to_midi - from_midi
    if absolute:
        diff = abs(diff)
    return diff


def lattice_distance(notes, from_i, to_i, absolute=False):
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
    if absolute:
        x_distance = abs(x_distance)
        y_distance = abs(y_distance)
    return x_distance, y_distance
