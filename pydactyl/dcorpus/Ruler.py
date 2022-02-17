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
    if midi_left < midi < midi_right:
        return True
    if midi_right < midi < midi_left:
        return True
    return False


class Ruler(ABC):
    def distance(self, from_midi, to_midi):
        """
        Estimate the distance between two piano keys identified by MIDI code.
        The original Parncutt paper simply uses semitone differences.
        :param from_midi: The starting piano key.
        :param to_midi: The ending piano key.
        :return: The distance between the two keys.
        """
        return to_midi - from_midi


class CacheRuler(Ruler):
    def __init__(self, preload=False):
        self._preload_cache = preload
        self._cache = {}
        self._distance_method = None
        if preload:
            self.cache_all()
            self._distance_method = self.cached_distance
        else:
            self._distance_method = self.dynamic_distance

    def cache_all(self):
        for from_midi in range(21, 109):
            for to_midi in range(21, 109):
                self._cache[(from_midi, to_midi)] = self.calculated_distance(from_midi, to_midi)

    def add_distance_to_cache(self, from_midi, to_midi, d):
        self._cache[(from_midi, to_midi)] = d

    def cached_distance(self, from_midi, to_midi):
        return self._cache[(from_midi, to_midi)]

    def dynamic_distance(self, from_midi, to_midi):
        if (from_midi, to_midi) in self._cache:
            return self._cache[(from_midi, to_midi)]
        d = self.calculated_distance(from_midi, to_midi)
        self.add_distance_to_cache(from_midi, to_midi, d)
        return self.cached_distance(from_midi, to_midi)

    def distance(self, from_midi, to_midi):
        return self._distance_method(from_midi, to_midi)

    @abstractmethod
    def calculated_distance(self, from_midi, to_midi):
        return 0.0


class PhysicalRuler(Ruler):
    def __init__(self):
        self._key_positions = PhysicalRuler.horizontal_key_positions()
        self._bounds_for_semitone_interval = None
        self.set_bounds_for_semitone_intervals()

    def distance(self, from_midi, to_midi):
        from_pos = self._key_positions[from_midi]
        to_pos = self._key_positions[to_midi]
        multiplier = 1
        dist = to_pos - from_pos
        if to_midi < from_midi:
            multiplier = -1
            dist = from_pos - to_pos
        for i in range(len(self._bounds_for_semitone_interval) - 1):
            if self._bounds_for_semitone_interval[i] <= dist <= self._bounds_for_semitone_interval[i+1]:
                return multiplier * i
        raise Exception("Distance between {0} and {1} could not be calculated".format(from_midi, to_midi))

    def set_bounds_for_semitone_intervals(self):
        avg_distances = list()
        for interval_size in range(0, 24):
            distance = 0
            for manifestation_num in range(0, 12):
                start_midi = 21 + manifestation_num
                end_midi = start_midi + interval_size
                distance += (self._key_positions[end_midi] - self._key_positions[start_midi])
            avg_distances.append(distance/12)

        self._bounds_for_semitone_interval = list()
        self._bounds_for_semitone_interval.append(0)

        for i in range(1, len(avg_distances)):
            if i == 1:
                self._bounds_for_semitone_interval.append(0)
            else:
                self._bounds_for_semitone_interval.append((avg_distances[i] + avg_distances[i-1])/2.0)

    @staticmethod
    def horizontal_key_positions():
        """
        Return a dictionary mapping MIDI pitch numbers to the millimeter offsets
        to their lengthwise center lines on the keyboard.
        """
        positions = dict()
        #           A    A#    B  C     C#   D   D#  E     F  F#    G
        offsets = [11.5, 15.5, 8, 23.5, 9.5, 14, 14, 9.5, 23.5, 8, 15.5, 11.5]
        cycle_index = 0
        value = 0
        for midi_id in range(21, 109):
            value += offsets[cycle_index % len(offsets)]
            positions[midi_id] = value
            cycle_index += 1

        return positions


class ImaginaryBlackKeyRuler(CacheRuler):
    def __init__(self, preload=False):
        super().__init__(preload=preload)

    def calculated_distance(self, from_midi, to_midi):
        d = 0
        black_to_left = is_black(from_midi)
        left_midi = from_midi
        right_midi = to_midi
        if from_midi > to_midi:
            left_midi = to_midi
            right_midi = from_midi
        for midi in range(left_midi + 1, right_midi + 1):
            if is_white(midi) and not black_to_left:
                d += 1
            black_to_left = is_black(midi)
            d += 1
        if from_midi > to_midi:
            d *= -1
        return d
