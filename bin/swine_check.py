#!/usr/bin/env python
__author__ = 'David Randolph'
# Copyright (c) 2020 David A. Randolph.
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
#
# Validate event count in dig_on_swine output files.
import os
import subprocess
from pathlib import Path
from music21 import abcFormat, converter, stream
# from pydactyl.dcorpus.DCorpus import DCorpus, DAnnotation
from pydactyl.dcorpus.DScore import DScore
from pydactyl.dcorpus.PigInOut import PigOut
from mido import MidiFile

ID = '002-1'
PIG_BASE_DIR = '/Users/dave/tb2/didactyl/dd/corpora/pig/'
PIG_BIN_DIR = PIG_BASE_DIR + 'SourceCode/Binary/'
PIG_SCRIPT_DIR = PIG_BASE_DIR + 'SourceCode/'
HMM1_CMD = PIG_SCRIPT_DIR + 'run_FHMM1.sh'
HMM2_CMD = PIG_SCRIPT_DIR + 'run_FHMM2.sh'
HMM3_CMD = PIG_SCRIPT_DIR + 'run_FHMM3.sh'
SIMPLE_MATCH_RATE_CMD = PIG_BIN_DIR + 'Evaluate_SimpleMatchRate'
PIG_ABCD_DIR = PIG_BASE_DIR + 'PianoFingeringDataset_v1.00/individual_abcd/'
PIG_STD_DIR = PIG_BASE_DIR + 'PianoFingeringDataset_v1.00/std_pig/'
PREDICTION_DIR = '/tmp/prediction/'
NAKAMURA_PREDICTION_DIR = '/tmp/nakamura'

mf_path = PIG_ABCD_DIR + ID + '.mid'
hdr_path = PIG_ABCD_DIR + ID + '.abcd'
s = converter.parse(mf_path)

mid = MidiFile(mf_path)
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    msg_cnt = 0
    for msg in track:
        if msg.type == 'note_on':
            msg_cnt += 1
            # print(msg)
    print("note_on count: {}".format(msg_cnt))

# ABCDF/MIDI => DScore => Standardized PIG
# Then compare generated std_pig to original std_pig.
d_score = DScore(midi_file_path=mf_path, abcd_header_path=hdr_path, title=ID)
piggo = PigOut(d_score=d_score)
to_file = PREDICTION_DIR + ID + "_fingering.txt"
pork = piggo.transform(annotation_index=0, to_file=to_file)

original_pig_file = PIG_STD_DIR + ID + '_fingering.txt'
cmd = "{} {} {}".format(SIMPLE_MATCH_RATE_CMD, original_pig_file, to_file)

returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
print('returned value:', returned_value)

# corpse = DCorpus()
# corpse.append(corpus_path=mf_path, header_path=hdr_path)
# for da_score in corpse.d_score_list():
#     piggo = PigOut(d_score=da_score)
#     to_file = "/tmp/" + ID + "_fingering.txt"
#     pork = piggo.transform(annotation_index=0, to_file=to_file)

# Reproduce the Nakamura results with the code they have provided.
# We use the pretrained model.
model_names = ['fhmm1', 'fhmm2', 'fhmm3', 'chmm']
for model in model_names:
    results = PigOut.nakamura_published(model=model, normalize=False)
    print("Nakamura {:>5} model (non-normalized): {}".format(model, results))
print("")
for model in model_names:
    normalized_results = PigOut.nakamura_published(model=model, normalize=True)
    print("Nakamura {:>5} model (normalized)    : {}".format(model, normalized_results))

print("Done")
