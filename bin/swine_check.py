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
import subprocess
import pprint
from music21 import converter
from pydactyl.dcorpus.DScore import DScore
from pydactyl.dcorpus.PigInOut import PigOut, PIG_BASE_DIR, PIG_SCRIPT_DIR, \
    PIG_ABCD_DIR, PIG_FINGERING_DIR, PIG_STD_DIR, PIG_FILE_SUFFIX, \
    PIG_SEGREGATED_STD_DIR, PIG_SEGREGATED_FINGERING_DIR, \
    PIG_RESULT_FHMM3_DIR
import mido

ID = '002-1'
PIG_BIN_DIR = PIG_BASE_DIR + 'SourceCode/Binary/'
HMM1_CMD = PIG_SCRIPT_DIR + 'run_FHMM1.sh'
HMM2_CMD = PIG_SCRIPT_DIR + 'run_FHMM2.sh'
HMM3_CMD = PIG_SCRIPT_DIR + 'run_FHMM3.sh'
SIMPLE_MATCH_RATE_CMD = PIG_BIN_DIR + 'Evaluate_SimpleMatchRate'
PREDICTION_DIR = '/tmp/prediction/'
NAKAMURA_PREDICTION_DIR = '/tmp/nakamura'

mf_path = PIG_ABCD_DIR + ID + '.mid'
hdr_path = PIG_ABCD_DIR + ID + '.abcd'
s = converter.parse(mf_path)

mid = mido.MidiFile(mf_path)
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    msg_cnt = 0
    for msg in track:
        if msg.type == 'note_on':
            msg_cnt += 1
            # print(msg)
    print("note_on count: {}".format(msg_cnt))

# dig_on_swine.py has translated the original PIG to ABCDF/MIDI.
# We now perform the follwing transforms:
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

gt_pig_files, gt_pig_paths = PigOut.pig_files_and_paths(piece_id='001')
pred_pig_path = PIG_RESULT_FHMM3_DIR + '001-1' + PIG_FILE_SUFFIX
print(gt_pig_paths)
print(pred_pig_path)
result = PigOut.single_prediction_complex_match_rates(gt_pig_paths=gt_pig_paths, pred_pig_path=pred_pig_path)
print(result)
match_count, note_count, m_gen = \
    PigOut.my_single_prediction_m_gen(gt_pig_paths=gt_pig_paths, pred_pig_path=pred_pig_path)
print(m_gen)
print("Difference between my m_gen and EvaluateMultipleGroundTruth: {}".format(m_gen - result['general']))
print("")
avg_m_gen, piece_m_gens = PigOut.my_average_m_gen(fingering_files_dir=PIG_FINGERING_DIR,
                                                  prediction_dir=PIG_RESULT_FHMM3_DIR, weight=False)
pprint.pprint(piece_m_gens)
print("My average M_gen for FHMM3: {}".format(avg_m_gen))
avg_m_gen, piece_m_gens = PigOut.my_average_m_gen(fingering_files_dir=PIG_FINGERING_DIR,
                                                  prediction_dir=PIG_RESULT_FHMM3_DIR, weight=True)
print("My normalized average M_gen for FHMM3: {}".format(avg_m_gen))
# exit()

# Check that Nakamura uses channel information for predictions.
pig_file_name = '001-1' + PIG_FILE_SUFFIX
pig_path = PIG_FINGERING_DIR + pig_file_name
pig_pred_path = '/tmp/' + pig_file_name
PigOut.run_hmm(model='fhmm3', in_fin=pig_path, out_fin=pig_pred_path)
zero_path = '/tmp/zero' + PIG_FILE_SUFFIX
zero_pred_path = '/tmp/zero-result' + PIG_FILE_SUFFIX
PigOut.zero_all_channels(pig_path=pig_path, to_file=zero_path)
PigOut.run_hmm(model='fhmm3', in_fin=zero_path, out_fin=zero_pred_path)
match_rate = PigOut.simple_match_rate(pig_pred_path, zero_pred_path)
# Should match 100% if channel is not an input to the model.
if match_rate['rate'] < 1.0:
    print("The FHMM3 model does rely on channels as input.")
print(match_rate)

# Test the overall accuracy and label-wise P/R/F for the order-three HMM model.
results = PigOut.nakamura_accuracy(fingering_files_dir=PIG_FINGERING_DIR, model='fhmm3', output="text")
# results = PigOut.nakamura_accuracy(fingering_files_dir=PIG_STD_DIR, model='fhmm3', output="text")
results = PigOut.nakamura_accuracy(fingering_files_dir=PIG_SEGREGATED_FINGERING_DIR, model='fhmm3', output="text")
# results = PigOut.nakamura_accuracy(fingering_files_dir=PIG_SEGREGATED_STD_DIR, model='fhmm3', output="text")

# corpse = DCorpus()
# corpse.append(corpus_path=mf_path, header_path=hdr_path)
# for da_score in corpse.d_score_list():
#     piggo = PigOut(d_score=da_score)
#     to_file = "/tmp/" + ID + "_fingering.txt"
#     pork = piggo.transform(annotation_index=0, to_file=to_file)

# Reproduce the Nakamura results with the code they have provided.
# We use the pretrained model.
print("")
model_names = ['fhmm1', 'fhmm2', 'fhmm3', 'chmm', 'human']
PigOut.output_nakamura_metrics_heading()
for model in model_names:
    results, piece_results = PigOut.nakamura_metrics(model=model, weight=False)
    # print("Nakamura {:>5} model (non-weighted): {}".format(model, results))
    # std_results = PigOut.nakamura_published(fingering_files_dir=PIG_STD_DIR, model=model, weight=False)
    # print("std_pig  {:>5} model (non-weighted): {}".format(model, std_results))
    results, piece_results = PigOut.nakamura_metrics(fingering_files_dir=PIG_SEGREGATED_FINGERING_DIR,
                                                     model=model, weight=False)
print("")
for model in model_names:
    results, piece_results = PigOut.nakamura_metrics(model=model, weight=True)
    # print("Nakamura {:>5} model (normalized)    : {}".format(model, normalized_results))
    # normalized_std_results = PigOut.nakamura_published(fingering_files_dir=PIG_STD_DIR, model=model, weight=True)
    # print("std_pig  {:>5} model (normalized): {}".format(model, normalized_std_results))
    results, piece_results = PigOut.nakamura_metrics(fingering_files_dir=PIG_SEGREGATED_FINGERING_DIR,
                                                     model=model, weight=True)
print("")
print("Done")
