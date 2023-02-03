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
import pydactyl.crf.CrfUtil as c

VALID_GROUPINGS = {
    'example': True,
    'segment': True,
    'piece': True
}


class DExperimentOpts:
    def __init__(self, opts):
        self.segment = True
        if 'segmenting' in opts:
            self.segmenting = opts['segmenting']

        self.random_state = 14
        if 'random_state' in opts:
            self.random_state = opts['random_state']

        self.consonance_threshold = c.CHORD_MS_THRESHOLD
        if 'consonance_threshold' in opts:
            self.consonance_threshold = opts['consonance_threshold']
        self.pickling = True
        if 'pickling' in opts:
            self.pickling = opts['pickling']

        self.engine = opts['engine']

        self.group_by = 'segment'
        if 'group_by' in opts:
            if opts['group_by'] in VALID_GROUPINGS:
                self.group_by = opts['group_by']
            else:
                raise Exception("Invalid group_by setting: {}".format(opts['group_by']))

        self.holdout_size = 0.30
        if 'holdout_size' in opts:
            self.holdout_size = opts['holdout_size']
        self.holdout_predefined = True
        if 'holdout_predefined' in opts:
            self.holdout_predefined = opts['holdout_predefined']
        feats = opts['model_features']
        self.model_features = feats
        self.model_version = feats.CRF_VERSION
        self.note_func = feats.my_note2features
        self.reverse = feats.REVERSE_NOTES
        self.staffs = opts['staffs']
        self.test_method = opts['test_method']
        self.fold_count = opts['fold_count']
        self.corpus_names = opts['corpus_names']
        self.segregate_hands = opts['segregate_hands']
        self.params = opts['params']
        self.param_grid = None
        if 'param_grid' in opts:
            self.param_grid = opts['param_grid']


