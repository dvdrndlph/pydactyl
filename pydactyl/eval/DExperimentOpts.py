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


class DExperimentOpts:
    def __init__(self, opts):
        self.engine = opts['engine']
        self.pickling = True
        self.consonance_threshold = c.CHORD_MS_THRESHOLD
        if 'consonance_threshold' in opts:
            self.consonance_threshold = opts['consonance_threshold']
        if 'pickling' in opts:
            self.pickling = opts['pickling']
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


