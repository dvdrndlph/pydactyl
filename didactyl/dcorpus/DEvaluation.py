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

class DEvaluation:
    """
    Class to keep track of how well each of the top ranked outputs of a model
    performed against a given gold standard.
    """
    def __init__(self, gold_d_annotation, scores=[]):
        self._gold_annotation = gold_d_annotation
        self._scores = list()
        self.scores(scores)

    def gold_d_annotation(self, annotation=None):
        if annotation:
            self._gold_annotation = annotation
        return self._gold_annotation

    def scores(self, scores=[]):
        if scores:
            for score in scores:
                self._scores.append(score)
        return self._scores

    def append_score(self, score):
        if score:
            self._scores.append(score)
            return True
        return False

    def score(self, index=None):
        if index is None:
            index = 0
        if len(self._scores) > 0 and index < len(self._scores):
            return self._scores[index]
        return None


class DEvaluationSet:
    def __init__(self, d_score, gold_indices=[]):
        self._gold_by_index = dict()
        self._gold_by_id = dict()
        abcd_header = d_score.abcd_header()
        gold_index = 0
        for annot in abcd_header.annotations():
            if gold_indices and gold_index not in gold_indices:
                continue
            evaluation = DEvaluation(gold_d_annotation=annot)
            self._gold_by_index[gold_index] = evaluation
            gold_id = annot.abcdf_id()
            self._gold_by_id[gold_id] = evaluation

    def evaluation(self, gold_index=None, gold_id=None):
        if gold_index in self._gold_by_index:
            return self._gold_by_index[gold_index]
        elif gold_id in self._gold_by_id:
            return self._gold_by_id[gold_id]
        return None

    def scores(self, gold_index=None, gold_id=None, scores=[]):
        evaluation = self.evaluation(gold_index=gold_index, gold_id=gold_id)
        if evaluation is None:
            raise Exception("No DEvaluation found")
        if scores:
            evaluation.scores(scores=scores)
        return evaluation.scores()
