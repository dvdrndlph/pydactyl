__author__ = 'David Randolph'

import os.path
# Copyright (c) 2021-2022 David A. Randolph.
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
from abc import ABC
import copy
import re
import csv
from pydactyl.dcorpus.DCorpus import DCorpus, DAnnotation
from pydactyl.dcorpus.PigInOut import PIG_SEGREGATED_ABCD_DIR
from pydactyl.dcorpus.PianoFingering import PianoFingering
from pydactyl.dcorpus.DEvaluation import DEvaluation, DEvalFunction
from pydactyl.dactyler.Parncutt import Parncutt, Jacobs, Badgerow, Balliauw, Ruler
from pydactyl.dactyler.Parncutt import FINGER_SPANS, BALLIAUW_LARGE_FINGER_SPANS, \
    BALLIAUW_MEDIUM_FINGER_SPANS, BALLIAUW_SMALL_FINGER_SPANS, ImaginaryBlackKeyRuler
from pydactyl.dactyler.Random import Random

CORPORA_DIR = os.path.expanduser('~/tb2/didactyl/dd/corpora/')
OUTPUT_DIR = os.path.expanduser('~/tb2/doc/data/badgerow/')
PIG_BASE_DIR = CORPORA_DIR + 'pig/PianoFingeringDataset_v1.00/'
PIG_INDY_DIR = PIG_BASE_DIR + 'individual_abcd/'
PIG_DIR = PIG_BASE_DIR + 'abcd/'
BERINGER_DIR = CORPORA_DIR + 'beringer/'
SCALES_DIR = BERINGER_DIR + 'scales/'
ARPEGGIOS_DIR = BERINGER_DIR + 'arpeggios/'
BROKEN_DIR = BERINGER_DIR + 'broken_chords/'
SCALES_STD_PIG_DIR = SCALES_DIR + 'std_pig/'
ARPEGGIOS_STD_PIG_DIR = ARPEGGIOS_DIR + 'std_pig/'
BROKEN_STD_PIG_DIR = BROKEN_DIR + 'std_pig/'
COMPLETE_LAYER_ONE_DIR = CORPORA_DIR + 'clementi/complete_layer_one/'
COMPLETE_LAYER_ONE_STD_PIG_DIR = COMPLETE_LAYER_ONE_DIR + 'std_pig/'

PIG_LIST_FILE = PIG_BASE_DIR + 'List.csv'

QUERY = dict()

QUERY['layer_one_by_annotator'] = '''
    with expo as (
        select e.id as experimentId, e.clientId, s.`type`,
               s.`description`, s.id as studyId
          from diii_nm.study s
         inner join diii_nm.experiment e
            on s.id = e.studyId
         where (description like '%Layer One%')
   ) select a.abcDF as 'fingering',
            a.clientId as 'authority',
            concat(`type`, '_study_', studyId) as 'transcriber',
            e.`description` as 'comment'
       from diii_nm.annotation a
      inner join expo e
         on a.experimentId = e.experimentId
      where a.abcDF is not null
        and a.abcDF not like '%x%'
        and a.selectionId = '{}'
'''

QUERY['full_american_by_annotator'] = '''
   select f.upper_staff as fingering,
           f.subject as 'authority',
           'Pydactyl' as 'transcriber'
      from finger f
     inner join parncutt p
        on f.exercise = p.exercise
     where f.exercise = {} 
       and f.upper_staff is not null
       and length(f.upper_staff) = p.length_full;  
'''

QUERY['full_american'] = '''
    select f.upper_staff as fingering,
           count(*) as weight,
           'Various Didactyl' as 'authority',
           'Pydactyl' as 'transcriber'
      from finger f
     inner join parncutt p
        on f.exercise = p.exercise
     where f.exercise = {}
       and f.upper_staff is not null
       and length(f.upper_staff) = p.length_full
     group by f.upper_staff
     order by weight desc'''

QUERY['all_american'] = '''
    select parncutt_fingering as fingering,
           total as weight,
           'Various Didactyl' as 'authority',
           'Pydactyl' as 'transcriber'
      from parncutt_binary
     where exercise = {}
     order by weight desc'''

QUERY['pure_american'] = '''
    select parncutt as fingering,
           'Various Didactyl' as 'authority',
           'Pydactyl' as 'transcriber',
           count(*) as weight
      from parncutt_american_pure
     where exercise = {} 
       and Advised = 'No'
     group by parncutt
     order by weight desc'''

QUERY['parncutt_published'] = '''
    select fingering,
           'Various Didactyl' as 'authority',
           'Pydactyl' as 'transcriber',
           subject_count as weight
      from parncutt_published
     where exercise = {}
     order by weight desc'''

FULL_ABC_QUERY = '''
    select exercise as piece_id,
           abc_full as abc_str
      from parncutt
     order by exercise'''

FRAGMENT_ABC_QUERY = '''
    select exercise as piece_id,
           abc_fragment as abc_str
      from parncutt
     order by exercise'''

LAYER_ONE_QUERY = '''
    select id as piece_id,
           abc_full as abc_str
      from diii_nm.selection_detail
     where id like 'c%1';
'''

BASE_HEADINGS = ['corpus', 'model', 'title', 'notes', 'ann_id', 'weight']
RANK_METHODS = ['hmg', 'norm_hmg',
                'al', 'norm_al',
                'rho_no_d', 'rho_uni_d',
                'p_sat', 'tri_p_sat', 'nua_p_sat', 'rlx_p_sat',
                'tri_D', 'nua_D', 'rlx_D',
                'norm_tri_D', 'norm_nua_D', 'norm_rlx_D']
BASE_RANK_HEADINGS = copy.deepcopy(BASE_HEADINGS)
BASE_RANK_HEADINGS.append('rank')
RANK_HEADINGS = copy.deepcopy(BASE_RANK_HEADINGS)
RANK_HEADINGS.extend(RANK_METHODS)
ERR_METHODS = ['hmg', 'hmg_rho', 'al', 'al_rho', 'clvrst',
               'tri', 'tri_rho', 'tri_nua', 'tri_nua_rho', 'tri_rlx', 'tri_rlx_rho', 'tri_clvrst']
ERR_HEADINGS = copy.deepcopy(BASE_HEADINGS)
ERR_HEADINGS.extend(ERR_METHODS)

WEIGHT_RE = r"^Weight:\s*(\d+)$"


class Corporeal(ABC):
    def __init__(self, output_dir=OUTPUT_DIR, rank_methods=RANK_METHODS, err_methods=ERR_METHODS, staff="upper"):
        self.output_dir = output_dir
        self.rank_methods = rank_methods
        self.rank_headings = copy.deepcopy(BASE_RANK_HEADINGS)
        self.rank_headings.extend(rank_methods)
        self.err_methods = err_methods
        self.err_headings = copy.deepcopy(BASE_HEADINGS)
        self.err_headings.extend(err_methods)
        self._err_result_cache = {}
        self._pivot_report_cache = {}
        self._rank_result_cache = {}
        self._staff = staff

    @staticmethod
    def enrich_pig_corpus(the_corpus: DCorpus):
        composer_for_prefix = dict()
        with open(PIG_LIST_FILE) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_num = 0
            for line in csv_reader:
                if line_num != 0:
                    file_prefix = line[0]
                    composer = line[1]
                    composer_for_prefix[file_prefix] = composer
                line_num += 1
        for d_score in the_corpus.d_score_list():
            title = d_score.title()
            prefix, extension = title.split(sep='-')
            composer = composer_for_prefix[prefix]
            d_score.composer(composer, with_periods=True)

    @staticmethod
    def enrich_beringer_corpus(the_corpus: DCorpus):
        for d_score in the_corpus.d_score_list():
            composer = 'Beringer'
            d_score.composer(composer, with_periods=True)

    @staticmethod
    def enrich_clementi_corpus(the_corpus: DCorpus):
        for d_score in the_corpus.d_score_list():
            composer = 'Clementi'
            d_score.composer(composer, with_periods=True)

    @staticmethod
    def enrich_czerny_corpus(the_corpus: DCorpus):
        for d_score in the_corpus.d_score_list():
            composer = 'Czerny'
            d_score.composer(composer, with_periods=True)

    def get_corpus(self, corpus_name):
        if corpus_name == 'pig':
            the_corpus = DCorpus()
            the_corpus.append_dir(corpus_dir=PIG_DIR, via_midi=True, split_header_extension='abcd')
            Corporeal.enrich_pig_corpus(the_corpus)
            return the_corpus

        if corpus_name == 'pig_indy':
            the_corpus = DCorpus()
            the_corpus.append_dir(corpus_dir=PIG_INDY_DIR, via_midi=True, split_header_extension='abcd')
            Corporeal.enrich_pig_corpus(the_corpus)
            return the_corpus

        if corpus_name == 'pig_seg':
            the_corpus = DCorpus()
            the_corpus.append_dir(corpus_dir=PIG_SEGREGATED_ABCD_DIR, via_midi=True, split_header_extension='abcd')
            Corporeal.enrich_pig_corpus(the_corpus)
            return the_corpus

        if corpus_name == 'scales':
            the_corpus = DCorpus()
            the_corpus.append_dir(corpus_dir=SCALES_DIR)
            Corporeal.enrich_beringer_corpus(the_corpus)
            return the_corpus

        if corpus_name == 'arpeggios':
            the_corpus = DCorpus()
            the_corpus.append_dir(corpus_dir=ARPEGGIOS_DIR)
            Corporeal.enrich_beringer_corpus(the_corpus)
            return the_corpus

        if corpus_name == 'broken':
            the_corpus = DCorpus()
            the_corpus.append_dir(corpus_dir=BROKEN_DIR)
            Corporeal.enrich_beringer_corpus(the_corpus)
            return the_corpus

        if corpus_name == 'beringer':
            the_corpus = DCorpus()
            for subdir in [SCALES_DIR, ARPEGGIOS_DIR, BROKEN_DIR]:
                the_corpus.append_dir(corpus_dir=subdir)
            Corporeal.enrich_beringer_corpus(the_corpus)
            return the_corpus

        if corpus_name == 'complete_layer_one':
            the_corpus = DCorpus()
            the_corpus.append_dir(corpus_dir=COMPLETE_LAYER_ONE_DIR)
            Corporeal.enrich_clementi_corpus(the_corpus)
            return the_corpus

        if corpus_name == 'clementi':
            the_corpus = DCorpus()
            return the_corpus

        piece_query = FRAGMENT_ABC_QUERY
        if corpus_name in ('full_american', 'full_american_by_annotator'):
            piece_query = FULL_ABC_QUERY

        if corpus_name in ('layer_one_by_annotator'):
            piece_query = LAYER_ONE_QUERY

        the_corpus = DCorpus()
        the_corpus.assemble_and_append_from_db(piece_query=piece_query,
                                               fingering_query=QUERY[corpus_name])
        Corporeal.enrich_czerny_corpus(the_corpus)
        return the_corpus

    def get_model(self, model_name, weights=None):
        if model_name == 'random':
            model = Random()
        elif model_name == 'parncutt':
            model = Parncutt()
        elif model_name == 'jacobs':
            model = Jacobs()
        elif model_name == 'jarules':
            model = Jacobs(finger_spans=FINGER_SPANS, ruler=Ruler())
        elif model_name == 'jaball':
            model = Jacobs(finger_spans=BALLIAUW_LARGE_FINGER_SPANS, ruler=ImaginaryBlackKeyRuler())
        elif model_name == 'jaSball':
            model = Jacobs(finger_spans=BALLIAUW_SMALL_FINGER_SPANS, ruler=ImaginaryBlackKeyRuler())
        elif model_name == 'jaMball':
            model = Jacobs(finger_spans=BALLIAUW_MEDIUM_FINGER_SPANS, ruler=ImaginaryBlackKeyRuler())
        elif model_name == 'badgerow':
            # model = Badgerow(finger_spans=FINGER_SPANS, ruler=PhysicalRuler())
            # model = Badgerow(finger_spans=BALLIAUW_LARGE_FINGER_SPANS, ruler=ImaginaryBlackKeyRuler())
            model = Badgerow()
        elif model_name == 'badpar':
            # model = Badgerow(finger_spans=FINGER_SPANS, ruler=PhysicalRuler())
            # model = Badgerow(finger_spans=BALLIAUW_FINGER_SPANS, ruler=ImaginaryBlackKeyRuler())
            model = Badgerow(finger_spans=FINGER_SPANS)
        elif model_name == 'balliauw':
            model = Balliauw()
        elif model_name == 'badball':
            model = Badgerow(finger_spans=BALLIAUW_LARGE_FINGER_SPANS, ruler=ImaginaryBlackKeyRuler())
        elif model_name == 'badSball':
            model = Badgerow(finger_spans=BALLIAUW_SMALL_FINGER_SPANS, ruler=ImaginaryBlackKeyRuler())
        elif model_name == 'badMball':
            model = Badgerow(finger_spans=BALLIAUW_MEDIUM_FINGER_SPANS, ruler=ImaginaryBlackKeyRuler())
        else:
            raise Exception("Bad model")

        if weights:
            model.init_rule_weights(weights)
        return model

    def get_fingered_system_scores(self, loaded_model: Parncutt, model_name, version=None, score_count=5, weights=None):
        advice = loaded_model.generate_advice(staff=self._staff, score_index=0, k=score_count)
        # print(advice)

        authority = model_name
        if version:
            authority = model_name + '_' + version
        base_score = loaded_model.score_by_index(0)
        sys_scores = []
        for i in range(score_count):
            sys_score = copy.deepcopy(base_score)
            sys_score.remove_annotations()
            abcdf = advice[0][i]
            ann = DAnnotation(abcdf=abcdf, authority=authority)
            sys_score.annotate(d_annotation=ann)
            PianoFingering.finger_score(d_score=sys_score, staff=self._staff)
            sys_scores.append(sys_score)
        return sys_scores

    # Deprecate?
    def get_system_scores(self, model_name, d_score, k=5, weights=None):
        if model_name == 'ideal':
            return DEvaluation.get_best_pseudo_model_scores(d_score=d_score, staff=self._staff)

        model = self.get_model(model_name=model_name, weights=weights)
        advice = model.generate_advice(staff=self._staff, score_index=0, k=k)
        # print(advice)

        sys_scores = []
        for r in (range(k)):
            sys_score = copy.deepcopy(d_score)
            sys_score.remove_annotations()
            abcdf = advice[0][r]
            ann = DAnnotation(abcdf=abcdf, authority=model_name)
            sys_score.annotate(d_annotation=ann)
            PianoFingering.finger_score(d_score=sys_score, staff=self._staff)
            sys_scores.append(sys_score)
        return sys_scores

    def set_err_result(self, err_result, tag, evil):
        evil.parameterize()
        if tag == 'hmg':
            evil.delta_function(DEvalFunction.delta_hamming)
        elif tag == 'al':
            evil.delta_function(DEvalFunction.delta_adjacent_long)
        elif tag == 'al_rho':
            evil.delta_function(DEvalFunction.delta_adjacent_long)
            evil.mu_function(DEvalFunction.mu_scale)
        elif tag == 'hmg_rho':
            evil.delta_function(DEvalFunction.delta_hamming)
            evil.mu_function(DEvalFunction.mu_scale)
        elif tag == 'tri':
            evil.mu_function(None)
        elif tag == 'tri_nua':
            evil.mu_function(None)
            evil.tau_function(DEvalFunction.tau_nuanced)
        elif tag == 'tri_rlx':
            evil.mu_function(None)
            evil.tau_function(DEvalFunction.tau_relaxed)
        elif tag == 'tri_rlx_rho':
            evil.mu_function(DEvalFunction.mu_scale)
            evil.tau_function(DEvalFunction.tau_relaxed)
        elif tag == 'tri_nua_rho':
            evil.mu_function(DEvalFunction.mu_scale)
            evil.tau_function(DEvalFunction.tau_nuanced)
        elif tag == 'tri_rho':
            evil.mu_function(DEvalFunction.mu_scale)
            evil.tau_function(DEvalFunction.tau_trigram)
        elif tag == 'clvrst' or tag == 'tri_clvrst':
            evil.parameterize(delta_function=DEvalFunction.delta_adjacent_long,
                              tau_function=DEvalFunction.tau_relaxed,
                              decay_function=DEvalFunction.decay_uniform,
                              mu_function=DEvalFunction.mu_scale, rho_decay_function=DEvalFunction.decay_uniform)
        else:
            raise Exception("Unknown tag: {}".format(tag))
        if re.match('^tri.*', tag):
            err_result[tag] = evil.expected_reciprocal_rank(trigram=True)
        else:
            err_result[tag] = evil.expected_reciprocal_rank()

    def _get_err_result_set(self, evil, corpus_name, model_name, title, note_count, annot_id, weight):
        err_result = {}

        err_result['corpus'] = corpus_name
        err_result['model'] = model_name
        err_result['title'] = title
        err_result['notes'] = note_count
        err_result['ann_id'] = annot_id
        err_result['weight'] = weight

        for tag in self.err_methods:
            self.set_err_result(err_result=err_result, tag=tag, evil=evil)
        return err_result

    def get_cmt_err(self, err_results):
        cmt_err = {}
        corpus = ''
        title = ''
        model = ''

        for res in err_results:
            if res['corpus'] != corpus or res['model'] != model or res['title'] != title:
                corpus = res['corpus']
                model = res['model']
                title = res['title']

            if (corpus, model, title) not in cmt_err:
                cmt_err[(corpus, model, title)] = {'sums': {}, 'people': 0, 'notes': res['notes']}
                for method in self.err_methods:
                    cmt_err[(corpus, model, title)]['sums'][method] = 0

            cmt_err[(corpus, model, title)]['people'] += res['weight']
            for meth in self.err_methods:
                cmt_err[(corpus, model, title)]['sums'][meth] += res[meth] * res['weight']
        return cmt_err

    def get_errs_per_person(self, cmt_err):
        """
        Returns err_per_person, weighted_err_per_person, note_total.
        mean_err_phrase is the mean of all ERR measures assigned on a particular phrase across all people.
        weighted_err_per_person aggregates all individual ERR scores weighted by the
        length of the phrase used to generate each of them. If we divide this
        aggregate by the sum total notes, we get the sum total ERR weighted by
        phrase length. If we divide this by the total number of annotations, we get weighted_mean_err.
        If we ignore the weighting by phrase length (notes) and treat each ERR measure the same, we get mean_err.
        mean_err seems easier to deal with for statistical analysis.
        """
        note_total = {}
        phrase_count = {}
        weighted_err_per_person = {}
        err_per_person = {}
        for (corpus, model, title) in cmt_err:
            if (corpus, model) not in weighted_err_per_person:
                weighted_err_per_person[(corpus, model)] = {}
                err_per_person[(corpus, model)] = {}
                note_total[(corpus, model)] = 0
                phrase_count[(corpus, model)] = 0
            res = cmt_err[(corpus, model, title)]
            note_total[(corpus, model)] += res['notes']
            phrase_count[(corpus, model)] += 1

            for meth in self.err_methods:
                if meth not in weighted_err_per_person[(corpus, model)]:
                    weighted_err_per_person[(corpus, model)][meth] = 0
                    err_per_person[(corpus, model)][meth] = 0
                phrase_avg_err = float(res['sums'][meth]) / res['people']
                weighted_err_per_person[(corpus, model)][meth] += phrase_avg_err * res['notes']
                err_per_person[(corpus, model)][meth] += phrase_avg_err
        result = {'err_per_person': err_per_person,
                  'weighted_err_per_person': weighted_err_per_person,
                  'note_total': note_total,
                  'phrase_count': phrase_count}
        return result

    def _get_weighted_mean_err(self, weighted_err_per_person, note_total):
        weighted_mean_err = {}
        for (corpus, model) in weighted_err_per_person:
            for method in weighted_err_per_person[(corpus, model)]:
                method_weighted_mean_err = weighted_err_per_person[(corpus, model)][method]
                method_weighted_mean_err /= note_total[(corpus, model)]
                weighted_mean_err[(corpus, model, method)] = method_weighted_mean_err
        return weighted_mean_err

    def _get_mean_err(self, err_per_person, phrase_count):
        mean_err = {}
        for (corpus, model) in err_per_person:
            for method in err_per_person[(corpus, model)]:
                method_mean_err = err_per_person[(corpus, model)][method]
                method_mean_err /= phrase_count[(corpus, model)]
                mean_err[(corpus, model, method)] = method_mean_err
        return mean_err

    def get_all_results(self, corpus_name, model, model_name, k=5, staff="upper", full_context=True, version='0000'):
        if (corpus_name, model_name, version, staff, k, full_context) in self._err_result_cache:
            cached_err_results = self._err_result_cache[(corpus_name, model_name, version, staff, k, full_context)]
            cached_pivot_reports = self._pivot_report_cache[(corpus_name, model_name, version, staff, k, full_context)]
            cached_rank_results = self._rank_result_cache[(corpus_name, model_name, version, staff, k, full_context)]
            return cached_err_results, cached_pivot_reports, cached_rank_results

        rank_results = []
        err_results = []
        pivot_reports = {}
        da_corpus = self.get_corpus(corpus_name=corpus_name)
        for da_score in da_corpus.d_score_list():
            model.load_score_as_corpus(d_score=da_score)
            system_scores = self.get_fingered_system_scores(loaded_model=model,
                                                            model_name=model_name, version=version)
            title = da_score.title()
            note_count = da_score.note_count(staff=self._staff)
            abcdh = da_score.abcd_header()
            last_annot_id = abcdh.annotation_count()
            for annot_id in range(1, last_annot_id + 1):
                annot = abcdh.annotation_by_id(annot_id)
                comment = annot.comments()
                mat = re.match(WEIGHT_RE, comment)
                if mat:
                    weight = int(mat.group(1))
                else:
                    weight = 1

                human_score = copy.deepcopy(da_score)
                PianoFingering.finger_score(d_score=human_score, staff=self._staff, id=annot_id)

                evil = DEvaluation(human_score=human_score, system_scores=system_scores,
                                   staff=staff, full_context=full_context)
                pivot_report_key = (model_name, corpus_name, title)
                pivot_heading = "{} over {} {} human {}".format(model_name, corpus_name, title, annot_id)
                pivot_report = evil.pivot_count_report(heading=pivot_heading)
                if pivot_report_key not in pivot_reports:
                    pivot_reports[pivot_report_key] = []
                pivot_reports[pivot_report_key].append(pivot_report)

                err_result = self._get_err_result_set(evil, corpus_name=corpus_name, model_name=model_name,
                                                     title=title, note_count=note_count,
                                                     annot_id=annot_id, weight=weight)
                err_results.append(err_result)

                for i in range(k):
                    rank = i + 1
                    result = self._get_result_set(evil, corpus_name=corpus_name, model_name=model_name,
                                                 title=title, note_count=note_count,
                                                 annot_id=annot_id, weight=weight, rank=rank)
                    rank_results.append(result)
                    evil.parameterize()  # Reset to defaults.
        self._err_result_cache[(corpus_name, model_name, version, staff, k, full_context)] = err_results
        self._pivot_report_cache[(corpus_name, model_name, version, staff, k, full_context)] = pivot_reports
        self._rank_result_cache[(corpus_name, model_name, version, staff, k, full_context)] = rank_results
        return err_results, pivot_reports, rank_results

    def get_err_results(self, corpus_name, model, model_name, staff="upper", full_context=True, version='0000'):
        if (corpus_name, model_name, version, staff) in self._err_result_cache:
            cached_err_results = self._err_result_cache[(corpus_name, model_name, version, staff)]
            return cached_err_results

        err_results = []
        da_corpus = self.get_corpus(corpus_name=corpus_name)
        for da_score in da_corpus.d_score_list():
            model.load_score_as_corpus(d_score=da_score)
            system_scores = self.get_fingered_system_scores(loaded_model=model,
                                                            model_name=model_name, version=version)
            title = da_score.title()
            note_count = da_score.note_count(staff=self._staff)
            abcdh = da_score.abcd_header()
            last_annot_id = abcdh.annotation_count()
            for annot_id in range(1, last_annot_id + 1):
                annot = abcdh.annotation_by_id(annot_id)
                comment = annot.comments()
                mat = re.match(WEIGHT_RE, comment)
                if mat:
                    weight = int(mat.group(1))
                else:
                    weight = 1

                human_score = copy.deepcopy(da_score)
                PianoFingering.finger_score(d_score=human_score, staff=self._staff, id=annot_id)

                evil = DEvaluation(human_score=human_score, system_scores=system_scores,
                                   staff=staff, full_context=full_context)
                err_result = self._get_err_result_set(evil, corpus_name=corpus_name, model_name=model_name,
                                                     title=title, note_count=note_count,
                                                     annot_id=annot_id, weight=weight)
                err_results.append(err_result)

        self._err_result_cache[(corpus_name, model_name, version, staff)] = err_results
        return err_results

    def get_mean_err(self, corpus_name, model, model_name, staff="upper", full_context=True, version='0000'):
        err_results = self.get_err_results(corpus_name=corpus_name, model=model, model_name=model_name,
                                           staff=staff, full_context=full_context, version=version)
        cmt_err = self.get_cmt_err(err_results=err_results)
        epp = self.get_errs_per_person(cmt_err=cmt_err)
        mean_err = self._get_mean_err(err_per_person=epp['err_per_person'], phrase_count=epp['phrase_count'])
        return mean_err

    def get_weighted_mean_err(self, corpus_name, model, model_name, staff="upper", full_context=True, version='0000'):
        err_results = self.get_err_results(corpus_name=corpus_name, model=model, model_name=model_name,
                                           staff=staff, full_context=full_context, version=version)
        cmt_err = self.get_cmt_err(err_results=err_results)
        epp = self.get_errs_per_person(cmt_err=cmt_err)
        mean_err = self._get_weighted_mean_err(weighted_err_per_person=epp['weighted_err_per_person'],
                                               note_total=epp['note_total'])
        return mean_err

    def set_result(self, result, tag, rank, evil: DEvaluation):
        evil.parameterize()
        if tag == 'hmg':
            result['hmg'] = evil.big_delta_at_rank(rank=rank)
        elif tag == 'norm_hmg':
            result['norm_hmg'] = evil.big_delta_at_rank(rank=rank, normalized=True)
        elif tag == 'rho_no_d':
            result['rho_no_d'] = evil.pivot_clashes_at_rank(rank=rank)
        elif tag == 'p_sat':
            result['p_sat'] = evil.prob_satisfied(rank=rank)

        elif tag == 'al':
            evil.delta_function(DEvalFunction.delta_adjacent_long)
            result['al'] = evil.big_delta_at_rank(rank=rank)
        elif tag == 'norm_al':
            evil.delta_function(DEvalFunction.delta_adjacent_long)
            result['norm_al'] = evil.big_delta_at_rank(rank=rank, normalized=True)

        elif tag == 'tri_D':
            result['tri_D'] = evil.trigram_big_delta_at_rank(rank=rank)
        elif tag == 'norm_tri_D':
            result['norm_tri_D'] = evil.trigram_big_delta_at_rank(rank=rank, normalized=True)
        elif tag == 'tri_p_sat':
            result['tri_p_sat'] = evil.trigram_prob_satisfied(rank=rank)

        elif re.match('.*nua_.*', tag):
            evil.tau_function(DEvalFunction.tau_nuanced)
            if tag == 'nua_D':
                result['nua_D'] = evil.trigram_big_delta_at_rank(rank=rank)
            elif tag == 'norm_nua_D':
                result['norm_nua_D'] = evil.trigram_big_delta_at_rank(rank=rank, normalized=True)
            elif tag == 'nua_p_sat':
                result['nua_p_sat'] = evil.trigram_prob_satisfied(rank=rank)

        elif re.match('.*rlx_.*', tag):
            evil.tau_function(DEvalFunction.tau_relaxed)
            if tag == 'rlx_D':
                result['rlx_D'] = evil.trigram_big_delta_at_rank(rank=rank)
            elif tag == 'norm_rlx_D':
                result['norm_rlx_D'] = evil.trigram_big_delta_at_rank(rank=rank, normalized=True)
            elif tag == 'rlx_p_sat':
                result['rlx_p_sat'] = evil.trigram_prob_satisfied(rank=rank)

        elif tag == 'rho_uni_d':
            evil.rho_decay_function(DEvalFunction.decay_uniform)
            result['rho_uni_d'] = evil.rho_at_rank(rank=rank)

        else:
            raise Exception("Unknown result tag: {}".format(tag))
        return result

    def _get_result_set(self, evil: DEvaluation, corpus_name, model_name, title, note_count, annot_id, weight, rank):
        result = dict()

        result['corpus'] = corpus_name
        result['model'] = model_name
        result['title'] = title
        result['notes'] = note_count
        result['ann_id'] = annot_id
        result['weight'] = weight
        result['rank'] = rank

        for tag in self.rank_methods:
            self.set_result(result=result, tag=tag, rank=rank, evil=evil)
        return result

    def full_file_path(self, base_name, name, version=None, output_dir=None, suffix='csv'):
        full_path = "{}/{}_{}.{}".format(self.output_dir, base_name, name, suffix)
        if version:
            full_path = "{}/{}_{}_{}.{}".format(output_dir, base_name, name, version, suffix)
        return full_path

    def open_file(self, base_name, name, output_dir=OUTPUT_DIR, version=None, suffix='csv', mode='w'):
        full_path = self.full_file_path(base_name, name, version, output_dir=output_dir, suffix=suffix)
        file = open(full_path, mode)
        return file
