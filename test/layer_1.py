#!/usr/bin/env python3
# import pprint
import os
from datetime import datetime
# from krippendorff import alpha
# from pprint import pprint
from pydactyl.dactyler.Dactyler import Dactyler
from pydactyl.dactyler.Parncutt import Parncutt, Jacobs, Badgerow
from pydactyl.dcorpus.DCorpus import DCorpus
# from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter

TARGET_BASE = "/Users/dave/tb2/didactyl/dd/corpora/clementi/cooked/phrase/"

# paths = ["/Users/dave/tb2/didactyl/dd/corpora/clementi/cooked/{}1.abcd".format(i) for i in range(1, 2)]
interp_paths = ["/Users/dave/tb2/didactyl/dd/corpora/clementi/interp/{}1.abcd".format(i) for i in range(1, 7)]
print(interp_paths)

phrase_ids = [
    "11_1", "11_2", "11_3",  # good
    "21_1", "21_2",  # good
    # "21_3",  # bad: no solution between 17 semitone interval.
    "21_4",  # good
    "31_1", "31_2", "31_3", "31_4",  # good
    # "31_5", # bad, 17 semitone interval.
    "31_6",  # good
    "41_1", "41_2", "41_3", "41_4", "41_5", "41_6",  # good
    "41_7",  # Why is this good? Interval jump is 19!
    "51_1", "51_2", "51_3", "51_4", "51_5", "51_6",  # good
    "61_1", "61_2", "61_3", "61_4", "61_5", "61_6",  # good
]
file_extension = 'abc'
phrase_dir = "/Users/dave/tb2/didactyl/dd/corpora/clementi/raw/phrase/{}.{}"
phrase_paths = [phrase_dir.format(f, file_extension) for f in phrase_ids]


def aggregate_pairs(total, newbie):
    for key in newbie:
        if key not in total:
            total[key] = 0
        total[key] += newbie[key]


def print_results(seg_suggestions, seg_costs, seg_lengths, file_name):
    print("Advice for {}".format(file_name))
    print("==================")
    for i in range(len(seg_suggestions)):
        print("Phrase {} of length {}".format(i, seg_lengths[i]))
        for j in range(len(seg_suggestions[i])):
            sugg = Dactyler.simplify_abcdf(abcdf=seg_suggestions[i][j])
            # sugg = seg_suggestions[i][j]
            print("{}:\t{}\tCost: {}".format(j+1, sugg, seg_costs[i][j]))
        print("")
    print("\n")


def result_abcd(seg_suggestions, file_name, abc_content,
                authority="Parncutt Model (2019)"):
    now = datetime.now()
    time_stamp = now.strftime('%Y-%m-%d %H:%M:%S')

    abcd = "% abcDidactyl v6\n"
    print("Generating abcD for {} from {}".format(file_name, authority))
    for i in range(len(seg_suggestions)):
        sugg = Dactyler.simplify_abcdf(abcdf=seg_suggestions[i])
        abcd += "% abcD fingering {}: {}@\n".format(i+1, sugg)
        abcd += "% Authority: {}\n".format(authority)
        abcd += "% Transcriber: Didactyl\n"
        abcd += "% Transcription date: {}\n".format(time_stamp)
    abcd += "% abcDidactyl END\n"
    abcd += abc_content
    return abcd


def print_pairs(label, pairs):
    grand_total = 0
    last_one = None
    header = ['']
    rows = [header]
    header_done = False
    for key in sorted(pairs):
        one, other = key.split('_')
        if one != last_one:
            if last_one is not None:
                header_done = True
            last_row = [one]
            rows.append(last_row)
        last_row.append(pairs[key])
        grand_total += pairs[key]
        if not header_done:
            header.append(other)
        last_one = one
        # if pairs[key] > 0 and one != other:
            # pair_str = " {}:{}".format(key, pairs[key])
            # output_str += pair_str
    print("{} Note total: {}".format(label, grand_total))
    print("\n".join([''.join(['{:>4}'.format(item) for item in row]) for row in rows]))
    print()


d_corpus = DCorpus(paths=interp_paths)
d_scores = d_corpus.d_score_list()
editor = 2
for staff in ['upper', 'lower', 'both']:
    plural = ''
    if staff == 'both':
        plural = 's'
    score_index = 0
    interpolation_pairs = {}
    annotation_pairs = {}
    for d_score in d_scores:
        d_score.assert_consistent_abcd(staff=staff)
        kappa, pairs = d_score.cohens_kappa(1, 7, staff=staff)
        aggregate_pairs(annotation_pairs, pairs)
        print("Annotate {} staff{} Section {}.1 Kappa: {}".format(staff, plural, score_index+1, kappa))
        kappa, pairs = d_score.cohens_kappa(14, 15, staff=staff, common_id=2)
        aggregate_pairs(interpolation_pairs, pairs)
        print("Interpol {} staff{} Section {}.1 Kappa: {}".format(staff, plural, score_index+1, kappa))
        score_index += 1
    print_pairs("Annotation pairs:", annotation_pairs)
    print_pairs("Interpolation pairs:", interpolation_pairs)
exit(0)

parncutt = Parncutt()
d_corpus = DCorpus(paths=phrase_paths)
parncutt.load_corpus(d_corpus=d_corpus)
jacobs = Jacobs()
jacobs.load_corpus(d_corpus=d_corpus)
justin = Badgerow()
justin.load_corpus(d_corpus=d_corpus)
model_years = {
    'Parncutt': '1997',
    'Jacobs': '2001',
    'Badgerow': '2019'
}

models = [parncutt, jacobs, justin]
for model in models:
    model_name = str(type(model)).split('.')[-1]
    model_name = model_name[0:-2]
    model_version = model.version()
    version_str = ".".join(map(str, model_version))

    authority = model_name + ' Model (' + model_years[model_name] + ')'
    for i in range(len(phrase_ids)):
        id = phrase_ids[i]
        seg_suggestions, seg_costs, seg_details, seg_lengths = \
            model.generate_segmented_advice(score_index=i, staff="upper", k=10)
        # print_results(seg_suggestions=seg_suggestions, seg_costs=seg_costs,
        #               seg_lengths=seg_lengths, file_name=id)
        abc_content = d_corpus.abc_string_by_index(i)
        abcd = result_abcd(seg_suggestions=seg_suggestions[0], file_name=id,
                           abc_content=abc_content, authority=authority)
        target_dir = TARGET_BASE + model_name.lower() + '/' + version_str + '/'

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        target_path = target_dir + id + '.abcdf'
        print(target_path)
        fh = open(target_path, "w+")
        fh.write(abcd)
        fh.close()
        print(abcd)
        print()

# model = Parncutt(segmenter=ManualDSegmenter())
# d_corpus = DCorpus(paths=paths)
# model.load_corpus(d_corpus=d_corpus)
#
# for i in range(len(files)):
    # file = files[i]
    # seg_suggestions, seg_costs, seg_details, seg_lengths = \
        # model.generate_segmented_advice(score_index=i, staff="upper", k=10)
    # print_results(seg_suggestions=seg_suggestions, seg_costs=seg_costs, seg_lengths=seg_lengths, file_name=file)
# print("Ranked advice:\n\t{0}".format("\n\t".join(suggestions)))
# print("Ranked costs :\n\t{0}".format("\n\t".join(str(x) for x in costs)))

# pp = pprint.PrettyPrinter(width=120)
# pp.pprint(details)
