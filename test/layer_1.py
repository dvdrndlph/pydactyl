#!/usr/bin/env python3
# import pprint
import os
from datetime import datetime
from pydactyl.dactyler.Dactyler import Dactyler
from pydactyl.dactyler.Parncutt import Parncutt, Jacobs, Badgerow
from pydactyl.dcorpus.DCorpus import DCorpus
# from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter
# from pprint import pprint

TARGET_BASE = "/Users/dave/tb2/didactyl/dd/corpora/clementi/cooked/phrase/"

# files = [
#     "11.abcd",
#     "21.abcd",
#     "31.abcd",
#     "41.abcd",
#     "51.abcd",
#     "61.abcd"]
# files = ["21.abcd"]
# paths = ["/Users/dave/tb2/didactyl/dd/corpora/clementi/cooked/{}".format(f) for f in files]

phrase_ids = [
    "11_1", "11_2", "11_3",  # good
    "21_1", "21_2",  # good
    # "21_3",  # bad: no solution between 17 semitone interval.
    "21_4",  # good
    "31_1", "31_2", "31_3", "31_4",  # good
    # "31_5", # bad, 17 semitone interval.
    "31_6",  # good
    "41_1", "41_2", "41_3", "41_4", "41_5", "41_6",  # good
    "41_7",  # Bad for Jacobs
    "51_1", "51_2", "51_3", "51_4", "51_5", "51_6",  # good
    "61_1", "61_2", "61_3", "61_4", "61_5", "61_6",  # good
]
file_extension = 'abc'
phrase_dir = "/Users/dave/tb2/didactyl/dd/corpora/clementi/raw/phrase/{}.{}"
phrase_paths = [phrase_dir.format(f, file_extension) for f in phrase_ids]


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
        abcd += "% Transcriber: Dydactyl\n"
        abcd += "% Transcription date: {}\n".format(time_stamp)
    abcd += "% abcDidactyl END\n"
    abcd += abc_content
    return abcd


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
models = [justin]
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
