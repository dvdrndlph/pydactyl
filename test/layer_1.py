#!/usr/bin/env python3
# import pprint
from pydactyl.dactyler.Dactyler import Dactyler
from pydactyl.dactyler.Parncutt import Parncutt
from pydactyl.dcorpus.DCorpus import DCorpus
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter


files = [
    "11.abcd",
#     "21.abcd",
#     "31.abcd",
    "41.abcd",
#     "51.abcd",
    "61.abcd"]
files = ["11.abcd"]
paths = ["/Users/dave/tb2/didactyl/dd/corpora/clementi/cooked/{}".format(f) for f in files]


def print_results(seg_suggestions, seg_costs, seg_lengths, file_name):
    print("\nAdvice for {}".format(file_name))
    print("==================")
    for i in range(len(seg_suggestions)):
        print("Phrase {} of length {}".format(i, seg_lengths[i]))
        for j in range(len(seg_suggestions[i])):
            sugg = Dactyler.simplify_abcdf(abcdf=seg_suggestions[i][j])
            # sugg = seg_suggestions[i][j]
            print("{}:\t{}\tCost: {}".format(j+1, sugg, seg_costs[i][j]))
        print("")
    print("\n")


model = Parncutt(segmenter=ManualDSegmenter())
d_corpus = DCorpus(paths=paths)
model.load_corpus(d_corpus=d_corpus)

for i in range(len(files)):
    file = files[i]
    seg_suggestions, seg_costs, seg_details, seg_lengths = \
        model.generate_segmented_advice(score_index=i, staff="upper", k=10)
    print_results(seg_suggestions=seg_suggestions, seg_costs=seg_costs, seg_lengths=seg_lengths, file_name=file)

# print("Ranked advice:\n\t{0}".format("\n\t".join(suggestions)))
# print("Ranked costs :\n\t{0}".format("\n\t".join(str(x) for x in costs)))

# pp = pprint.PrettyPrinter(width=120)
# pp.pprint(details)
