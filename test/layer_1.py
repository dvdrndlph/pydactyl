#!/usr/bin/env python3
import pprint
from pydactyl.dactyler.Parncutt import Parncutt
from pydactyl.dcorpus.DCorpus import DCorpus
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter

model = Parncutt(segmenter=ManualDSegmenter(),
                 segment_combiner="cost")
# d_corpus = DCorpus(paths=["/Users/dave/malody.abcd"])
d_corpus = DCorpus(paths=["/tmp/malody.abcd"])
model.load_corpus(d_corpus=d_corpus)
advice = model.advise()
print("Best advice: {0}".format(advice))
# Gold-standard embedded in input file.
hamming_dists = model.evaluate_strike_distance()
print("Hamming distance from gold standard: {0}".format(hamming_dists[0]))



seg_suggestions, seg_costs, seg_details, seg_lengths = \
    model.generate_segmented_advice(score_index=0, staff="upper", k=10)

print("Ranked advice:\n\t{0}".format("\n\t".join(suggestions)))
print("Ranked costs :\n\t{0}".format("\n\t".join(str(x) for x in costs)))

# pp = pprint.PrettyPrinter(width=120)
# pp.pprint(details)
