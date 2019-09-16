from nltk.metrics.agreement import AnnotationTask
from krippendorff import alpha
import numpy as np

# The following example borrowed via the krippendorf modules doctest from
# https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2016/07/fulltext.pdf, page 8.
reliability_data = [[1,      2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
                    [1,      2, 3, 3, 2, 2, 4, 1, 2, 5,      np.nan, 3],
                    [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5,      1,      np.nan],
                    [1,      2, 3, 3, 2, 4, 4, 1, 2, 5,      1,      np.nan]]
print(round(alpha(reliability_data, level_of_measurement='nominal'), 3))
# 0.743

annotation_data = []
coder_id = 0
for r in reliability_data:
    item_id = 0
    for v in r:
        if not np.isnan(v):
            record = [coder_id, item_id, v]
            annotation_data.append(record)
        item_id += 1
    coder_id += 1
print(annotation_data)

annot_task = AnnotationTask(data=annotation_data)
print(round(annot_task.alpha(), 3))
# 0.705
