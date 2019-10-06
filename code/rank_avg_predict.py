import plac
import numpy as np
import pandas as pd
from glob import glob

from scipy.stats import rankdata

LABELS = ["redemption_status"]
submission = pd.read_csv('../input/sample_submission.csv')


@plac.annotations(
    glob_files=("path to folder with csv (for example: '../for_blend/*.csv') ", "positional", None, str),
    outfile = ("path and name with ouptut csv (for example: '../submit/rank_avg_v1.csv'", "positional", None, str)
)
def main(glob_files, outfile):
    predict_list = []
    for i, glob_file in enumerate(glob(glob_files)):
        print("reading ...", i, glob_file)
        predict_list.append(pd.read_csv(glob_file)[LABELS].values)

    print("Rank averaging on ", len(predict_list), " files")
    predictions = np.zeros_like(predict_list[0])
    for predict in predict_list:
        for i in range(1):
            predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i]) / predictions.shape[0])
    predictions /= len(predict_list)


    submission[LABELS] = predictions
    submission.to_csv(outfile, index=False)
    print("Rank averaging submit is written to:  ", outfile)


if __name__ == "__main__":
    plac.call(main)