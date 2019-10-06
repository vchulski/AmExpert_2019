import plac
import os
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn import model_selection

import lightgbm
import catboost

PATH_TO_DATA = "../input/"

@plac.annotations(
    train_name=("name of train file in input folder: ", "positional", None, str),
    test_name=("name of test file in input folder: ", "positional", None, str),
    model_name=("name of model to use (lgb/cat): ", "option", "m", str),
    folds_num=("number of folds to use: ", "option", "f", int),
    predict=("make predict on test: ", "option", "p", int),
)
def main(train_name, test_name, model_name, folds_num, predict):
    print()
    TARGET = "redemption_status"
    COLS_TO_DROP = ["id"]
    train = pd.read_csv(os.path.join(PATH_TO_DATA, train_name + ".csv"))
    test = pd.read_csv(os.path.join(PATH_TO_DATA, test_name + ".csv"))
    ss = pd.read_csv(os.path.join(PATH_TO_DATA, "sample_submission.csv"))

    X = train.drop(TARGET, axis=1)
    X = X.drop(COLS_TO_DROP, axis=1)
    y = train[[TARGET]].values.ravel()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    if model_name is None:
        model_name = "lgb"

    if model_name == "lgb":
        clf = lightgbm.LGBMClassifier()
    elif model_name == "cat":
        categ_feat_idx = np.where(X.dtypes == 'category')[0]
        clf = catboost.CatBoostClassifier(verbose=0, cat_features=categ_feat_idx)
    else:
        print("Unknown model specified")
        return

    clf.fit(X_train, y_train)
    y_test_pred = clf.predict_proba(X_test)[:, 1]
    holdout_score = round(metrics.roc_auc_score(y_test, y_test_pred), 3)
    print(model_name + ":")
    print("-" * 44)
    print(f"Holdout score: {holdout_score}")

    if folds_num is None:
        folds_num = 3

    cv_scores = model_selection.cross_val_score(
        clf,
        X,
        y,
        cv=model_selection.StratifiedKFold(folds_num, shuffle=True, random_state=42),
        scoring="roc_auc",
        verbose=0,
    )

    print(
        f"CV {folds_num} StratifiedFold score: {round(cv_scores.mean(), 3)} | std: {round(cv_scores.std(), 3)}"
    )

    if predict is None:
        predict = 0

    if predict == 1:
        clf.fit(X, y)
        test = test.drop(COLS_TO_DROP, axis=1)
        test_preds = clf.predict_proba(test)[:, 1]
        ss[TARGET] = test_preds
        filename = f"../submit/baseline_{model_name}_{holdout_score}_holdout_{round(cv_scores.mean(), 3)}_CV.csv"
        ss.to_csv(filename, index=False)
        print(f"Submission saved to: ", filename)
    elif predict == 0:
        pass


if __name__ == "__main__":
    plac.call(main)
