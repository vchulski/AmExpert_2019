import numpy as np


def age_range_to_label(x):
    if x == "18-25":
        label = 1
    elif x == "26-35":
        label = 2
    elif x == "36-45":
        label = 3
    elif x == "46-55":
        label = 4
    elif x == "56-70":
        label = 5
    elif x == "70+":
        label = 6
    else:
        label = 0

    return label


def marital_status_to_label(x):
    if x == "Married":
        label = 2
    elif x == "Single":
        label = 1
    elif x == "Unknown":
        label = 0
    else:
        label = 0

    return label


def family_size_to_label(x):
    if x == "1":
        label = 1
    elif x == "2":
        label = 2
    elif x == "3":
        label = 3
    elif x == "4":
        label = 4
    elif x == "5+":
        label = 5
    else:
        label = 0

    return label


def children_no_to_label(x):
    if x == "1":
        label = 1
    elif x == "2":
        label = 2
    elif x == "3+":
        label = 3
    elif x == "Unknown":
        label = 0
    else:
        label = 0

    return label


def preprocess_demographics_data(demographics, processing_type="custom"):
    df = demographics.copy()
    df = df.fillna({"marital_status": "Unknown", "no_of_children": "Unknown"})

    if processing_type == "custom":
        df["age_range"] = df.loc[:, "age_range"].apply(age_range_to_label)
        df["marital_status"] = df.loc[:, "marital_status"].apply(
            marital_status_to_label
        ).astype(int)
        df["family_size"] = df.loc[:, "family_size"].apply(family_size_to_label)
        df["no_of_children"] = df.loc[:, "no_of_children"].apply(children_no_to_label)
        df["my_children_per_family"] = (
            df["no_of_children"] / df["family_size"]
        ).astype(np.float64)
    elif processing_type == "ohe":
        pass
    elif processing_type == "cat":
        cat_cols = [
            "age_range",
            "marital_status",
            "rented",
            "family_size",
            "income_bracket",
        ]
        for c in cat_cols:
            df[c] = df[c].astype("category")

    return df
