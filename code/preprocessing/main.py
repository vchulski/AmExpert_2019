__author__ = "Vadim_Shestopalov"
__email__ = "vdmshestopalov@gmail.com"

# TO DO:
# add OHE transformration
# add proper verbose


import os
import plac
import numpy as np
import pandas as pd

from sklearn import preprocessing
from demographics import preprocess_demographics_data

PATH_TO_DATA = "/home/vchulski/work/AV/AmExpert_2019/input"

# Merge data for train
train = pd.read_csv(os.path.join(PATH_TO_DATA, "train.csv"))
campaign_data = pd.read_csv(
    os.path.join(PATH_TO_DATA, "campaign_data.csv")
)  # key: "campaign_id"
customer_demographics = pd.read_csv(
    os.path.join(PATH_TO_DATA, "customer_demographics.csv")
)  # "customer_id"
coupon_item_mapping = pd.read_csv(
    os.path.join(PATH_TO_DATA, "coupon_item_mapping.csv")
)  # "coupon_id"
customer_transaction_data = pd.read_csv(
    os.path.join(PATH_TO_DATA, "customer_transaction_data.csv")
)  # "customer_id"
item_data = pd.read_csv(
    os.path.join(PATH_TO_DATA, "item_data.csv")
)  # with coupon mapping by: "item_id"
test = pd.read_csv(os.path.join(PATH_TO_DATA, "test.csv"))


@plac.annotations(
    output_name=("Name of the output: ", "positional", None, str),
    processing_type=("Name of processing type (custom/ohe/cat) ", "option", "p", str),
)
def main(output_name, processing_type):
    print("Preparing train dataframe ... ")
    transformed_train = prepare_full_df(
        train,
        campaign_data,
        customer_demographics,
        customer_transaction_data,
        coupon_item_mapping,
        item_data,
        processing_type,
    )

    transformed_train.to_csv(os.path.join(PATH_TO_DATA, "train_"+output_name + ".csv"))
    print(f"Transformed train saved to: {str(os.path.join(PATH_TO_DATA, 'train_'+output_name+'.csv'))}")

    print("Preparing test dataframe ... ")
    transformed_test = prepare_full_df(
        test,
        campaign_data,
        customer_demographics,
        customer_transaction_data,
        coupon_item_mapping,
        item_data,
        processing_type,
    )
    transformed_test.to_csv(os.path.join(PATH_TO_DATA, "test_"+output_name + ".csv"))
    print(f"Transformed train saved to: {str(os.path.join(PATH_TO_DATA, 'test' + output_name + '.csv'))}")


def preprocess_campaign_data(campaign, for_eda=False):
    df = campaign.copy()

    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df.campaign_type.values))
    df["campaign_type"] = lbl.transform(list(df["campaign_type"].values))

    df["start_date"] = pd.to_datetime(df["start_date"], format="%d/%m/%y")
    df["end_date"] = pd.to_datetime(df["end_date"], format="%d/%m/%y")

    if for_eda:
        df["start_day"] = df["start_date"].apply(lambda ts: ts.day).astype(np.int32)
        df["start_yyyymm"] = (
            df["start_date"].apply(lambda ts: 100 * ts.year + ts.month).astype(np.int32)
        )  # wtf! why this works?
        df["start_dow"] = (
            df["start_date"].apply(lambda ts: ts.dayofweek).astype(np.int8)
        )
        df["end_day"] = df["end_date"].apply(lambda ts: ts.day).astype(np.int32)
        df["end_yyyymm"] = (
            df["end_date"].apply(lambda ts: 100 * ts.year + ts.month).astype(np.int32)
        )  # wtf! why this works?
        df["end_dow"] = df["end_date"].apply(lambda ts: ts.dayofweek).astype(np.int8)

    df["campaign_duration"] = (
        ((df["end_date"] - df["start_date"]) / np.timedelta64(1, "m")).round(0)
        / (60 * 24)
    ).astype(int)

    if not for_eda:
        df = df.drop(["start_date", "end_date"], axis=1)

    return df


def preprocess_item_data(item, processing_type="custom"):
    df = item.copy()
    cols = ["brand_type", "category"]

    if processing_type == "custom":
        for f in cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))
    elif processing_type == "ohe":  # TO DO: add this type
        pass
    elif processing_type == "cat":
        for c in cols:
            df[c] = df[c].astype("category")

    return df


def preprocess_customer_transaction(customer_transaction):
    df = customer_transaction.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    df["transaction_day"] = df["date"].apply(lambda ts: ts.day).astype(np.int32)
    df["transaction_yyyymm"] = (
        df["date"].apply(lambda ts: 100 * ts.year + ts.month).astype(np.int32)
    )  # wtf! why this works?
    df["transaction_dow"] = df["date"].apply(lambda ts: ts.dayofweek).astype(np.int8)

    df = df.drop(["date"], axis=1)

    return df


def prepare_full_df(
    dataframe,
    campaign,
    demographics,
    customer_transaction,
    coupon_item_mapping,
    item_data,
    processing_type="custom",
    # for_eda=False,
    # verbose=0
):
    df = dataframe.copy()

    assert len(df) == len(dataframe)

    campaign_data = preprocess_campaign_data(campaign)
    df = pd.merge(df, campaign_data, on="campaign_id", how="left")

    assert len(df) == len(dataframe)

    customer_demographics_data = preprocess_demographics_data(
        demographics, processing_type
    )
    df = pd.merge(df, customer_demographics_data, on="customer_id", how="left")

    assert len(df) == len(dataframe)

    item_data = preprocess_item_data(item_data, processing_type)

    coupon_mapping_item_merged = pd.merge(
        coupon_item_mapping, item_data, on="item_id", how="left"
    )
    coupon_item = (
        coupon_mapping_item_merged.groupby("coupon_id")
        .nunique()
        .drop("coupon_id", axis=1)
    )

    df = pd.merge(df, coupon_item, on="coupon_id", how="left")

    assert len(df) == len(dataframe)

    customer_transaction_data = preprocess_customer_transaction(customer_transaction)
    aggregations = {
        "selling_price": ["mean", "std", "min", "max"],
        "other_discount": ["mean", "std", "min", "max"],
        "coupon_discount": ["mean", "std", "min", "max"],
    }
    transacation_customer_aggs = customer_transaction_data.groupby("customer_id").agg(
        aggregations
    )
    transacation_customer_aggs.columns = list(
        map("_".join, transacation_customer_aggs.columns.values)
    )

    df = pd.merge(df, transacation_customer_aggs, on="customer_id", how="left")

    assert len(df) == len(dataframe)

    transacation_items_aggs = customer_transaction_data.groupby("item_id").agg(
        aggregations
    )
    transacation_items_aggs.columns = list(
        map("_".join, transacation_items_aggs.columns.values)
    )

    coupon_merged_with_transaction_stats = pd.merge(
        coupon_item_mapping, transacation_items_aggs, on="item_id", how="left"
    )
    coupon_merged_with_transaction_stats_grouped = coupon_merged_with_transaction_stats.groupby(
        "coupon_id"
    ).mean()

    df = pd.merge(
        df, coupon_merged_with_transaction_stats_grouped, on="coupon_id", how="left"
    )

    assert len(df) == len(dataframe)

    return df


if __name__ == "__main__":
    plac.call(main)
