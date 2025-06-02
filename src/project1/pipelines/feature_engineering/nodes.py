import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.decomposition import PCA
from collections import Counter
import joblib
from xgboost import XGBClassifier


def get_dataset_info(*dataframes):

    for df in dataframes:
        print(df.info())
    return dataframes

def preparing_dataset(df:pd.DataFrame) -> pd.DataFrame:

    #create a column is_repeat_buyer such that if customer_unique_id is repeated, those rows are 1, while the ones with no repeated customer_unique_id are 0
    df['order_approved_at'] = pd.to_datetime(df['order_approved_at'], dayfirst=True, errors='coerce')
    #drop all rows that are have duplicated order_id
    df.drop_duplicates(subset=['order_id'], inplace=True)

    
    return df

def one_hot_encode(df):

    #one hot encode customer_city, customer_state, seller_city, seller_state, order_status, product_category_name_english
    df = pd.get_dummies(df, columns=['customer_city', 'customer_state', 'seller_city', 'seller_state', 'order_status', 'product_category_name_english', 'payment_type'], dummy_na=False)
    print(df.head())
    print(df.info())
    return df

def splitting_buyers(df):

    # For all the rows with is_repeat_buyer == 0, keep removing rows with most recent order_purchase_timestamp until number of rows with is_repeat_buyer==0 is 2 times number of rows with is_repeat_buyer==1
    # Also assuming 'order_purchase_timestamp' column exists and is in a sortable format (e.g., datetime objects)
    # Ensure 'order_purchase_timestamp' is in datetime format
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    # Separate the DataFrame into repeat and non-repeat buyers
    repeat_buyers_df = df[df['is_repeat_buyer'] == 1]
    non_repeat_buyers_df = df[df['is_repeat_buyer'] == 0].copy()

    # Sort non-repeat buyers by order timestamp in descending order
    non_repeat_buyers_df = non_repeat_buyers_df.sort_values(by='order_purchase_timestamp', ascending=False)

    # Calculate the target number of non-repeat buyer rows
    target_non_repeat_count = 3 * len(repeat_buyers_df)

    # While the number of non-repeat buyer rows is greater than the target,
    # remove the row with the most recent order timestamp
    while len(non_repeat_buyers_df) > target_non_repeat_count:
        # Drop the first row (which has the most recent timestamp after sorting)
        non_repeat_buyers_df = non_repeat_buyers_df.iloc[1:]

    # Concatenate the repeat and non-repeat buyer dataframes
    df = pd.concat([repeat_buyers_df, non_repeat_buyers_df])

    print("DataFrame after balancing non-repeat buyers:")
    print(df.info())
    print("Number of repeat buyers:", len(repeat_buyers_df))
    print("Number of non-repeat buyers:", len(non_repeat_buyers_df))
    return df

def dropping_columns(df):

    #Creating delivery_days BEFORE dropping datetime columns
    df['delivery_days'] = (
    pd.to_datetime(df['order_delivered_customer_date'], errors='coerce') -
    pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
    ).dt.days

    #Droping irrelevant columns AFTER creating new features
    df.drop([
        'order_id',
        'customer_id',
        'product_id',
        'seller_id',
        'order_purchase_timestamp',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
        'customer_unique_id',
        'shipping_limit_date',
        'review_id',
        'review_creation_date',
        'review_answer_timestamp',
        'product_category_name'
    ], axis=1, inplace=True)


    #Dropping rows with NaNs if any (from datetime conversion)
    df.dropna(inplace=True)

    #Verify result
    print(df.info())
    return df
