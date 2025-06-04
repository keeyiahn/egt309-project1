import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.decomposition import PCA
from collections import Counter
import joblib
#from xgboost import XGBClassifier

def get_dataset_info(df):
    print(df.info())
    return df

def add_bulk_buy(df):
    # Feature engineer 'bulk_buy' column
    df['bulk_buy'] = df.duplicated(subset=['customer_unique_id', 'order_id'], keep=False).astype(int)
    return df

def add_y_column(df):
    # prompt: create a column is_repeat_buyer such that if customer_unique_id is repeated, those rows are 1, while the ones with no repeated customer_unique_id are 0
    df['is_repeat_buyer'] = df['customer_unique_id'].duplicated(keep=False).astype(int)
    return df

def remove_dup_orders(df:pd.DataFrame) -> pd.DataFrame:
    #drop all rows that are have duplicated order_id
    df.drop_duplicates(subset=['order_id'], inplace=True)
    return df

def one_hot_encode(df):

    #one hot encode customer_city, customer_state, seller_city, seller_state, order_status, product_category_name_english
    df = pd.get_dummies(df, columns=['customer_city', 'customer_state', 'seller_city', 'seller_state', 'order_status', 'product_category_name_english', 'payment_type'], dummy_na=False)
    print(df.head())
    print(df.info())
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
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
        'customer_unique_id',
        'shipping_limit_date',
        'review_id',
        'review_creation_date',
        'review_answer_timestamp',
        'product_category_name',
        'was_delivered'
    ], axis=1, inplace=True)

    # Step 4: Drop rows with NaNs if any (from datetime conversion)
    df.dropna(inplace=True)

    return df

def normalize_features(df):
    # prompt: normalise the values in df to be between 0 and 1
    columns_to_normalize = df.select_dtypes(include=np.number).columns.tolist()

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply the scaler to the selected columns
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def removing_non_repeat_buyers(df):
    full_df = df.copy()
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    # Separate the DataFrame into repeat and non-repeat buyers
    repeat_buyers_df = df[df['is_repeat_buyer'] == 1]
    non_repeat_buyers_df = df[df['is_repeat_buyer'] == 0].copy()

    # Sort non-repeat buyers by order timestamp in descending order
    non_repeat_buyers_df = non_repeat_buyers_df.sort_values(by='order_purchase_timestamp', ascending=False)

    # Calculate the target number of non-repeat buyer rows
    target_non_repeat_count = 2*len(repeat_buyers_df)

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

    #Verify result
    print(df.info())
    return df, full_df

def init_potential_buyers(*args):
    df = args[0]
    full_df = args[1]
    # prompt: find the set of rows in full_df but not in df, add them to a new dataframe called potential_repeat
    # Identify the indices that are in full_df but not in df
    indices_to_add = full_df.index.difference(df.index)

    # Create a new dataframe called potential_repeat with these rows
    potential_repeat = full_df.loc[indices_to_add].copy()
    potential_repeat = potential_repeat[potential_repeat['is_repeat_buyer'] == 0]

    potential_repeat.drop('order_purchase_timestamp', axis=1, inplace=True)
    df.drop('order_purchase_timestamp', axis=1, inplace=True)
    return df, potential_repeat


