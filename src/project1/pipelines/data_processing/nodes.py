import pandas as pd
from pyspark.sql import Column
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import DoubleType


def get_dataset_info(*dataframes):
    for df in dataframes:
        print(df.info())
    return dataframes

def merge_datasets(customers, orders, order_items, order_payments, order_reviews, products, sellers, product_cats):
    merged_dataframe = customers.merge(orders, on="customer_id", how="left") \
        .merge(order_items, on="order_id", how="left") \
        .merge(order_payments, on="order_id", how="left") \
        .merge(order_reviews, on="order_id", how="left") \
        .merge(products, on="product_id", how="left") \
        .merge(sellers, on="seller_id", how="left") \
        .merge(product_cats, on="product_category_name", how="left")
    
    print("---- BEFORE CLEANING ----")
    print(merged_dataframe.shape)
    print(merged_dataframe.columns)
    print(f"Duplicated rows: {merged_dataframe.duplicated().sum()}")
    print(f"Missing values:\n{merged_dataframe.isna().sum()}")
    print("hi"+str(merged_dataframe.isnull().sum()))

    return merged_dataframe

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Remove leading and trailing spaces from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Convert review timestamps if not already
    df['review_answer_timestamp'] = pd.to_datetime(df['review_answer_timestamp'], errors='coerce')

    # Fill missing order_delivered_customer_date where review exists
    df['order_delivered_customer_date'] = df['order_delivered_customer_date'].fillna(
        df['review_answer_timestamp']
    )

    # --- Convert timestamps to datetime format (if not already) ---
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors='coerce')

    # --- Create delivery duration in days ---
    df['delivery_time_days'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days

    # --- Flag if the order was delivered ---
    df['was_delivered'] = df['order_delivered_customer_date'].notnull().astype(int)

    # --- Drop timestamps not needed for customer-centric analysis ---
    df.drop(columns=[
        'order_approved_at',                # Internal event
        'order_delivered_carrier_date'     # Operational event
    ], inplace=True)
    
    df.drop(columns=['review_comment_title', 'review_comment_message'], inplace=True)

    # List of incomplete order columns (all must be NaN to be dropped)
    cols_to_check = [
        'order_item_id',
        'product_id',
        'seller_id',
        'shipping_limit_date',
        'price',
        'freight_value',
        'seller_zip_code_prefix',
        'seller_city',
        'seller_state'
    ]

    # Drop rows where ALL of these columns are missing
    df = df[~df[cols_to_check].isnull().all(axis=1)]

    # Define review-related columns
    review_cols = [
        'review_id',
        'review_score',
        'review_creation_date',
        'review_answer_timestamp'
    ]

    # Drop rows where ALL review-related columns are missing
    df = df[~df[review_cols].isnull().all(axis=1)]

    # Identify rows missing all key product-related fields
    product_missing_rows = df[
        df[['product_category_name', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty']]
        .isnull()
        .all(axis=1)
    ]

    # Drop them from merged_df
    df = df.drop(index=product_missing_rows.index)

    # Convert 'freight_value' to numeric, handling errors
    df['freight_value'] = pd.to_numeric(df['freight_value'], errors='coerce')

    # Define the columns to impute
    dim_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']

    # Group by product_category_name_english and compute the mean for each dimension
    category_means = df.groupby('product_category_name_english')[dim_cols].mean()

    # Function to fill missing values using category mean
    def impute_with_category_mean(row):
        if pd.isnull(row['product_weight_g']):
            row['product_weight_g'] = category_means.loc[row['product_category_name_english'], 'product_weight_g']
        if pd.isnull(row['product_length_cm']):
            row['product_length_cm'] = category_means.loc[row['product_category_name_english'], 'product_length_cm']
        if pd.isnull(row['product_height_cm']):
            row['product_height_cm'] = category_means.loc[row['product_category_name_english'], 'product_height_cm']
        if pd.isnull(row['product_width_cm']):
            row['product_width_cm'] = category_means.loc[row['product_category_name_english'], 'product_width_cm']
        return row

    # Apply the imputation row-wise
    df = df.apply(impute_with_category_mean, axis=1)
    df.dropna(inplace=True)

    print("---- AFTER CLEANING ----")
    print(df.shape)
    print(df.columns)
    print(f"Duplicated rows: {df.duplicated().sum()}")
    print(f"Missing values:\n{df.isna().sum()}")
    
    return df


def transform_dataset(df):
    # Drop irrelevant or redundant columns for repeat buyer prediction
    cols_to_drop = [
        'order_id',                     # Unique ID, not useful for modeling
        'customer_id',                  # Redundant with customer_unique_id
        'review_id',                    # Unique per review, not predictive
        'product_id',                   # Too specific, not useful unless modeling per product
        'seller_id',                    # High cardinality, low interpretability
        'shipping_limit_date',         # Redundant with delivery_time_days
        'review_creation_date',        # Already captured by review_score/sentiment
        'order_item_id',               # Row-level ordering, not predictive
        'product_category_name',       # Redundant with English translation
        'seller_zip_code_prefix',      # Too granular
    ]

    # Drop the columns
    df.drop(columns=cols_to_drop, inplace=True)

    # Confirm removal
    print(f"Remaining columns: {df.columns.tolist()}")

    # Count number of orders per customer
    customer_order_counts = df['customer_unique_id'].value_counts().reset_index()
    customer_order_counts.columns = ['customer_unique_id', 'order_count']

    # Label as repeat (1) or non-repeat (0) buyer
    customer_order_counts['is_repeat_buyer'] = customer_order_counts['order_count'].apply(lambda x: 1 if x > 1 else 0)

    # Display summary
    print(customer_order_counts['is_repeat_buyer'].value_counts())
    customer_order_counts.head()

    return df

