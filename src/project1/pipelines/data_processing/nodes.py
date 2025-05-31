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
    
    # Convert 'freight_value' to numeric, handling errors
    df['freight_value'] = pd.to_numeric(df['freight_value'], errors='coerce')
    
    # Fill NaN values with 0
    df.fillna(0, inplace=True)

    print("---- AFTER CLEANING ----")
    print(df.shape)
    print(df.columns)
    print(f"Duplicated rows: {df.duplicated().sum()}")
    print(f"Missing values:\n{df.isna().sum()}")
    print("hi"+str(df.isnull().sum()))
    
    return df


