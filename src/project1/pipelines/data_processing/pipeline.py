from kedro.pipeline import node, pipeline
from .nodes import *

def create_data_pipeline(**kwargs):
    return pipeline([
        node(
            func=get_dataset_info,
            inputs=[
                "customers_data",
                "geolocation",
                "order_items",
                "order_payments",
                "order_reviews",
                "orders",
                "products",
                "sellers",
                "product_cats"
            ],
            outputs="dataframes"
        ),
        node(
            func=merge_datasets,
            inputs=["customers_data", 
                    "orders", 
                    "order_items", 
                    "order_payments", 
                    "order_reviews", 
                    "products", 
                    "sellers", 
                    "product_cats"],
            outputs="merged_dataset"
        ),
        node(
            func=clean_dataset,
            inputs="merged_dataset",
            outputs="cleaned_dataset"
        ),
        node(
            func=transform_dataset,
            inputs="cleaned_dataset",
            outputs="transformed_dataset"
        )
    ])