from kedro.pipeline import node, pipeline
from .nodes import *

def create_feature_pipeline(**kwargs):
    return pipeline([
            node(
                func=get_dataset_info,
                inputs="cleaned_dataset",
                outputs="01_cleaned_dataset"
            ),
            node(
                func=remove_dup_orders,
                inputs="01_cleaned_dataset",
                outputs="unique_orders_dataset"
            ),
            node(
                func=add_y_column,
                inputs="unique_orders_dataset",
                outputs="y_col_dataset"

            ),
            node(
                func=one_hot_encode,
                inputs="y_col_dataset",
                outputs="ohe_dataset"

            ),
            node(
                func=removing_non_repeat_buyers,
                inputs="ohe_dataset",
                outputs="truncated_dataset"
            ),
            node(
                func=dropping_columns,
                inputs="truncated_dataset",
                outputs="output_dataset"
            )


    ])