from kedro.pipeline import node, pipeline
from .nodes import *

def create_feature_pipeline(**kwargs):
    return pipeline([
            node(
                func=get_dataset_info,
                inputs="cleaned_dataset",
                outputs="01_cleaned_dataset",
            ),
            node(
                func=add_bulk_buy,
                inputs="01_cleaned_dataset",
                outputs="added_bulk_buyers_dataset",
            ),
            node(
                func=add_y_column,
                inputs="added_bulk_buyers_dataset",
                outputs="y_col_dataset",
            ),
            node(
                func=remove_dup_orders,
                inputs="y_col_dataset",
                outputs="unique_orders_dataset",
            ),
            node(
                func=one_hot_encode,
                inputs="unique_orders_dataset",
                outputs="ohe_dataset",

            ),
            node(
                func=dropping_columns,
                inputs="ohe_dataset",
                outputs="truncated_dataset",
            ),
            node(
                func=normalize_features,
                inputs="truncated_dataset",
                outputs="normalized_dataset",
            ),
            node(
                func=removing_non_repeat_buyers,
                inputs="normalized_dataset",
                outputs=["normalized_dataset_2", "non_repeat_dataset"],
            ),
            node(
                func=init_potential_buyers,
                inputs=["normalized_dataset_2", "non_repeat_dataset"],
                outputs=["training_dataset", "inference_dataset"]
            )
    ])