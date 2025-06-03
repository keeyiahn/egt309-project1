from kedro.pipeline import node, pipeline
from .nodes import *

def create_model_pipeline(**kwargs):
    return pipeline([
            node(
                func=get_dataset_info,
                inputs=["cleaned_dataset"],
                outputs="dataframe"
            ),
            node(
                func=preparing_dataset,
                inputs=["cleaned_dataset"],
                outputs="dataframe"

            ),
            node(
                func=one_hot_encode,
                inputs=["cleaned_dataset"],
                outputs="dataframe"

            ),
            node(
                func=splitting_buyers,
                inputs=["cleaned_dataset"],
                outputs="dataframe"
            ),
            node(
                func=dropping_columns,
                inputs=["cleaned_dataset"],
                outputs="dataframe"
            )


    ])