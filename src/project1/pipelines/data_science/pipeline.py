from kedro.pipeline import node, pipeline
from .nodes import *

def create_model_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=get_dataset_info,
                inputs="cleaned_dataset",
                outputs="dataframes"
            ),

        ]
    )