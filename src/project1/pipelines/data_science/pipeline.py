from kedro.pipeline import node, pipeline
from .nodes import *

def create_model_pipeline(**kwargs):
    return pipeline([
            
            node(
                func=splitting_data,
                inputs="output_dataset",
                outputs=[""],
            ),

            node(
                func=reduce_columns,
                inputs="",
                outputs="",
            ),

            node(
             func=training_model,
             inputs="",
             outputs="",   
            )

    ])