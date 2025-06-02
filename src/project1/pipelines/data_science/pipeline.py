from kedro.pipeline import node, pipeline
from .nodes import *

def create_model_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func= ,
                inputs= ,
                outputs= ,
                name= ,
            ),
            
        ]
    )