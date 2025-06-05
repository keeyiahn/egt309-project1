from kedro.pipeline import node, pipeline
from .nodes import *

def create_reporting_pipeline(**kwargs):
    return pipeline([
        node(
            func=impt_features,
            inputs=["clf","X_train"],
            outputs=None,
        ),
        node(
            func=prediction,
            inputs=["clf", "inference_dataset", "X_train"],
            outputs="predicted_repeat_df",
        ),
    ])
