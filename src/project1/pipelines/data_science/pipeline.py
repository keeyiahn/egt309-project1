from kedro.pipeline import node, pipeline
from .nodes import *

def create_model_pipeline(**kwargs):
    return pipeline([
            
            node(
                func=splitting_data,
                inputs="training_dataset",
                outputs=["X_train","X_test","y_train","y_test"],
            ),
            #node(
                #func=pca,
                #inputs=["X_train","X_text"],
                #outputs=["X_train_pca","X_test_pca"],
            #),
            node(
             func=model_training,
             inputs=["X_train","X_test","y_train","y_test"],
             outputs="clf",   
            ),
<<<<<<< HEAD

            node(
             func=classification,
             inputs=["output_dataset, X_train, X_test,, y_train, y_test"],
             outputs="classification_report",   
            ),

            node(
                func=prediction,
                inputs="classification_report"
                outputs=""
            ),

=======
>>>>>>> 8b74da313c79b71bfec40c37eb6ea955d20abca0
    ])