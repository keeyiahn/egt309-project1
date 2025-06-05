import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import logging
# ML libraries
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import learning_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

def impt_features(clf, X_train):
    # Get feature importances
    feature_importances = clf.feature_importances_

    # Create a mapping of feature names to importances
    feature_importance_dict = dict(zip(X_train.columns, feature_importances))

    # Sort features by importance in descending order
    sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

    # Print the top 20 features and their importances
    print("\nTop 20 Features by Importance:")
    for feature, importance in sorted_features[:20]:
        print(f"{feature}: {importance:.4f}")
    return 

def prediction(clf, potential_repeat, X_train):
    # Ensure potential_repeat has the same columns in the same order as X_train
    potential_repeat_aligned = potential_repeat[X_train.columns]

    # Test the classifier on potential_repeat
    potential_repeat_predictions = clf.predict(potential_repeat_aligned)

    # Count the number of predictions for each class
    prediction_counts = pd.Series(potential_repeat_predictions).value_counts()

    print("\nNumber of predictions on potential_repeat:")
    print(f"Class 0 (Not Repeat Buyer): {prediction_counts.get(0, 0)}")
    print(f"Class 1 (Predicted Repeat Buyer): {prediction_counts.get(1, 0)}")

    # Create a DataFrame with rows from potential_repeat where is_repeat_buyer is predicted as 1
    predicted_repeat_df = potential_repeat[potential_repeat_predictions == 1].copy()

    print("\nDataFrame with potential repeat buyers (predicted is_repeat_buyer == 1):")
    predicted_repeat_df.head()
    return predicted_repeat_df


