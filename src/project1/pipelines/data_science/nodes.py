import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.decomposition import PCA
from collections import Counter
import joblib
#from xgboost import XGBClassifier

def get_dataset_info(*dataframes):

    for df in dataframes:
        print(df.info())
    return dataframes


def splitting_data(df):
    # We use stratifiedshufflesplit instead of train_test_split to handle class imbalances
    # Features and target
    X = df.drop('is_repeat_buyer', axis=1)
    y = df['is_repeat_buyer']

    # Split into train/test using StratifiedShuffleSplit
    # This preserves the percentage of samples for each class
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in strat_split.split(X, y):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]


    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # Verify the class distribution in train and test sets
    print("\nClass distribution in original data:")
    print(y.value_counts(normalize=True))

    print("\nClass distribution in training set:")
    print(y_train.value_counts(normalize=True))

    print("\nClass distribution in test set:")
    print(y_test.value_counts(normalize=True))
    return df

def reduce_columns(X_train, X_test):
    # Use principal component analysis to reduce columns to 70 components
    # Initialize PCA with the desired number of components
    n_components = 70
    pca = PCA(n_components=n_components)

    # Fit PCA on the training data and transform both train and test sets
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("Shape after PCA on training data:", X_train_pca.shape)
    print("Shape after PCA on testing data:", X_test_pca.shape)

    # Optionally, you can check the explained variance ratio
    print("\nExplained variance ratio by each component:")
    print(pca.explained_variance_ratio_)

    print("\nTotal explained variance by 70 components:")
    print(np.sum(pca.explained_variance_ratio_))

    # Now you can use X_train_pca and X_test_pca for training your models
    # For example, training a RandomForestClassifier
    # rf_model = RandomForestClassifier(random_state=42)
    # rf_model.fit(X_train_pca, y_train)
    # predictions = rf_model.predict(X_test_pca)
    # print(classification_report(y_test, predictions))
    return X_train, X_test

def training_model(df, X_train, X_test):
    # Prepare features and target
    X = df.drop('is_repeat_buyer', axis=1)
    y = df['is_repeat_buyer']


    # Train-test split (stratify to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Calculate scale_pos_weight
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # Train XGBoost
    # Removed use_label_encoder as it's deprecated
    clf = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    clf.fit(X_train, y_train)

    # Step 5: Accuracy scores
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {test_acc:.4f}")

    # Step 6: Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))


