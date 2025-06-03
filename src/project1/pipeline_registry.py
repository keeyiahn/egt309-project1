"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_processing import create_data_pipeline as data_processing_pipeline
from .pipelines.feature_engineering import create_model_pipeline as feature_engineering_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        "__default__": data_processing_pipeline() + feature_engineering_pipeline(),
        "data_processing": data_processing_pipeline(),
        "feature_engineering": feature_engineering_pipeline(),
    }
