"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_processing import create_data_pipeline as data_processing_pipeline
from .pipelines.feature_engineering import create_feature_pipeline as feature_engineering_pipeline
from .pipelines.data_science import create_model_pipeline as model_pipeline
from .pipelines.reporting import create_reporting_pipeline as reporting_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        "__default__": data_processing_pipeline() + feature_engineering_pipeline() + model_pipeline() + reporting_pipeline(),
        "data_processing": data_processing_pipeline(),
        "feature_engineering": feature_engineering_pipeline(),
        "data_science": model_pipeline(),
        "reporting": reporting_pipeline(),
    }
