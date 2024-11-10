import streamlit as st
import pandas as pd
import numpy as np
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import (
    get_model, REGRESSION_MODELS, CLASSIFICATION_MODELS
)
from autoop.core.ml.metric import (
    get_metric, REGRESSION_METRICS, CLASSIFICATION_METRICS
)
from autoop.core.ml.pipeline import Pipeline

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """Method used for rendering the subtitle"""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def select_dataset() -> Dataset:
    """
    Function used to render a selectbox for user to choose a dataset for
    the pipeline.

    Returns:
        Dataset: The selected dataset object.
    """
    automl = AutoMLSystem.get_instance()
    datasets = automl.registry.list(type="dataset")

    if not datasets:
        st.error("No datasets found. Please upload a dataset first.")
        st.stop()

    options = [f"{d.name} (v{d.version})" for d in datasets]
    mapping = {option: dataset for option, dataset in zip(options, datasets)}

    selected_option = st.selectbox("Select a dataset", options)
    selected_artifact = mapping[selected_option]

    selected_dataset = Dataset(
        name=selected_artifact.name,
        version=selected_artifact.version,
        asset_path=selected_artifact.asset_path,
        metadata=selected_artifact.metadata,
        tags=selected_artifact.tags,
        data=selected_artifact.data,
    )

    return selected_dataset


def select_features(dataset: Dataset) -> tuple:
    """
    A function used to render the feature selection section of the page
    for the pipeline.

    Args:
        dataset (Dataset): The dataset object used for training.

    Returns:
        tuple: The selected input features, target feature, task type and
        a mapping of feature names to their objects.
    """
    features = detect_feature_types(dataset)
    map_features = {feature.name: feature for feature in features}
    feature_names = [feature.name for feature in features]

    st.subheader("Select Features")

    input_features = st.multiselect("Select input features", feature_names)
    target_feature = st.selectbox("Select target feature", feature_names)

    if not input_features:
        st.error("Please select at least one input feature.")
        st.stop()
    if not target_feature:
        st.error("Please select a target feature.")
        st.stop()

    target_feature_selected = map_features[target_feature]
    if target_feature_selected.type == 'categorical':
        task_type = "classification"
    elif target_feature_selected.type == 'numerical':
        task_type = "regression"
    else:
        task_type = "unknown"
        st.error("Target feature type is unknown. Please select a "
                 "valid target feature.")
        return None, None, None, None

    st.write(f"Task type: **{task_type}**")
    return input_features, target_feature, task_type, map_features


def select_model_metrics(type: str) -> tuple:
    """
    Method used for rendering all content of page that is related to selecting
    the model and metrics for the pipeline.

    Args:
        type (str): The task type selected by user.

    Returns:
        tuple: The selected model name and metrics, in a list of strings.
    """
    if type == "classification":
        models = CLASSIFICATION_MODELS
        metric = CLASSIFICATION_METRICS
    elif type == "regression":
        models = REGRESSION_MODELS
        metric = REGRESSION_METRICS
    else:
        st.error("Task type is unknown. Please select a valid target feature.")
        return None, None

    model_name = st.selectbox("Select a model", models)
    metrics = st.multiselect("Select metrics", metric)

    if not model_name:
        st.error("Please select a model.")
        return None, None
    if not metrics:
        st.error("Please select at least one metric.")
        return None, None

    return model_name, metrics


def pipeline_summary(dataset: Dataset,
                     features: list,
                     target_feature: str,
                     task_type: str,
                     model_name: str,
                     split: float,
                     metrics: list) -> None:
    """
    Method used to display the summary of pipeline options selected by user.

    Args:
        dataset (Dataset): The dataset object used for training.
        features (list): The list of input features selected by user.
        target_feature (str): The target feature selected by user.
        task_type (str): The task type selected by user.
        model_name (str): The model name selected by user.
        split (float): The train-test split ratio.
        metrics (list): The list of metrics selected by user.
    """
    st.subheader("Pipeline Summary")

    st.write(f"**Dataset:** {dataset.name} (v{dataset.version})")
    st.write(f"**Input Features:** {', '.join(features)}")
    st.write(f"**Target Feature:** {target_feature}")
    st.write(f"**Task Type:** {task_type}")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Train-Test Split Ratio:** {split}")
    st.write(f"**Metrics:** {', '.join(metrics)}")


def train_pipeline(dataset: Dataset,
                   features: list,
                   target_feature: str,
                   model_name: str,
                   split: float,
                   metrics: list,
                   map_features: dict) -> dict:
    """
    Function used to train the pipeline given all its info selected by user.

    Args:
        dataset (Dataset): The dataset object used for training.
        features (list): The list of input features selected by user.
        target_feature (str): The target feature selected by user.
        model_name (str): The model name selected by user.
        split (float): The train-test split ratio.
        metrics (list): The list of metrics selected by user.
        map_features (dict): A mapping of feature names to their objects.

    Returns:
        dict: The results of the pipeline execution.
    """
    try:
        model_features = [map_features[feature] for feature in features]
        model_target_feature = map_features[target_feature]

        model_object = get_model(model_name)

        model_metric = [get_metric(metric) for metric in metrics]

        pipeline = Pipeline(
            metrics=model_metric,
            dataset=dataset,
            model=model_object,
            input_features=model_features,
            target_feature=model_target_feature,
            split=split
        )

        with st.spinner("Training the pipeline..."):
            results = pipeline.execute()
        st.success("Training completed!")

        return results
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None


def write_results(results: dict) -> None:
    """
    Method used for rendering the results of pipeline execution.

    Args:
        results (dict): The results of the pipeline execution
    """
    st.subheader("Pipeline Results")

    st.write("**Training Metrics:**")
    for name, value in results["train_metrics"]:
        st.write(f"{name}: {value}")

    st.write("**Test Metrics:**")
    for name, value in results["test_metrics"]:
        st.write(f"{name}: {value}")

    st.write("**Predictions:**")
    pred_array = np.array(results["predictions"]).flatten()
    predictions = pd.DataFrame(pred_array, columns=["Predictions"])
    st.write(predictions)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline"
                  " to train a model on a dataset.")

if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'last_inputs' not in st.session_state:
    st.session_state['last_inputs'] = None

dataset = select_dataset()
if not dataset:
    st.stop()

features, target_feature, task_type, map_features = select_features(dataset)
if not (features and target_feature):
    st.stop()

model_name, metrics = select_model_metrics(task_type)
if not (model_name and metrics):
    st.stop()

ratio = st.slider("Train-Test Split Ratio", 0.1, 0.9, 0.8, 0.05)

pipeline_summary(dataset,
                 features,
                 target_feature,
                 task_type,
                 model_name,
                 ratio,
                 metrics)

inputs = {
    "dataset": dataset,
    "features": features,
    "target_feature": target_feature,
    "task_type": task_type,
    "model_name": model_name,
    "ratio": ratio,
    "metrics": metrics,
}


if st.session_state['last_inputs'] != inputs:
    st.session_state['results'] = None

st.session_state['last_inputs'] = inputs

if st.button("Train Pipeline"):
    results = train_pipeline(dataset,
                             features,
                             target_feature,
                             model_name,
                             ratio,
                             metrics,
                             map_features)
    if results:
        st.session_state['results'] = results
        write_results(results)
else:
    if st.session_state['results']:
        write_results(st.session_state['results'])
