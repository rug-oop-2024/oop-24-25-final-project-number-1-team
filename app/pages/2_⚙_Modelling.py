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

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

def select_dataset():
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

def select_features(dataset):
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
        st.error("Target feature type is unknown. Please select a valid target feature.")
        return None, None, None, None
    
    st.write(f"Task type: **{task_type}**")
    return input_features, target_feature, task_type, map_features

def select_model_metrics(type):
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

def pipeline_summary(dataset, features, target_feature, task_type, model_name, split, metrics):
    st.subheader("Pipeline Summary")

    st.write(f"**Dataset:** {dataset.name} (v{dataset.version})")
    st.write(f"**Input Features:** {', '.join(features)}")
    st.write(f"**Target Feature:** {target_feature}")
    st.write(f"**Task Type:** {task_type}")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Train-Test Split Ratio:** {split}")
    st.write(f"**Metrics:** {', '.join(metrics)}")

def train_pipeline(dataset, features, target_feature, model_name, split, metrics, map_features):
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

def write_results(results):
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
write_helper_text("In this section, you can design a machine learning pipeline "
                  "to train a model on a dataset.")

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

pipeline_summary(dataset, features, target_feature, task_type, model_name, ratio, metrics)

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
    results = train_pipeline(dataset, features, target_feature, model_name, ratio, metrics, map_features)
    if results:
        st.session_state['results'] = results
        write_results(results)
else:
    if st.session_state['results']:
        write_results(st.session_state['results'])