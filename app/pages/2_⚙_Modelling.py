import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import get_model, REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.metric import get_metric, REGRESSION_METRICS, CLASSIFICATION_METRICS
from autoop.core.ml.pipeline import Pipeline

st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

if datasets:
    options = [f"{d.name} (v{d.version})" for d in datasets]
    mapping = {option: dataset for option, dataset in zip(options, datasets)}

    selected_option = st.selectbox("Select a dataset", options)
    selected_artifact = mapping[selected_option]

    if selected_artifact.type != "dataset":
        st.error("Selected artifact is not a dataset. Please select a valid dataset.")
        st.stop()
    selected_dataset = Dataset(
        name=selected_artifact.name,
        version=selected_artifact.version,
        asset_path=selected_artifact.asset_path,
        metadata=selected_artifact.metadata,
        tags=selected_artifact.tags,
        data=selected_artifact.read(),
    ) 



    features = detect_feature_types(selected_dataset)
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
    
    # find task type
    target_feature_selected = map_features[target_feature]
    if target_feature_selected.type == 'categorical':
        task_type = "classification"
    elif target_feature_selected.type == 'numerical':
        task_type = "regression"
    else:
        task_type = "unknown"
        st.error("Target feature type is unknown. Please select a valid target feature.")
        st.stop()
    
    st.write(f"Task type: **{task_type}**")

    if task_type == "classification":
        models = CLASSIFICATION_MODELS
    elif task_type == "regression":
        models = REGRESSION_MODELS
    else:
        st.error("Task type is unknown. Please select a valid target feature.")
        st.stop()
    
    selected_name = st.selectbox("Select a model", models)

    split_ratio = st.slider(
        "Select train-test split ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.8,
        step=0.05
    )

    if task_type == "regression":
        metric = REGRESSION_METRICS
    elif task_type == "classification":
        metric = CLASSIFICATION_METRICS
    else:
        st.error("Task type is unknown. Please select a valid target feature.")
        st.stop()
    
    selected_metrics = st.multiselect("Select metrics", metric)

    st.subheader("Pipeline Summary")

    st.write(f"**Dataset:** {selected_option}")
    st.write(f"**Input Features:** {', '.join(input_features)}")
    st.write(f"**Target Feature:** {target_feature}")
    st.write(f"**Task Type:** {task_type}")
    st.write(f"**Model:** {selected_name}")
    st.write(f"**Train-Test Split Ratio:** {split_ratio}")
    st.write(f"**Metrics:** {', '.join(selected_metrics)}")

    if st.button("Train Pipeline"):
        model_features = [map_features[feature] for feature in input_features]
        model_target_feature = map_features[target_feature]

        model_object = get_model(selected_name)

        model_metric = [get_metric(metric) for metric in selected_metrics]

        pipeline = Pipeline(
            metrics=model_metric,
            dataset=selected_dataset,
            model=model_object,
            input_features=model_features,
            target_feature=model_target_feature,
            split=split_ratio
        )

        with st.spinner("Training the pipeline..."):
            results = pipeline.execute()
        st.success("Training completed!")

        st.subheader("Pipeline Execution Results")

        st.write("**Training Metrics:**")

        for name, value in results["train_metrics"]:
            st.write(f"{name}: {value}")

        st.write("**Test Metrics:**")
        
        for name, value in results["test_metrics"]:
            st.write(f"{name}: {value}")
        
        if st.checkbox("Show Predictions"):
            st.write("**Predictions:**")
            st.write(results["predictions"])
        
else:
    st.error("No datasets found. Please upload a dataset first.")
    st.stop()