import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()  # singleton init
datasets = automl.registry.list(type="dataset")

st.set_page_config(
    page_title="Datasets",
    page_icon="📊",
)


def write_helper_text(text: str) -> None:
    """Method used for rendering the subtitle"""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def write_dataset_card(dataset: Dataset) -> None:
    """
    This function renders all the information for a dataset saved
    in the registry in a card format.

    Args:
        dataset (Dataset): The dataset object used to render its info.
    """
    st.write(f"**Name:** {dataset.name}")
    st.write(f"**Version:** {dataset.version}")
    if dataset.tags:
        st.write(f"**Tags:** {', '.join(dataset.tags)}")
    else:
        st.write("No tags available")
    st.write(f"**Asset Path:** {dataset.asset_path}")

    with st.expander("Additional metadata"):
        st.json(dataset.metadata)

    if st.button("Delete dataset", key=f"del_{dataset.id}"):
        automl.registry.delete(dataset.id)
        st.success(f"Dataset '{dataset.name}' deleted successfully.")
        st.rerun()


def write_upload_dataset_form() -> None:
    """
    This function renders the section of the page for uploading a new
    dataset in csv format and creating a new dataset artifact in the registry.
    """
    st.write("Preview of the dataset:")
    data = pd.read_csv(file)
    st.write(data.head())

    name = st.text_input("Name of the dataset")
    version = st.text_input("Version of the dataset", value="1.0.0")
    tags = st.text_input("Tags (comma separated)").split(",")

    if st.button("Create new dataset"):
        # converting to a dataframe
        dataset = Dataset.from_dataframe(
            data=data,
            name=name,
            asset_path=f"{name}.csv",
            version=version,
        )
        dataset.tags = [tag.strip() for tag in tags if tag.strip()]

        # req3: we user artifact registry to save converted dataset artifact
        automl.registry.register(dataset)
        st.success(f"Dataset '{name} created successfully!")
        st.rerun()


st.title("Dataset Management")
write_helper_text("From here, you can add, delete and view datasets.")

# View existing Datasets

st.subheader("Existing Datasets")

if not datasets:
    st.write("No datasets found.")
else:
    for dataset in datasets:
        write_dataset_card(dataset)

# Add new Datasets

st.subheader("Upload new Dataset")
file = st.file_uploader("Choose a CSV file", type="csv")

if file:
    write_upload_dataset_form()
