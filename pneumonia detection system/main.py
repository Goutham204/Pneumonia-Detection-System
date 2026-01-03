import streamlit as st
import requests
import pandas as pd
import os
import datetime
import json

st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("Pneumonia Detection from X-Rays")

if "predictions_df" not in st.session_state:
    st.session_state.predictions_df = None

if "predicted" not in st.session_state:
    st.session_state.predicted = False

uploaded_files = st.file_uploader(
    "Choose chest X-ray images (JPEG/PNG)",
    type=["jpeg", "jpg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("Uploaded Images")
    for f in uploaded_files:
        st.write(f.name)

    if st.button("Predict Images"):
        files = [("images", (f.name, f, f.type)) for f in uploaded_files]

        with st.spinner("Running predictions..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/predict",
                    files=files
                )
            except requests.exceptions.ConnectionError:
                st.error("Backend is not running.")
                st.stop()

        if response.status_code == 200:
            df = pd.DataFrame(response.json())

            cols = [c for c in df.columns if c != "confidence"] + ["confidence"]
            df = df[cols]

            st.session_state.predictions_df = df
            st.session_state.predicted = True

            st.subheader("Prediction Results")
            st.dataframe(df)
        else:
            st.error("Prediction failed.")

if not st.session_state.predicted:
    st.stop()

predictions_df = st.session_state.predictions_df

st.subheader("Report Settings")

output_format = st.selectbox(
    "Select report output format",
    ["CSV", "JSON", "CSV+JSON"]
)

report_mode = st.selectbox(
    "Report mode",
    ["Create New", "Append to Existing File"]
)

os.makedirs("reports", exist_ok=True)
existing_reports = [
    f for f in os.listdir("reports")
    if f.endswith(".csv") or f.endswith(".json")
]

selected_existing_file = None
if report_mode == "Append to Existing File":
    if not existing_reports:
        st.warning("No existing reports found. Create a new report first.")
        st.stop()

    selected_existing_file = st.selectbox(
        "Select existing report",
        existing_reports
    )

if st.button("Generate Report"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if report_mode == "Create New":
        if output_format in ["CSV", "CSV+JSON"]:
            csv_path = f"reports/report_{timestamp}.csv"
            predictions_df.to_csv(csv_path, index=False)
            st.success(f"CSV report created: {csv_path}")

        if output_format in ["JSON", "CSV+JSON"]:
            json_path = f"reports/report_{timestamp}.json"
            predictions_df.to_json(json_path, orient="records", indent=4)
            st.success(f"JSON report created: {json_path}")

    else:
        file_path = os.path.join("reports", selected_existing_file)
        ext = os.path.splitext(file_path)[1]

        if ext == ".csv":
            predictions_df.to_csv(file_path, mode="a", index=False, header=False)
            st.success(f"Data appended to {selected_existing_file}")

        elif ext == ".json":
            with open(file_path, "r+") as f:
                existing_data = json.load(f)
                existing_data.extend(predictions_df.to_dict(orient="records"))
                f.seek(0)
                json.dump(existing_data, f, indent=4)
            st.success(f"Data appended to {selected_existing_file}")

st.subheader("Download Prediction Results")

download_df = predictions_df.copy()
download_json = None

if report_mode == "Append to Existing File" and selected_existing_file:
    file_path = os.path.join("reports", selected_existing_file)
    ext = os.path.splitext(file_path)[1]

    if ext == ".csv" and os.path.exists(file_path):
        download_df = pd.read_csv(file_path)

    elif ext == ".json" and os.path.exists(file_path):
        with open(file_path, "r") as f:
            download_json = json.load(f)

csv_data = download_df.to_csv(index=False)
json_data = (
    json.dumps(download_json, indent=4)
    if download_json is not None
    else download_df.to_json(orient="records", indent=4)
)

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "Download CSV",
        csv_data,
        file_name="pneumonia_predictions_full.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        "Download JSON",
        json_data,
        file_name="pneumonia_predictions_full.json",
        mime="application/json"
    )

st.divider()
if st.button("Reset Session"):
    st.session_state.predictions_df = None
    st.session_state.predicted = False
    st.rerun()
