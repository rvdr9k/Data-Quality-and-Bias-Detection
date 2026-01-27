import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(
    page_title="Dataset Health Checker",
    layout="wide"
)

# ---------- Sidebar (Inputs) ----------

st.sidebar.title("Dataset Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

target_column = st.sidebar.text_input(
    "Target Column",
    help="Leave empty if no target column is available"
)

run_button = st.sidebar.button("Run Analysis")

st.sidebar.markdown("---")
st.sidebar.write("The system will automatically infer:")
st.sidebar.write("- Dataset quality (clean / messy / biased)")
st.sidebar.write("- Task type (regression / classification)")
st.sidebar.write("- Key data issues")

# ---------- Main Title ----------

st.title("Dataset Health Checker")

st.write(
    "Upload a dataset and run automated analysis to assess data quality, "
    "target health, and modeling readiness."
)

st.markdown("---")

# ---------- Run Analysis ----------

if run_button:
    if uploaded_file is None:
        st.error("Please upload a dataset first.")
    else:
        with st.spinner("Running analysis..."):
            files = {
                "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
            }
            
            data = {
                "target_column": target_column if target_column else ""
            }
            
            try:
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code != 200:
                    st.error(f"API error: {response.text}")
                else:
                    result = response.json()
                    
                    if result.get("status") != "ok":
                        st.error(result.get("message", "Analysis failed"))
                    else:
                        verdict = result.get("verdict", {})
                        report = result.get("report", "")
                        
                        # ---------- Verdict Section ----------
                        
                        st.success("Analysis completed successfully.")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.subheader("Dataset Verdict")
                            
                            dataset_type = verdict.get("dataset_type", "unknown")
                            st.metric(
                                label="Inferred Dataset Type",
                                value=dataset_type.capitalize()
                            )
                            
                            task_type = verdict.get("task_type", "unknown")
                            st.metric(
                                label="Task Type",
                                value=task_type.replace("_", " ").capitalize()
                            )
                        
                        with col2:
                            st.subheader("Dataset Type Scores")
                            
                            scores = verdict.get("dataset_scores", {})
                            
                            if scores:
                                st.write("Relative confidence for each class:")
                                
                                st.progress(min(1.0, scores.get("clean", 0.0)))
                                st.caption(f"Clean score: {round(scores.get('clean', 0.0), 3)}")
                                
                                st.progress(min(1.0, scores.get("messy", 0.0)))
                                st.caption(f"Messy score: {round(scores.get('messy', 0.0), 3)}")
                                
                                st.progress(min(1.0, scores.get("biased", 0.0)))
                                st.caption(f"Biased score: {round(scores.get('biased', 0.0), 3)}")
                        
                        st.markdown("---")
                        
                        
                        
                        st.subheader("Analysis Report")
                        
                        st.text_area(
                            label="",
                            value=report,
                            height=250
                        )
                        
                        # Download button
                        st.download_button(
                            label="Download Report",
                            data=report,
                            file_name="analysis_report.txt",
                            mime="text/plain"
                        )
                        
                        st.markdown("---")
                        
                       
                        
                        with st.expander("Show JSON output"):
                            st.json(verdict)
            
            except Exception as e:
                st.error(f"Failed to connect to backend: {str(e)}")

else:
    st.info("Upload a dataset from the sidebar and click 'Run Analysis' to begin.")
