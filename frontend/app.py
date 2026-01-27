import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="Automated Dataset Analysis", layout="centered")

st.title("Automated Dataset Analysis Tool")

st.write("Upload a dataset and optionally provide a target column. The system will analyze the dataset and infer its quality automatically.")

# File upload
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# Target column input
target_column = st.text_input("Target column (optional)")


if st.button("Analyze"):
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
                        
                        st.success("Analysis completed successfully.")
                        
                        
                        st.subheader("Dataset Verdict")
                        st.write(f"**Inferred dataset type:** {verdict.get('dataset_type')}")
                        
                        scores = verdict.get("dataset_scores", {})
                        if scores:
                            st.write("**Dataset type scores:**")
                            st.json(scores)
                        
                        
                        st.subheader("Analysis Report")
                        st.text(report)
                        
                        
                        with st.expander("Show full raw verdict (JSON)"):
                            st.json(verdict)
            
            except Exception as e:
                st.error(f"Failed to connect to backend: {str(e)}")
