from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
from engine import (
    run_analysis_pipeline,
    generate_verdict,
    generate_text_report
)

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Analysis backend is running"}

@app.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(None),
    
):
    # 1. Read uploaded file into DataFrame
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        else:
            return {"status": "error", "message": "Only CSV or Excel files are supported"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to read file: {str(e)}"}
    
    # 2. Run analysis pipeline
    try:
        result = run_analysis_pipeline(
            df,
            target_column=target_column,
            
        )
        
        verdict = generate_verdict(result)
        report = generate_text_report(verdict)
        
    except Exception as e:
        return {"status": "error", "message": f"Analysis failed: {str(e)}"}
    
    # 3. Return both JSON verdict and text report
    return {
        "status": "ok",
        "verdict": verdict,
        "report": report
    }
