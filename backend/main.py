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
