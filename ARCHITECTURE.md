
---

# 📄 ARCHITECTURE.md 

```markdown
# Architecture

This system is built as a three-layer architecture:

1. Frontend (Streamlit)  
2. Backend API (FastAPI)  
3. Analysis Engine  

## Component Roles

### Frontend
- Accepts file upload  
- Accepts optional target column  
- Displays verdict, scores, and report  

### Backend API
- Handles file upload  
- Loads data into pandas  
- Calls the analysis engine  
- Returns JSON verdict and text report  

### Analysis Engine
The engine performs:

1. Metric extraction  
2. Dataset type inference (clean / messy / biased using scores)  
3. Routing to interpreters  
4. Verdict and report generation  

## Design Principles

- No user-provided dataset type  
- No file I/O inside the engine  
- Deterministic and explainable logic  
- Clear separation between UI, API, and engine  
