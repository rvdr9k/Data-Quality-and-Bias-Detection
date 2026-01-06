# Data Quality & Bias Detection (ML Readiness System)

## Overview
This project is a **data-first ML readiness and risk analysis system** designed to audit datasets *before model training*.  
It identifies **hidden data issues**—such as leakage, bias, imbalance, missingness, and instability—that commonly cause ML models to fail in production.

The goal is **not data validation**, but **data risk reasoning**.

---

## What This Project Does
- Profiles datasets to understand structure and health  
- Detects missing data patterns and non-random missingness  
- Analyzes feature distributions and skew  
- Identifies potential target leakage  
- Evaluates bias and imbalance across groups  
- Assesses dataset stability and drift  
- Produces a **data health score** with explanations and recommendations  

---

## What This Project Does NOT Do
- Train machine learning models  
- Perform AutoML or hyperparameter tuning  
- Replace data observability tools (e.g., Soda, WhyLabs)  
- Act as a production monitoring system  

---

## Why This Exists
Many ML failures originate **before modeling**, due to:
- Target leakage inflating accuracy  
- Biased or imbalanced data  
- Poor label quality  
- Silent data drift  

This system helps answer one critical question:

> *“Is this dataset safe and trustworthy to train a model on?”*

---

## Core Concepts Covered
- Data profiling & exploratory analysis  
- Statistical reasoning for data quality  
- Bias and fairness at the data level  
- Target leakage detection  
- Risk scoring and explainability  

---

## High-Level Architecture
1. Dataset ingestion  
2. Analytical checks (profiling, bias, leakage, drift)  
3. Risk interpretation & scoring  
4. Human-readable and machine-readable reports  

---

## Tech Stack
- Python  
- Pandas, NumPy, SciPy  
- scikit-learn (diagnostics only)  
- FastAPI (API layer)  
- Pydantic  
- Matplotlib  

---

## Intended Outcome
- Strong understanding of **why ML systems fail**  
- Ability to reason about data risk *before* modeling  
- A reusable framework for dataset audits and ML readiness checks  

---


