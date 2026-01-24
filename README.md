# Dataset Health Checker

This project is a small system that performs automated analysis of tabular datasets.

Instead of manually doing exploratory data analysis for every dataset, this tool extracts a fixed set of metrics, routes the dataset through appropriate analysis modules, and generates a structured report describing the dataset quality and modeling readiness.

The focus is on **analysis and diagnostics**, not model training.

---

## What Problem This Solves

In most data science projects, a large amount of time is spent on:

- checking missing values  
- understanding target columns  
- detecting messy formatting  
- identifying biased or unstable data  

This project automates that initial analysis step and produces:

- a structured verdict object  
- a short human-readable report  

It is designed to be:

- deterministic  
- explainable  
- modular  
- easy to extend  

---

## High-Level Architecture

The system is built as a pipeline with clear layers:

1. **Metrics Extractor**  
   Computes all dataset-level statistics once.

2. **Router**  
   Decides which analysis modules to run based on:
   - dataset type (clean / messy / biased)  
   - target type (regression / classification)

3. **Interpreters**  
   Modular analysis units:
   - Core Interpreter  
   - Clean Interpreter  
   - Messy Interpreter  
   - Regression Interpreter  
   - Bias Interpreter  

4. **Orchestrator**  
   Runs only the required interpreters and collects findings.

5. **Verdict Generator**  
   Assembles a unified structured result.

6. **Explanation & Report Generator**  
   Converts the verdict into a short text report.

All interpreters operate only on precomputed metrics.  
No interpreter accesses the raw dataframe directly.

---

## Project Phases

The project is developed in clear phases:

- Phase 1: Dataset loading and schema inspection  
- Phase 2: Unified metrics extraction  
- Phase 3: Routing logic  
- Phase 4: Modular interpreters  
- Phase 5: Orchestration and verdict assembly  
- Phase 6: Explanation and reporting  

Each phase is implemented and tested incrementally in notebooks.
