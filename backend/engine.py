import pandas as pd
import numpy as np

def extract_metrics(df, target_column=None):
    df = df.copy()
    
    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\n", " ", regex=False)
    )
    
    metrics = {}
    
    # Dataset-level metrics
    metrics["dataset"] = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "is_empty": df.empty
    }
    
    # Schema & column roles
    dtypes = df.dtypes.astype(str).to_dict()
    
    metrics["schema"] = {
        "columns": df.columns.tolist(),
        "dtypes": dtypes,
        "column_roles": {
            "numeric": df.select_dtypes(include="number").columns.tolist(),
            "categorical": df.select_dtypes(include="object").columns.tolist(),
            "boolean": df.select_dtypes(include="bool").columns.tolist()
        }
    }
    
    # Missingness metrics
    missing_per_column = df.isna().mean().to_dict()
    missing_per_row = df.isna().mean(axis=1).to_list()
    
    metrics["missingness"] = {
        "per_column_missing_pct": missing_per_column,
        "per_row_missing_pct": missing_per_row,
        "any_missing": any(v > 0 for v in missing_per_column.values())
    }
    
    # Uniqueness & cardinality
    nunique_per_column = df.nunique(dropna=False).to_dict()
    
    uniqueness_ratio = {
        col: nunique_per_column[col] / len(df) if len(df) > 0 else 0
        for col in df.columns
    }
    
    constant_columns = [col for col, n in nunique_per_column.items() if n == 1]
    
    identifier_like_columns = [
        col for col, r in uniqueness_ratio.items()
        if r > 0.95
    ]
    
    metrics["uniqueness"] = {
        "nunique_per_column": nunique_per_column,
        "uniqueness_ratio_per_column": uniqueness_ratio,
        "constant_columns": constant_columns,
        "identifier_like_columns": identifier_like_columns
    }
    
    # Target metrics
    metrics["target"] = {
        "name": target_column,
        "exists": target_column in df.columns if target_column else False
    }
    
    if target_column and target_column in df.columns:
        target_series = df[target_column]
        
        target_dtype = str(target_series.dtype)
        
        if target_series.dtype == bool:
            target_type = "boolean"
        elif target_series.dtype == object:
            target_type = "categorical"
        else:
            target_type = "numeric"
        
        metrics["target"].update({
            "dtype": target_dtype,
            "type": target_type,
            "missing_pct": target_series.isna().mean(),
            "nunique": target_series.nunique(dropna=False),
            "uniqueness_ratio": (
                target_series.nunique(dropna=False) / len(df) if len(df) > 0 else 0
            )
        })
        
        if target_type == "numeric":
            metrics["target"]["distribution_stats"] = target_series.describe().to_dict()
        
        if target_type in ["boolean", "categorical"]:
            vc = target_series.value_counts(normalize=True, dropna=False).to_dict()
            metrics["target"]["value_counts"] = vc
            
            if target_type == "boolean":
                metrics["target"]["balance_rate"] = target_series.mean()
    
    # Numeric distribution metrics
    numeric_cols = df.select_dtypes(include="number").columns
    
    dist_stats = {}
    
    for col in numeric_cols:
        desc = df[col].describe()
        dist_stats[col] = {
            "min": desc["min"],
            "max": desc["max"],
            "mean": desc["mean"],
            "std": desc["std"],
            "25%": desc["25%"],
            "50%": desc["50%"],
            "75%": desc["75%"]
        }
    
    metrics["distribution"] = {
        "per_numeric_column_stats": dist_stats
    }
    
    # Formatting & type-confusion flags
    currency_like = []
    comma_numeric = []
    mixed_type = []
    
    for col in df.select_dtypes(include="object").columns:
        series = df[col].dropna().astype(str)
        
        if series.str.contains("$", regex=False).any():
            currency_like.append(col)
        
        if series.str.contains(",", regex=False).any():
            comma_numeric.append(col)
        
        numeric_ratio = series.str.replace(".", "", regex=False).str.isnumeric().mean()
        if 0 < numeric_ratio < 1:
            mixed_type.append(col)
    
    metrics["formatting"] = {
        "currency_like_columns": currency_like,
        "comma_separated_numeric_columns": comma_numeric,
        "mixed_type_columns": mixed_type
    }
    
    return metrics

def route_checks(metrics, dataset_type):
    target_info = metrics.get("target", {})
    
    # Hard stop if no target
    if not target_info.get("exists", False):
        return {
            "status": "stop",
            "reason": "Target column does not exist",
            "task_type": None,
            "dataset_type": dataset_type,
            "run_interpreters": [],
            "skip_modules": ["all"]
        }
    
    # Determine task type
    target_type = target_info.get("type")
    
    if target_type == "numeric":
        task_type = "regression"
    elif target_type == "boolean":
        task_type = "binary_classification"
    elif target_type == "categorical":
        task_type = "multiclass_classification"
    else:
        task_type = "unknown"
    
    run_interpreters = ["core"]
    skip_modules = []
    
    # Dataset-type routing
    if dataset_type == "clean":
        run_interpreters.append("clean")
        skip_modules.append("messy")
    elif dataset_type == "messy":
        run_interpreters.append("messy")
        skip_modules.append("clean")
    elif dataset_type == "biased":
        pass
    
    # Task-type routing
    if task_type == "regression":
        run_interpreters.append("regression")
        skip_modules.append("bias")
    
    elif task_type == "binary_classification":
        run_interpreters.append("bias")
        skip_modules.append("regression")
    
    elif task_type == "multiclass_classification":
        run_interpreters.append("classification")
        skip_modules.append("regression")
    
    router_config = {
        "status": "ok",
        "task_type": task_type,
        "dataset_type": dataset_type,
        "run_interpreters": run_interpreters,
        "skip_modules": skip_modules
    }
    
    return router_config

def core_interpreter(metrics):
    findings = {}
    
    # Missingness findings
    missing_info = metrics.get("missingness", {})
    per_col_missing = missing_info.get("per_column_missing_pct", {})
    
    missing_columns = [col for col, pct in per_col_missing.items() if pct > 0]
    high_missing_columns = {
        col: pct for col, pct in per_col_missing.items() if pct > 0.3
    }
    
    findings["missing_columns"] = missing_columns
    findings["high_missing_columns"] = high_missing_columns
    
    # Uniqueness findings
    uniq_info = metrics.get("uniqueness", {})
    
    constant_columns = uniq_info.get("constant_columns", [])
    identifier_like_columns = uniq_info.get("identifier_like_columns", [])
    
    findings["constant_columns"] = constant_columns
    findings["identifier_like_columns"] = identifier_like_columns
    
    # Target validity findings
    target_info = metrics.get("target", {})
    
    target_findings = {}
    
    if not target_info.get("exists", False):
        target_findings["exists"] = False
    else:
        target_findings["exists"] = True
        target_findings["missing_pct"] = target_info.get("missing_pct")
        target_findings["type"] = target_info.get("type")
        target_findings["uniqueness_ratio"] = target_info.get("uniqueness_ratio")
    
    findings["target"] = target_findings
    
    # Leakage candidates
    leakage_candidates = {}
    
    leakage_candidates["identifier_like_columns"] = (
        metrics.get("uniqueness", {}).get("identifier_like_columns", [])
    )
    
    findings["leakage_candidates"] = leakage_candidates
    
    return findings

def clean_interpreter(metrics):
    findings = {}
    
    # Missingness in clean data should be near zero
    missing_info = metrics.get("missingness", {})
    per_col_missing = missing_info.get("per_column_missing_pct", {})
    
    any_missing = [col for col, pct in per_col_missing.items() if pct > 0]
    
    findings["any_missing_columns"] = any_missing
    
    # Type stability issues (object columns that look numeric)
    formatting_info = metrics.get("formatting", {})
    
    mixed_type_columns = formatting_info.get("mixed_type_columns", [])
    currency_like_columns = formatting_info.get("currency_like_columns", [])
    comma_numeric_columns = formatting_info.get("comma_separated_numeric_columns", [])
    
    findings["mixed_type_columns"] = mixed_type_columns
    findings["currency_like_columns"] = currency_like_columns
    findings["comma_numeric_columns"] = comma_numeric_columns
    
    # Numeric sanity checks
    dist_info = metrics.get("distribution", {}).get("per_numeric_column_stats", {})
    
    zero_variance_columns = []
    
    for col, stats in dist_info.items():
        if stats["std"] == 0:
            zero_variance_columns.append(col)
    
    findings["zero_variance_numeric_columns"] = zero_variance_columns
    
    return findings

def messy_interpreter(metrics):
    findings = {}
    
    # Missingness profile (messy tolerates missing, but records distribution)
    missing_info = metrics.get("missingness", {})
    per_col_missing = missing_info.get("per_column_missing_pct", {})
    
    high_missing_columns = {
        col: pct for col, pct in per_col_missing.items() if pct > 0.3
    }
    
    moderate_missing_columns = {
        col: pct for col, pct in per_col_missing.items() if 0 < pct <= 0.3
    }
    
    findings["high_missing_columns"] = high_missing_columns
    findings["moderate_missing_columns"] = moderate_missing_columns
    
    # Formatting issues (core messy signals)
    formatting_info = metrics.get("formatting", {})
    
    mixed_type_columns = formatting_info.get("mixed_type_columns", [])
    currency_like_columns = formatting_info.get("currency_like_columns", [])
    comma_numeric_columns = formatting_info.get("comma_separated_numeric_columns", [])
    
    findings["mixed_type_columns"] = mixed_type_columns
    findings["currency_like_columns"] = currency_like_columns
    findings["comma_numeric_columns"] = comma_numeric_columns
    
    # Cardinality extremes (too low or too high)
    uniq_info = metrics.get("uniqueness", {})
    uniqueness_ratio = uniq_info.get("uniqueness_ratio_per_column", {})
    
    low_cardinality_columns = [
        col for col, r in uniqueness_ratio.items() if r < 0.01
    ]
    
    high_cardinality_columns = [
        col for col, r in uniqueness_ratio.items() if r > 0.9
    ]
    
    findings["low_cardinality_columns"] = low_cardinality_columns
    findings["high_cardinality_columns"] = high_cardinality_columns
    
    return findings

def regression_interpreter(metrics):
    findings = {}
    
    # Target distribution analysis
    target_info = metrics.get("target", {})
    
    target_findings = {}
    
    if target_info.get("type") == "numeric":
        dist = target_info.get("distribution_stats", {})
        
        target_findings["min"] = dist.get("min")
        target_findings["max"] = dist.get("max")
        target_findings["mean"] = dist.get("mean")
        target_findings["std"] = dist.get("std")
        
        # Simple skew proxy: mean vs median
        median = dist.get("50%")
        mean = dist.get("mean")
        
        if median is not None and mean is not None:
            target_findings["mean_minus_median"] = mean - median
        
        # Range sanity
        if dist.get("min") is not None and dist.get("max") is not None:
            target_findings["range"] = dist.get("max") - dist.get("min")
    
    findings["target_distribution"] = target_findings
    
    # Numeric feature sanity
    dist_info = metrics.get("distribution", {}).get("per_numeric_column_stats", {})
    
    zero_variance_features = []
    extreme_range_features = []
    
    for col, stats in dist_info.items():
        if stats["std"] == 0:
            zero_variance_features.append(col)
        
        if stats["min"] is not None and stats["max"] is not None:
            if stats["max"] > 1e6 * max(1, abs(stats["min"])):
                extreme_range_features.append(col)
    
    findings["zero_variance_features"] = zero_variance_features
    findings["extreme_range_features"] = extreme_range_features
    
    return findings

def bias_interpreter(metrics):
    findings = {}
    
    target_info = metrics.get("target", {})
    
    # Bias interpreter only valid for boolean targets
    if target_info.get("type") != "boolean":
        findings["status"] = "skipped"
        findings["reason"] = "Target is not boolean"
        return findings
    
    # Baseline rate
    baseline_rate = target_info.get("balance_rate")
    
    findings["baseline_rate"] = baseline_rate
    
    # Group-level outcome rates
    group_bias = {}
    
    schema_info = metrics.get("schema", {})
    candidate_group_columns = (
        schema_info.get("column_roles", {}).get("categorical", []) +
        schema_info.get("column_roles", {}).get("boolean", [])
    )
    
    findings["candidate_sensitive_columns"] = candidate_group_columns
    findings["group_outcome_rates"] = {}  # placeholder for later extension
    
    return findings

def infer_dataset_type(metrics):
    scores = {
        "clean": 0.0,
        "messy": 0.0,
        "biased": 0.0
    }

    # ---- Messy signals ----
    missing_pct = metrics["missingness"]["any_missing"]
    high_missing = len(metrics["missingness"]["per_column_missing_pct"])
    mixed_cols = len(metrics["formatting"]["mixed_type_columns"])
    formatting_issues = (
        len(metrics["formatting"]["currency_like_columns"]) +
        len(metrics["formatting"]["comma_separated_numeric_columns"])
    )

    if metrics["missingness"]["any_missing"]:
        scores["messy"] += 0.4
    if mixed_cols > 0:
        scores["messy"] += 0.3
    if formatting_issues > 0:
        scores["messy"] += 0.3

    # ---- Bias signals ----
    target_type = metrics.get("target", {}).get("type")
    if target_type == "boolean":
        scores["biased"] += 0.4

    sensitive_cols = len(
        metrics.get("schema", {})
        .get("column_roles", {})
        .get("categorical", [])
    )

    if sensitive_cols > 0:
        scores["biased"] += 0.2

    # ---- Clean signals ----
    if scores["messy"] < 0.2 and scores["biased"] < 0.2:
        scores["clean"] += 0.6
    if not metrics["missingness"]["any_missing"]:
        scores["clean"] += 0.2

    dataset_type = max(scores, key=scores.get)

    return dataset_type, scores


def run_analysis_pipeline(df, target_column):
    # Step 1: Extract metrics
    metrics = extract_metrics(df, target_column=target_column)

    dataset_type, dataset_scores = infer_dataset_type(metrics)
    
    # Step 2: Route checks
    router_config = route_checks(metrics, dataset_type=dataset_type)
    
    # If router says stop, return early
    if router_config["status"] == "stop":
        return {
            "status": "stopped",
            "reason": router_config.get("reason"),
            "metrics": metrics,
            "router": router_config,
            "findings": {}
        }
    
    findings = {}
    
    # Step 3: Always run core interpreter
    if "core" in router_config["run_interpreters"]:
        findings["core"] = core_interpreter(metrics)
    
    # Step 4: Conditionally run dataset interpreters
    if "clean" in router_config["run_interpreters"]:
        findings["clean"] = clean_interpreter(metrics)
    
    if "messy" in router_config["run_interpreters"]:
        findings["messy"] = messy_interpreter(metrics)
    
    # Step 5: Conditionally run task interpreters
    if "regression" in router_config["run_interpreters"]:
        findings["regression"] = regression_interpreter(metrics)
    
    if "bias" in router_config["run_interpreters"]:
        findings["bias"] = bias_interpreter(metrics)
    
    # Final unified output
    result = {
        "status": "ok",
        "dataset_type": dataset_type,
        "dataset_scores": dataset_scores,
        "task_type": router_config["task_type"],
        "router": router_config,
        "metrics": metrics,
        "findings": findings
    }
    
    return result

def generate_verdict(result):
    verdict = {}
    
    verdict["status"] = result.get("status")
    verdict["dataset_type"] = result.get("dataset_type")
    verdict["task_type"] = result.get("task_type")
    
    findings = result.get("findings", {})
    
    summary = {}
    
    # Core summary
    core = findings.get("core", {})
    summary["missing_columns"] = core.get("missing_columns", [])
    summary["high_missing_columns"] = core.get("high_missing_columns", {})
    summary["constant_columns"] = core.get("constant_columns", [])
    summary["identifier_like_columns"] = core.get("identifier_like_columns", [])
    
    # Clean summary
    if "clean" in findings:
        clean = findings.get("clean", {})
        summary["clean_issues"] = {
            "any_missing_columns": clean.get("any_missing_columns", []),
            "mixed_type_columns": clean.get("mixed_type_columns", []),
            "currency_like_columns": clean.get("currency_like_columns", []),
            "comma_numeric_columns": clean.get("comma_numeric_columns", []),
            "zero_variance_numeric_columns": clean.get("zero_variance_numeric_columns", [])
        }
    
    # Messy summary
    if "messy" in findings:
        messy = findings.get("messy", {})
        summary["messy_profile"] = {
            "high_missing_columns": messy.get("high_missing_columns", {}),
            "moderate_missing_columns": messy.get("moderate_missing_columns", {}),
            "mixed_type_columns": messy.get("mixed_type_columns", []),
            "currency_like_columns": messy.get("currency_like_columns", []),
            "comma_numeric_columns": messy.get("comma_numeric_columns", [])
        }
    
    # Regression summary
    if "regression" in findings:
        reg = findings.get("regression", {})
        summary["regression_profile"] = {
            "target_distribution": reg.get("target_distribution", {}),
            "zero_variance_features": reg.get("zero_variance_features", []),
            "extreme_range_features": reg.get("extreme_range_features", [])
        }
    
    # Bias summary
    if "bias" in findings:
        bias = findings.get("bias", {})
        summary["bias_profile"] = bias
    
    verdict["summary"] = summary
    
    return verdict

def explain_verdict(verdict):
    summary = verdict.get("summary", {})
    
    explanations = []
    
    # Missingness explanation
    missing_cols = summary.get("missing_columns", [])
    high_missing = summary.get("high_missing_columns", {})
    
    if not missing_cols:
        explanations.append("No missing values detected in any column.")
    else:
        explanations.append(f"Missing values detected in {len(missing_cols)} columns.")
        
        if high_missing:
            explanations.append(
                f"{len(high_missing)} columns have high missingness (>30%)."
            )
    
    # Constant / identifier-like columns
    constant_cols = summary.get("constant_columns", [])
    id_like_cols = summary.get("identifier_like_columns", [])
    
    if constant_cols:
        explanations.append(
            f"{len(constant_cols)} constant columns detected: {constant_cols}."
        )
    
    if id_like_cols:
        explanations.append(
            f"{len(id_like_cols)} identifier-like columns detected: {id_like_cols}."
        )
    
    # Clean issues
    clean_issues = summary.get("clean_issues")
    
    if clean_issues is not None:
        any_missing = clean_issues.get("any_missing_columns", [])
        mixed = clean_issues.get("mixed_type_columns", [])
        currency = clean_issues.get("currency_like_columns", [])
        comma = clean_issues.get("comma_numeric_columns", [])
        zero_var = clean_issues.get("zero_variance_numeric_columns", [])
        
        if not any_missing and not mixed and not currency and not comma and not zero_var:
            explanations.append("No data quality issues detected under clean-data assumptions.")
        else:
            if any_missing:
                explanations.append(f"Unexpected missing values in columns: {any_missing}.")
            if mixed:
                explanations.append(f"Mixed-type columns detected: {mixed}.")
            if currency:
                explanations.append(f"Currency-formatted columns detected: {currency}.")
            if comma:
                explanations.append(f"Comma-formatted numeric columns detected: {comma}.")
            if zero_var:
                explanations.append(f"Zero-variance numeric columns detected: {zero_var}.")
    
    # Regression profile
    reg_profile = summary.get("regression_profile")
    
    if reg_profile is not None:
        target_dist = reg_profile.get("target_distribution", {})
        
        if target_dist:
            explanations.append(
                f"Target range: {target_dist.get('min')} to {target_dist.get('max')}, "
                f"mean = {round(target_dist.get('mean', 0), 2)}, "
                f"std = {round(target_dist.get('std', 0), 2)}."
            )
    
    # Bias profile
    bias_profile = summary.get("bias_profile")
    
    if bias_profile is not None:
        if bias_profile.get("status") == "skipped":
            explanations.append("Bias analysis was skipped because the target is not boolean.")
        else:
            baseline = bias_profile.get("baseline_rate")
            explanations.append(f"Baseline positive rate: {baseline}.")
    
    return explanations

def generate_text_report(verdict):
    header = [
        "Automated Dataset Analysis Report",
        "-" * 35,
        f"Dataset type: {verdict.get('dataset_type')}",
        f"Task type: {verdict.get('task_type')}",
        ""
    ]
    
    explanations = explain_verdict(verdict)
    
    body = [f"- {line}" for line in explanations]
    
    report_lines = header + body
    
    report = "\n".join(report_lines)
    
    return report
