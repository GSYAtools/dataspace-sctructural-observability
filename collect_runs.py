#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_runs.py
Recolecta CSVs y parámetros (JSON) de múltiples ejecuciones del runner estructural.
Genera:
 - combined_flat.csv   : concatenación de todos los structural_double_flat_*.csv con columnas extra de parámetros
 - runs_metadata.csv   : una fila por ejecución con args y métricas agregadas (tau_jsd, tau_delta, n_rows, n_alerts)
Uso:
    python collect_runs.py --dirs path/run1 path/run2 ... --out summary_dir
o
    python collect_runs.py --dirs-file runs.txt --out summary_dir
"""
import argparse
import json
import sys
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import warnings

def find_latest(pattern, directory: Path):
    files = list(directory.glob(pattern))
    if not files:
        return None
    # pick latest modified
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def load_json_results(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load JSON {json_path}: {e}")
        return None

def flatten_args(args_dict):
    # Flatten nested args if present; keep simple scalar types
    flat = {}
    if not isinstance(args_dict, dict):
        return flat
    for k, v in args_dict.items():
        # skip large or complex objects
        if isinstance(v, (str, int, float, bool)) or v is None:
            flat[f"arg_{k}"] = v
        else:
            # attempt simple serialization
            try:
                flat[f"arg_{k}"] = json.dumps(v, ensure_ascii=False)
            except Exception:
                flat[f"arg_{k}"] = str(type(v))
    return flat

def collect_from_run(run_dir: Path):
    run_dir = Path(run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        warnings.warn(f"Run dir not found or not a directory: {run_dir}")
        return None, None

    # find CSV (flat) and JSON (results)
    csv_file = find_latest("structural_double_flat_*.csv", run_dir)
    json_file = find_latest("structural_double_end2end_*.json", run_dir)
    # fallback: any json in dir
    if json_file is None:
        js = list(run_dir.glob("*.json"))
        json_file = js[-1] if js else None

    meta = {}
    if json_file:
        data = load_json_results(json_file)
        if data is not None:
            # args may be under "args" key
            args = data.get("args", data.get("params", None))
            meta.update(flatten_args(args or {}))
            # add a few top-level metrics if present
            for key in ("tau_jsd", "tau_delta", "global_ref", "n_calibration"):
                if key in data:
                    # store as json-string if not scalar
                    val = data[key]
                    if isinstance(val, (str, int, float, bool)) or val is None:
                        meta[f"run_{key}"] = val
                    else:
                        try:
                            meta[f"run_{key}"] = json.dumps(val, ensure_ascii=False)[:200]
                        except Exception:
                            meta[f"run_{key}"] = str(type(val))
    else:
        warnings.warn(f"No json results found in {run_dir}")

    df = None
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            warnings.warn(f"Failed to read CSV {csv_file}: {e}")
            df = None

    # add context metadata
    run_info = {
        "run_dir": str(run_dir),
        "csv_file": str(csv_file) if csv_file else None,
        "json_file": str(json_file) if json_file else None
    }
    run_info.update(meta)
    return df, run_info

def main():
    parser = argparse.ArgumentParser(description="Collect structural run CSVs and parameters from multiple run folders")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dirs", nargs="+", help="List of run directories")
    group.add_argument("--dirs-file", help="File with one run directory per line")
    parser.add_argument("--out", default="collected_results", help="Output directory to store combined CSVs")
    parser.add_argument("--drop_columns", nargs="*", default=[], help="Columns to drop from per-run CSVs if needed")
    args = parser.parse_args()

    if args.dirs_file:
        with open(args.dirs_file, "r", encoding="utf-8") as f:
            dirs = [line.strip() for line in f if line.strip()]
    else:
        dirs = args.dirs

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    runs_meta = []
    for run_path in dirs:
        df, meta = collect_from_run(Path(run_path))
        if meta is None:
            continue
        runs_meta.append(meta)
        if df is None:
            continue
        # optional: drop unwanted columns
        for c in args.drop_columns:
            if c in df.columns:
                df = df.drop(columns=[c])
        # add run-level metadata columns to each row (prefix run_)
        for k, v in meta.items():
            # ensure no conflict with existing columns
            colname = f"run_{k}" if not str(k).startswith("run_") else str(k)
            df[colname] = v
        combined_rows.append(df)

    # concat all
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True, sort=False)
        combined_csv = out_dir / "combined_flat.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved combined flat CSV with {len(combined_df)} rows to {combined_csv}")
    else:
        print("No CSV rows found to combine.")
        combined_csv = None

    # runs metadata dataframe
    if runs_meta:
        meta_df = pd.DataFrame.from_records(runs_meta)
        # try to coerce numeric columns
        for col in meta_df.columns:
            try:
                meta_df[col] = pd.to_numeric(meta_df[col])
            except Exception:
                pass
        meta_csv = out_dir / "runs_metadata.csv"
        meta_df.to_csv(meta_csv, index=False)
        print(f"Saved runs metadata for {len(meta_df)} runs to {meta_csv}")
    else:
        print("No runs metadata collected.")

    # additionally, produce a small per-run summary (if combined df exists)
    if combined_csv is not None:
        comb = pd.read_csv(combined_csv)
        summaries = []
        grouped = comb.groupby("run_run_dir") if "run_run_dir" in comb.columns else comb.groupby("run_dir")
        for name, g in grouped:
            # try to use thresholds from run columns
            tau_jsd = None
            tau_delta = None
            # possible column names
            for c in ("run_tau_jsd","run_run_tau_jsd","run_tau_delta","run_run_tau_delta","run_tau_jsd"):
                if c in g.columns:
                    try:
                        tau_jsd = float(g[c].iloc[0])
                    except Exception:
                        pass
            # fallback to metadata file already generated
            n = len(g)
            alerts_jsd = int(g['providers_jsd'].gt(tau_jsd).sum()) if tau_jsd is not None and 'providers_jsd' in g.columns else None
            mean_jsd = float(g['providers_jsd'].dropna().mean()) if 'providers_jsd' in g.columns else None
            mean_delta = float(g['Delta_norm'].dropna().mean()) if 'Delta_norm' in g.columns else None
            summaries.append({
                "run_dir": name,
                "n_rows": n,
                "mean_jsd": mean_jsd,
                "alerts_jsd": alerts_jsd,
                "mean_delta": mean_delta
            })
        sum_df = pd.DataFrame(summaries)
        sum_csv = out_dir / "runs_summary.csv"
        sum_df.to_csv(sum_csv, index=False)
        print(f"Saved runs summary to {sum_csv}")

if __name__ == "__main__":
    main()