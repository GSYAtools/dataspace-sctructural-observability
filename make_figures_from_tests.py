#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures_from_test.py
Lee recursivamente JSONs generados por structural_double_end2end.py
y genera las figuras:
 - fig3_indicators_vs_p.png  (H y HHI vs p)
 - fig2a_mean_jsd_vs_p.png  (media JSD vs p)
 - fig2b_pct_alerts_vs_p.png (pct activaciones vs p)
 - hist_jsd_by_cfg.png      (hist: pooled_jsd vs omit total)
Salida en --outdir
"""
import argparse
import json
from pathlib import Path
import math
import statistics
import sys

# plotting libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_json_files(input_dir):
    p = Path(input_dir)
    files = list(p.rglob("structural_double_end2end_*.json"))
    files = sorted(files)
    return files

def safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def summarize_test_results(test_results, tau_jsd):
    """
    Devuelve dict resumen por configuración:
    { cfg_name: { 'n': int, 'mean_jsd': float or None, 'p95': float or None,
                  'alerts': int, 'pct_alerts': float, 'H_mean': float, 'HHI_mean': float } }
    """
    out = {}
    for cfg, items in (test_results or {}).items():
        n = len(items) if items is not None else 0
        jsd_vals = []
        alerts = 0
        hs = []
        hhis = []
        no_data = 0
        for it in (items or []):
            jsd = it.get("providers_jsd")
            if jsd is not None:
                try:
                    jsd_vals.append(float(jsd))
                except Exception:
                    pass
            if it.get("decision") == "alert":
                alerts += 1
            if it.get("n_frag", 0) == 0:
                no_data += 1
            # H and HHI may be present
            h = it.get("H")
            hh = it.get("HHI")
            if h is not None:
                try:
                    hs.append(float(h))
                except Exception:
                    pass
            if hh is not None:
                try:
                    hhis.append(float(hh))
                except Exception:
                    pass
        mean_jsd = statistics.mean(jsd_vals) if jsd_vals else None
        p95_jsd = (sorted(jsd_vals)[max(0,min(len(jsd_vals)-1, math.ceil(0.95*len(jsd_vals))-1))] if jsd_vals else None)
        pct_alerts = (alerts / n * 100.0) if n>0 else None
        h_mean = statistics.mean(hs) if hs else None
        hhi_mean = statistics.mean(hhis) if hhis else None
        out[cfg] = {
            "n": n,
            "n_no_data": no_data,
            "mean_jsd": mean_jsd,
            "p95_jsd": p95_jsd,
            "alerts": alerts,
            "pct_alerts": pct_alerts,
            "H_mean": h_mean,
            "HHI_mean": hhi_mean
        }
    return out

def aggregate_partial_or_omit(summary_dict, kind_prefix):
    """
    Aggrega (media simple) sobre claves que empiezan por kind_prefix, p.ej. 'partial_' o 'omit_'
    Devuelve dict con aggregated metrics.
    """
    keys = [k for k in summary_dict.keys() if k.startswith(kind_prefix)]
    if not keys:
        return None
    vals_mean_jsd = [summary_dict[k]["mean_jsd"] for k in keys if summary_dict[k]["mean_jsd"] is not None]
    alerts = sum(summary_dict[k]["alerts"] for k in keys)
    total_n = sum(summary_dict[k]["n"] for k in keys)
    pct_alerts = (alerts / total_n * 100.0) if total_n>0 else None
    mean_jsd_overall = statistics.mean(vals_mean_jsd) if vals_mean_jsd else None
    mean_H = statistics.mean([summary_dict[k]["H_mean"] for k in keys if summary_dict[k]["H_mean"] is not None]) if keys else None
    mean_HHI = statistics.mean([summary_dict[k]["HHI_mean"] for k in keys if summary_dict[k]["HHI_mean"] is not None]) if keys else None
    return {
        "keys": keys,
        "mean_jsd": mean_jsd_overall,
        "pct_alerts": pct_alerts,
        "mean_H": mean_H,
        "mean_HHI": mean_HHI,
        "total_n": total_n
    }

def load_json_safe(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_p_from_args_or_path(jobj, path):
    # Try jobj["args"]["degrade_p"], could be "0.5" or "0.5,..." -> keep first
    args = jobj.get("args", {}) if isinstance(jobj, dict) else {}
    dp = args.get("degrade_p")
    if dp is not None:
        # allow formats like "0.5" or "0.5,..." or numeric
        if isinstance(dp, (float,int)):
            return float(dp)
        if isinstance(dp, str):
            first = dp.split(",")[0].strip()
            try:
                return float(first)
            except Exception:
                pass
    # fallback: try folder name containing 'testQ' and a mapping:
    p = None
    try:
        # parent folder name
        pstr = Path(path).parent.name
        # if name contains a number like testQ1, testQ2, try to map by index (user may expect mapping)
        import re
        m = re.search(r"testQ(\d+)", pstr, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            # default mapping: testQ1->0.25, testQ2->0.5, testQ3->0.75
            mapping = {1:0.25, 2:0.5, 3:0.75}
            if idx in mapping:
                return mapping[idx]
        # else, try to parse any float inside folder name
        m2 = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", pstr)
        if m2:
            try:
                return float(m2.group(0))
            except:
                pass
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Genera figuras a partir de JSONs structurals (recursivo).")
    parser.add_argument("--input-dir", "-i", required=True, help="Directorio raíz donde buscar (recursivo) los JSONs.")
    parser.add_argument("--outdir", "-o", default="figs", help="Directorio de salida para figuras.")
    parser.add_argument("--pattern", default="structural_double_end2end_*.json", help="Patrón de nombre de JSON a buscar.")
    parser.add_argument("--save-csv", action="store_true", help="Guardar un CSV resumen con métricas por archivo.")
    args = parser.parse_args()

    files = find_json_files(args.input_dir)
    if not files:
        print(f"[ERROR] No se encontraron JSONs en {args.input_dir} con patrón 'structural_double_end2end_*.json' (busqueda recursiva).", file=sys.stderr)
        print("Revisa que los archivos estén en subcarpetas como testQ1/structural_double_end2end_*.json", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    # For plotting across p values
    collect = {}
    # collect hist data for pooled_jsd and omit_jsd
    pooled_jsd_by_file = {}
    omit_jsd_by_file = {}

    for f in files:
        try:
            jobj = load_json_safe(f)
        except Exception as e:
            print(f"[WARN] No se pudo leer {f}: {e}", file=sys.stderr)
            continue

        tau_jsd = jobj.get("tau_jsd", jobj.get("tau_jsd", None))
        # test_results is a dict: cfg -> list(items)
        test_results = jobj.get("test_results") or jobj.get("test_results", {})
        summary = summarize_test_results(test_results, tau_jsd)
        # aggregated partial and omit
        agg_partial = aggregate_partial_or_omit(summary, "partial_")
        agg_omit = aggregate_partial_or_omit(summary, "omit_")
        # pooled_jsd from calibration
        pooled_jsd = jobj.get("pooled_jsd") or jobj.get("pooled_jsd_sample") or jobj.get("pooled_jsd_sample", None)
        if pooled_jsd is None:
            pooled_jsd = jobj.get("pooled_jsd_sample", None) or jobj.get("pooled_jsd", None)
        # flatten omit providers' jsd values (concatenate per-config jsd lists if present)
        omit_jsd_list = []
        for k in summary.keys():
            if k.startswith("omit_"):
                # extract raw items to get their providers_jsd if present
                items = test_results.get(k, []) or []
                for it in items:
                    v = it.get("providers_jsd")
                    if v is not None:
                        try:
                            omit_jsd_list.append(float(v))
                        except:
                            pass

        # get degrade p
        pval = infer_p_from_args_or_path(jobj, f)
        # fallback: try to parse from filename
        if pval is None:
            try:
                name = f.name
                import re
                m = re.search(r"(\d\.\d+)", name)
                if m:
                    pval = float(m.group(1))
            except:
                pval = None

        # store per-file summary
        row = {
            "file": str(f),
            "p": pval,
            "tau_jsd": tau_jsd,
            "summary": summary,
            "agg_partial": agg_partial,
            "agg_omit": agg_omit,
            "pooled_jsd": pooled_jsd,
            "omit_jsd_list": omit_jsd_list
        }
        rows.append(row)

        # collect for plots indexed by p
        key_p = pval if pval is not None else str(f)
        if key_p not in collect:
            collect[key_p] = {"partial_means_jsd": [], "partial_pct_alerts": [], "H_means": [], "HHI_means": [], "pooled_jsd": [], "omit_jsd": []}
        if agg_partial:
            if agg_partial.get("mean_jsd") is not None:
                collect[key_p]["partial_means_jsd"].append(agg_partial["mean_jsd"])
            if agg_partial.get("pct_alerts") is not None:
                collect[key_p]["partial_pct_alerts"].append(agg_partial["pct_alerts"])
            if agg_partial.get("mean_H") is not None:
                collect[key_p]["H_means"].append(agg_partial["mean_H"])
            if agg_partial.get("mean_HHI") is not None:
                collect[key_p]["HHI_means"].append(agg_partial["mean_HHI"])
        # pooled_jsd
        if pooled_jsd:
            try:
                collect[key_p]["pooled_jsd"].extend([float(x) for x in pooled_jsd])
            except:
                pass
        # omit
        if omit_jsd_list:
            collect[key_p]["omit_jsd"].extend(omit_jsd_list)

    # Build DataFrame summarizing per-p
    rows_summary = []
    for pkey, d in sorted(collect.items(), key=lambda x: float(x[0]) if isinstance(x[0], (int,float)) or (isinstance(x[0], str) and x[0].replace('.','',1).isdigit()) else x[0]):
        # average across possibly multiple files with same p
        mean_jsd = statistics.mean(d["partial_means_jsd"]) if d["partial_means_jsd"] else None
        mean_pct_alerts = statistics.mean(d["partial_pct_alerts"]) if d["partial_pct_alerts"] else None
        mean_H = statistics.mean(d["H_means"]) if d["H_means"] else None
        mean_HHI = statistics.mean(d["HHI_means"]) if d["HHI_means"] else None
        rows_summary.append({"p": float(pkey) if (isinstance(pkey,(int,float)) or (isinstance(pkey,str) and pkey.replace('.','',1).isdigit())) else pkey,
                             "mean_jsd": mean_jsd,
                             "pct_alerts": mean_pct_alerts,
                             "H_mean": mean_H,
                             "HHI_mean": mean_HHI,
                             "pooled_jsd": d["pooled_jsd"],
                             "omit_jsd": d["omit_jsd"]})

    df = pd.DataFrame(rows_summary)
    df = df.sort_values(by="p")

    # Print a quick textual summary
    print("\nResumen agregado por valor p:")
    print(df[["p","mean_jsd","pct_alerts","H_mean","HHI_mean"]].to_string(index=False))

    if args.save_csv:
        csvp = outdir / "summary_by_p.csv"
        df.to_csv(csvp, index=False)
        print(f"[SAVED] CSV resumen: {csvp}")

    # ---------------------------
    # FIG 1: indicators vs p (H and HHI) - sin titles, ejes en español
    # ---------------------------
    fig1_path = outdir / "fig3_indicators_vs_p.png"
    if not df.empty:
        p_vals = df["p"].astype(float).tolist()
        H_vals = df["H_mean"].tolist()
        HHI_vals = df["HHI_mean"].tolist()
        # plot H (left) and HHI (right axis) - sin título
        fig, ax1 = plt.subplots(figsize=(7,4.5))
        ax1.set_xlabel("Parámetro de degradación $p$")
        ax1.set_ylabel("Entropía estructural (media)", fontsize=10)
        ax1.plot(p_vals, H_vals, marker='o', label="Entropía estructural (media)", linestyle='-')
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()
        ax2.set_ylabel("Índice de concentración HHI (media)", fontsize=10)
        ax2.plot(p_vals, HHI_vals, marker='s', label="HHI (media)", linestyle='--')
        ax2.tick_params(axis='y')
        # construir leyenda combinada (tomando handles de ambos ejes)
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig1_path, dpi=200)
        plt.close(fig)
        print(f"[SAVED] {fig1_path}")
    else:
        print("[WARN] No hay datos para fig3_indicators_vs_p")

    # ---------------------------
    # FIG 2: mean JSD vs p and pct alerts vs p (dos figuras separadas) - ejes en español, sin título
    # ---------------------------
    fig2a = outdir / "fig2a_mean_jsd_vs_p.png"
    fig2b = outdir / "fig2b_pct_alerts_vs_p.png"
    if not df.empty:
        x = df["p"].astype(float).tolist()
        y_jsd = df["mean_jsd"].tolist()
        y_pct = df["pct_alerts"].tolist()

        # media JSD vs p (eje vertical descriptivo)
        fig = plt.figure(figsize=(6,4))
        plt.plot(x, y_jsd, marker='o')
        plt.xlabel("Parámetro de degradación $p$")
        plt.ylabel("Media de divergencia estructural (JSD)")
        plt.tight_layout()
        fig.savefig(fig2a, dpi=200)
        plt.close(fig)
        print(f"[SAVED] {fig2a}")

        # pct activaciones vs p
        fig = plt.figure(figsize=(6,4))
        plt.plot(x, y_pct, marker='o')
        plt.xlabel("Parámetro de degradación $p$")
        plt.ylabel("Consultas con activación (JSD > τ) [%]")
        plt.tight_layout()
        fig.savefig(fig2b, dpi=200)
        plt.close(fig)
        print(f"[SAVED] {fig2b}")
    else:
        print("[WARN] No hay datos para fig2a/fig2b")


    print("\nHecho. Figuras guardadas en:", outdir)

if __name__ == "__main__":
    main()
