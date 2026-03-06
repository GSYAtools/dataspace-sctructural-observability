#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_hist_jsd.py
Genera histograma comparativo (Calibración vs Omisión total) en % por bin,
promediando por fichero para que cada ejecución pese igual (evita que B=5000 domine).

Uso:
  python plot_hist_jsd.py --input-dir . --out figures --bins 30

Si --out es un directorio, el fichero resultante será OUT/hist_jsd_by_cfg.png
"""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import statistics
import sys

def find_json_files(input_dir):
    p = Path(input_dir)
    if not p.exists():
        raise SystemExit(f"No existe el directorio {input_dir}")
    files = list(p.rglob("*.json"))
    return [f for f in files if f.is_file()]

def load_jsd_lists(json_path):
    """Extrae pooled_jsd (calibración) y omit_jsd_list (lista de JSD bajo omit total).
       Si el JSON no es un dict (p. ej. lista top-level) devuelve (None,None) y lo ignora.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        print(f"[WARN] fallo leyendo {json_path}: {e}", file=sys.stderr)
        return None, None

    if not isinstance(j, dict):
        # fichero tipo queries.json u otro; no es un resultado de ejecución -> ignorar
        # Opcional: puedes listar cuáles son estos ficheros
        # print(f"[DBG] Ignorando JSON no-dict (probablemente no es output) {json_path}")
        return None, None

    # pooled_jsd puede estar guardado en distintos campos según versión
    pooled = j.get("pooled_jsd_sample") if "pooled_jsd_sample" in j else j.get("pooled_jsd") if "pooled_jsd" in j else j.get("pooled_jsd_sample", [])
    if pooled is None:
        pooled = []
    # fallback: algunos dumps podrían tener 'pooled_jsd' under different key - try a few possibilities
    if not pooled and "pooled_jsd_stored" in j:
        pooled = j.get("pooled_jsd_stored", [])

    pooled_list = []
    try:
        if isinstance(pooled, list):
            pooled_list = [float(x) for x in pooled]
    except Exception:
        pooled_list = []

    omit_list = []
    # Intentar extraer omit_* desde test_results
    tr = j.get("test_results") or {}
    if isinstance(tr, dict):
        for k, v in tr.items():
            if isinstance(k, str) and k.startswith("omit_"):
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            jsd = it.get("providers_jsd")
                            if jsd is not None:
                                try:
                                    omit_list.append(float(jsd))
                                except Exception:
                                    pass
    # Alternativas directas en el JSON (compatibilidad)
    alt_omit = j.get("omit_jsd_list") or j.get("omit_jsd") or None
    if alt_omit:
        try:
            if isinstance(alt_omit, list):
                omit_list.extend([float(x) for x in alt_omit])
        except Exception:
            pass

    return pooled_list, omit_list

def hist_percent(values, bins):
    """Retorna porcentaje por bin (suma ≈ 100) y bin edges.
       Si values está vacío devuelve zeros."""
    if not values:
        counts = np.zeros(len(bins)-1, dtype=float)
        return counts, bins
    counts, edges = np.histogram(values, bins=bins)
    n = counts.sum()
    if n > 0:
        pct = counts / float(n) * 100.0
    else:
        pct = np.zeros_like(counts, dtype=float)
    return pct, edges

def mean_pct(list_of_arrays):
    if not list_of_arrays:
        return None
    arr = np.vstack(list_of_arrays)
    return arr.mean(axis=0)

def main():
    parser = argparse.ArgumentParser(description="Genera histograma comparativo JSD: calibración vs omisión total (promediado por ejecución).")
    parser.add_argument("--input-dir", required=True, help="Directorio raíz donde buscar JSONs (recursivo).")
    parser.add_argument("--out", default="hist_jsd_by_cfg.png", help="Ruta de salida (archivo o directorio).")
    parser.add_argument("--bins", type=int, default=30, help="Número de bins para los histogramas.")
    parser.add_argument("--range", type=float, nargs=2, default=None, help="Rango x (min max). Si no dado se infiere del max observado.")
    args = parser.parse_args()

    files = find_json_files(args.input_dir)
    if not files:
        raise SystemExit(f"No se encontraron JSONs en {args.input_dir}")

    # Determinar rango x (si no provisto) buscando max en pooled/omit
    maxval = 0.0
    any_data = False
    for f in files:
        pooled, omit = load_jsd_lists(f)
        if pooled:
            any_data = True
            maxval = max(maxval, max(pooled))
        if omit:
            any_data = True
            maxval = max(maxval, max(omit))
    if not any_data:
        raise SystemExit("No se encontraron listas pooled_jsd ni omit_jsd en los JSONs. Asegúrate de pasar la carpeta con resultados.")

    if args.range:
        xmin, xmax = args.range
    else:
        xmax = max(maxval, 0.05)
        xmin = 0.0

    bins = np.linspace(xmin, xmax, args.bins + 1)

    per_file_pcts_pooled = []
    per_file_pcts_omit = []
    n_pooled_files = 0
    n_omit_files = 0

    for f in files:
        pooled, omit = load_jsd_lists(f)
        if pooled:
            pct_pooled, _ = hist_percent(pooled, bins)
            per_file_pcts_pooled.append(pct_pooled)
            n_pooled_files += 1
        if omit:
            pct_omit, _ = hist_percent(omit, bins)
            per_file_pcts_omit.append(pct_omit)
            n_omit_files += 1

    if not per_file_pcts_pooled and not per_file_pcts_omit:
        raise SystemExit("No hay datos válidos extraídos de los JSONs (pooled_jsd / omit_jsd).")

    mean_pooled_pct = mean_pct(per_file_pcts_pooled)
    mean_omit_pct = mean_pct(per_file_pcts_omit)

    centers = (bins[:-1] + bins[1:]) / 2.0
    width = (bins[1] - bins[0]) * 0.9

    plt.figure(figsize=(8,4.5))
    if mean_pooled_pct is not None:
        plt.bar(centers, mean_pooled_pct, width=width, alpha=0.6, label="Calibración (media por ejecución)", align='center', edgecolor='k')
    if mean_omit_pct is not None:
        plt.bar(centers, mean_omit_pct, width=width, alpha=0.6, label="Omisión total de proveedor (media por ejecución)", align='center', edgecolor='k')

    plt.xlabel("Divergencia estructural (JSD)")
    plt.ylabel("Porcentaje de observaciones por bin (%)")
    plt.legend(frameon=True)
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    outp = Path(args.out)
    if outp.is_dir() or str(outp).endswith(("/", "\\")):
        outp = outp / "hist_jsd_by_cfg.png"
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, dpi=150)
    plt.close()
    print(f"[SAVED] {outp}")
    # resumen informativo
    print(f"[INFO] JSONs analizados: {len(files)}; con pooled_jsd: {n_pooled_files}; con omit_jsd: {n_omit_files}")
    print(f"[INFO] bins: {len(bins)-1}, rango x: [{xmin:.6f}, {xmax:.6f}] (cada fichero normalizado a 100%)")

if __name__ == "__main__":
    main()