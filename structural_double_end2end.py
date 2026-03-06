#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structural_double_end2end.py
End-to-end: calibración original (JSD bootstrap OOB) + representación doble φ (providers + scalars)
- retrieve inlined (Faiss optional)
- conserva exactamente la lógica de calibración original que me pasaste
- añade cálculo mu/sigma de phi y bootstrap para tau_delta (Delta_norm)
- guarda JSON, CSV y figuras (matplotlib)
"""
import os
import sys
import time
import json
import math
import random
import argparse
import statistics
import csv
import pickle
from pathlib import Path
from collections import Counter, defaultdict

# plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# optional KS test
try:
    from scipy.stats import ks_2samp
except Exception:
    ks_2samp = None

# Try import faiss and numpy for vector search
HAS_FAISS = True
try:
    import faiss
    import numpy as np
except Exception:
    HAS_FAISS = False
    faiss = None
    np = None

# ------------------ Basic utilities (entropy, HHI, redundancy) ------------------ #

def shannon_entropy_from_counts(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p)
    return float(ent)

def effective_N_from_entropy(entropy):
    try:
        return float(math.exp(entropy))
    except OverflowError:
        return float("inf")

def hhi_from_counts(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(sum((v / total) ** 2 for v in counts.values()))

def compute_redundancy(fragments):
    texts = [fr.get("text", "").strip() for fr in fragments]
    counts = Counter(texts)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return int(dup)

# ------------------ Jensen-Shannon divergence on dicts ------------------ #

def jensen_shannon_divergence(p_dict, q_dict):
    keys = list(set(p_dict.keys()) | set(q_dict.keys()))
    if not keys:
        return 0.0
    P = []
    Q = []
    for k in keys:
        P.append(float(p_dict.get(k, 0.0)))
        Q.append(float(q_dict.get(k, 0.0)))
    sP = sum(P) or 1.0
    sQ = sum(Q) or 1.0
    P = [x/sP for x in P]
    Q = [x/sQ for x in Q]
    M = [(p_i + q_i)/2.0 for p_i, q_i in zip(P, Q)]
    def kl(a,b):
        s = 0.0
        for ai, bi in zip(a,b):
            if ai > 0 and bi > 0:
                s += ai * math.log(ai/bi)
        return s
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)

# ------------------ FaissIndexCache and retrieval (inlined) ------------------ #

class FaissIndexCache:
    def __init__(self):
        self.cache = {}

    def load(self, dp_path):
        dp_path = Path(dp_path)
        dp_name = dp_path.name
        if dp_name in self.cache:
            return self.cache[dp_name]
        idx_file = dp_path / "index.faiss"
        meta_file = dp_path / "index.pkl"
        if not idx_file.exists():
            alt = dp_path / "faiss.index"
            if alt.exists():
                idx_file = alt
        if not idx_file.exists():
            self.cache[dp_name] = {"index": None, "meta": None, "metric_type": None}
            return self.cache[dp_name]
        if faiss is None:
            raise RuntimeError("faiss no está disponible en el entorno. Instala faiss-cpu para usar búsqueda vectorial local.")
        try:
            index = faiss.read_index(str(idx_file))
        except Exception:
            index = None
        metric_type = None
        try:
            if index is not None:
                metric_type = int(getattr(index, "metric_type", None))
        except Exception:
            metric_type = None
        meta = None
        if meta_file.exists():
            try:
                with open(meta_file, "rb") as f:
                    meta = pickle.load(f)
            except Exception:
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    meta = None
        self.cache[dp_name] = {"index": index, "meta": meta, "metric_type": metric_type}
        return self.cache[dp_name]

    def search(self, dp_name, vector, k=3, dp_dir_base=None):
        info = self.cache.get(dp_name)
        if info is None or info.get("index") is None:
            if dp_dir_base:
                self.load(Path(dp_dir_base) / dp_name)
                info = self.cache.get(dp_name)
        idx = info.get("index") if info else None
        meta = info.get("meta") if info else None
        if idx is None:
            return []
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if k <= 0:
            k = 1
        try:
            D, I = idx.search(vec, k)
        except Exception:
            return []
        results = []
        for dist, idx_pos in zip(D[0], I[0]):
            if idx_pos < 0:
                continue
            text = ""
            md = {}
            if isinstance(meta, list) and idx_pos < len(meta):
                m = meta[idx_pos]
                if isinstance(m, dict):
                    text = m.get("page_content") or m.get("text") or m.get("document") or ""
                    md = m
                else:
                    text = str(m)
                    md = {"source": None}
            else:
                text = f"<doc_{idx_pos}>"
                md = {"index_pos": int(idx_pos)}
            results.append({
                "text": text,
                "doc_id": md.get("source", "") or str(md.get("index_pos", idx_pos)),
                "score": float(dist),
                "source": dp_name,
                "metadata": md
            })
        return results

def retrieve_from_indexes(indexes_dir, query_obj, per_dp_topk, dp_list,
                          embeddings_map=None, faiss_cache=None,
                          max_search_k=300, near_threshold_dp=None,
                          near_threshold_global=None,
                          use_partial_k_map=False, partial_k_map=None):
    """
    Compatibilidad con la implementación anterior:
      - per_dp_topk: dict con "__default__" y opcionales dp-specific (k por DP)
      - near_threshold_dp: dict dp->thr (o None) ; near_threshold_global: fallback
      - use_partial_k_map: si True, ignora per_dp_topk y aplica truncado por proveedor
        usando partial_k_map (dict similar a per_dp_topk pero que significa cuota
        por proveedor antes de recomponer globalmente).
    Devuelve lista de fragments ordenada (mixed-metric warning si corresponde).
    """
    fragments = []
    base = Path(indexes_dir)
    default_k = int(per_dp_topk.get("__default__", 3)) if per_dp_topk is not None else 3
    qid = query_obj.get("id")
    use_vector = (embeddings_map is not None and qid in embeddings_map and HAS_FAISS)

    if use_vector and faiss_cache is None:
        faiss_cache = FaissIndexCache()

    for dp in dp_list:
        idx_path = base / dp
        if not idx_path.exists() or not idx_path.is_dir():
            print(f"[DBG RET] dp={dp} index missing, skipping")
            continue

        try:
            if use_vector:
                vec = embeddings_map[qid]
                raw_k = max_search_k if max_search_k else default_k
                dp_results = faiss_cache.search(dp, vec, k=raw_k, dp_dir_base=indexes_dir) or []

                # limpieza: filtrar scores inválidos
                clean = []
                for r in dp_results:
                    try:
                        sc = float(r.get("score", 0.0))
                        if not (math.isnan(sc) or math.isinf(sc)):
                            clean.append(r)
                    except Exception:
                        # descartamos elementos sin score numérico
                        continue

                # -------------------------
                # FILTRADO POR UMBRAL (near_threshold_dp / global)
                # -------------------------
                thr = None
                if near_threshold_dp and dp in near_threshold_dp and near_threshold_dp[dp] is not None:
                    thr = near_threshold_dp[dp]
                else:
                    thr = near_threshold_global

                dp_info = faiss_cache.cache.get(dp, {}) if faiss_cache else {}
                metric_type = dp_info.get("metric_type", None)

                filtered = []
                if thr is None:
                    filtered = clean[:]
                else:
                    is_l2 = (metric_type is not None and faiss is not None and metric_type == getattr(faiss, "METRIC_L2", 0))
                    for r in clean:
                        try:
                            sc = float(r.get("score", 0.0))
                        except Exception:
                            continue
                        if is_l2:
                            # L2: menor = mejor
                            if sc <= thr:
                                filtered.append(r)
                        else:
                            # IP/otros: mayor = mejor
                            if sc >= thr:
                                filtered.append(r)

                # -------------------------
                # ORDENADO por calidad según metric_type
                # -------------------------
                try:
                    if metric_type is not None and faiss is not None and metric_type == getattr(faiss, "METRIC_L2", 0):
                        filtered_sorted = sorted(filtered, key=lambda x: float(x.get("score", 0.0)))
                        sort_dir = "asc(L2)"
                    else:
                        filtered_sorted = sorted(filtered, key=lambda x: float(x.get("score", 0.0)), reverse=True)
                        sort_dir = "desc(IP/other)"
                except Exception as e:
                    print(f"[DBG RET] dp={dp} sort_error={e} — manteniendo orden sin sort")
                    filtered_sorted = filtered[:]
                    sort_dir = "none"

                # -------------------------
                # TRUNCADO/OMITIDO POR per_dp_topk o partial_k_map (si aplica)
                # -------------------------
                if use_partial_k_map:
                    # aplicar lógica de truncado por proveedor:
                    # - agrupar por 'source' (aunque aquí todos son del mismo dp), ordenar cada grupo por score
                    # - truncar acuerdo a partial_k_map[dp] o partial_k_map["__default__"]
                    # - recomponer y ordenar globalmente por score
                    if partial_k_map is None:
                        # fallback: reducir a max(1, default_k//3)
                        cut_k = max(1, default_k // 3)
                    else:
                        cut_k = int(partial_k_map.get(dp, partial_k_map.get("__default__", max(1, default_k//3))))
                    # en nuestro caso 'filtered_sorted' viene sólo de un dp; truncar directamente:
                    if cut_k == 0:
                        if len(filtered_sorted) > 0:
                            print(f"[DBG RET] dp={dp} omitted by partial_k_map (found {len(filtered_sorted)} hits, applied_k=0)")
                        continue
                    if len(filtered_sorted) > cut_k:
                        filtered_sorted = filtered_sorted[:cut_k]
                else:
                    # comportamiento original: per_dp_topk
                    try:
                        dp_k = int(per_dp_topk.get(dp, default_k)) if per_dp_topk is not None else default_k
                    except Exception:
                        dp_k = default_k

                    if dp_k == 0:
                        if len(filtered_sorted) > 0:
                            print(f"[DBG RET] dp={dp} omitted by per_dp_topk (found {len(filtered_sorted)} hits, applied_k=0)")
                        continue

                    if dp_k is not None and dp_k > 0 and len(filtered_sorted) > dp_k:
                        filtered_sorted = filtered_sorted[:dp_k]
                              
                fragments.extend(filtered_sorted)

        except Exception as e:
            print(f"[WARN] fallo retrieval {dp}: {e}", file=sys.stderr)

    # finalmente, ordenar todos los fragments combinados. Atención: se mezclan métricas L2/IP tal como antes.
    # Si quieres comparabilidad entre DPs con métricas distintas, normaliza scores por DP antes de mezclar.
    try:
        fragments_sorted = sorted(fragments, key=lambda x: x.get("score", 0), reverse=True)
    except Exception:
        fragments_sorted = fragments

    return fragments_sorted

# ------------------ Embeddings loader ------------------ #

def load_embeddings_map(embeddings_file):
    with open(embeddings_file, "r", encoding="utf-8") as f:
        arr = json.load(f)
    qlist = []
    emb_map = {}
    for i, item in enumerate(arr):
        if isinstance(item, dict):
            qid = item.get("id") or item.get("qid") or f"Q{i+1:04d}"
            qtext = item.get("text") or item.get("query") or ""
            emb = item.get("embedding")
            calib = item.get("calibration_set", False)
            qlist.append({"id": str(qid), "text": qtext, "calibration_set": calib})
            if emb is not None:
                emb_map[str(qid)] = emb
        elif isinstance(item, str):
            qid = f"Q{i+1:04d}"
            qlist.append({"id": qid, "text": item, "calibration_set": False})
        else:
            qid = f"Q{i+1:04d}"
            qlist.append({"id": qid, "text": str(item), "calibration_set": False})
    return qlist, emb_map

# ------------------ φ representation builder (from fragments) ------------------ #

def build_phi_from_fragments(dist_dict, counts_dict, n_frag, redundancy, latency_ms, dp_list):
    """
    Orden φ:
      [p_1,...,p_m, H, HHI, N_eff, n_frag, redundancy, latency_ms]
    """
    probs = [float(dist_dict.get(dp, 0.0)) for dp in dp_list]
    H = shannon_entropy_from_counts(Counter(counts_dict))
    HHI_v = hhi_from_counts(Counter(counts_dict))
    N_eff = effective_N_from_entropy(H)
    n_frag_v = float(n_frag)
    redundancy_v = float(redundancy)
    latency_v = float(latency_ms)
    vec = probs + [H, HHI_v, N_eff, n_frag_v, redundancy_v, latency_v]
    if np is None:
        return vec
    return np.asarray(vec, dtype=np.float64)

# ------------------ mu/sigma and per-query delta ------------------ #

def compute_mu_sigma_from_phi(phi_list):
    if not phi_list:
        return None, None
    if np is None:
        k = len(phi_list[0])
        arr = [[float(x[i]) for x in phi_list] for i in range(k)]
        mu = [sum(col)/len(col) for col in arr]
        sigma = []
        for col in arr:
            m = sum(col)/len(col)
            var = sum((v - m)**2 for v in col) / (len(col)-1) if len(col) > 1 else 0.0
            sigma.append(math.sqrt(var))
        sigma = [s if s > 0 else 1e-9 for s in sigma]
        return mu, sigma
    arr = np.vstack([np.asarray(v, dtype=np.float64) for v in phi_list])
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0, ddof=1)
    eps = 1e-9
    sigma_safe = np.where(sigma > 0.0, sigma, eps)
    return mu, sigma_safe

def per_query_vector_and_deltas(phi_vec, mu, sigma):
    if phi_vec is None or mu is None or sigma is None:
        return {"phi": None, "delta_vec": None, "delta_norm_vec": None, "Delta_norm": None}
    if np is None:
        delta = [float(a - b) for a, b in zip(phi_vec, mu)]
        delta_norm = [d / (s if s != 0 else 1e-9) for d, s in zip(delta, sigma)]
        Delta_norm = math.sqrt(sum(x*x for x in delta_norm))
        return {"phi": list(phi_vec), "delta_vec": delta, "delta_norm_vec": delta_norm, "Delta_norm": float(Delta_norm)}
    delta = phi_vec - mu
    delta_norm = delta / sigma
    Delta_norm = float(np.linalg.norm(delta_norm))
    return {"phi": phi_vec.tolist(), "delta_vec": delta.tolist(), "delta_norm_vec": delta_norm.tolist(), "Delta_norm": Delta_norm}

# ------------------ scalar histogram helpers (kept) ------------------ #

def compute_bins_for_scalar(values, bins=20, vmin=None, vmax=None):
    if np is not None:
        if vmin is None or vmax is None:
            if len(values):
                vmin = float(np.min(values))
                vmax = float(np.max(values))
            else:
                vmin = 0.0
                vmax = 1.0
            if vmin == vmax:
                vmax = vmin + 1.0
        edges = np.linspace(vmin, vmax, bins+1)
        return edges
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    step = (vmax - vmin) / bins
    edges = [vmin + i*step for i in range(bins+1)]
    return edges

def scalar_value_to_hist(value, bin_edges, eps=1e-9):
    n_bins = len(bin_edges) - 1
    vec = [0.0]*n_bins
    for i in range(n_bins):
        if (value >= bin_edges[i] and value < bin_edges[i+1]) or (i == n_bins-1 and value == bin_edges[-1]):
            vec[i] = 1.0
            break
    s = sum(vec)
    if s == 0:
        vec = [1.0/n_bins]*n_bins
    vec = [(v + eps) for v in vec]
    s2 = sum(vec)
    return [v/s2 for v in vec]

# ------------------ CLI / main (calibration preserved EXACTLY) ------------------ #

def main():
    parser = argparse.ArgumentParser(description="Runner estructural DOBLE (JSD + φ + bootstrap para JSD y Delta_norm).")
    parser.add_argument("--queries", required=True, help="Path a queries JSON (con calibration_set flag)")
    parser.add_argument("--embeddings", required=False, help="(Opcional) Path a JSON de queries con campo 'id' y 'embedding' para búsqueda vectorial")
    parser.add_argument("--indexes", default="indexes/federated", help="Directorio base de índices (subcarpetas por DP)")
    parser.add_argument("--out", default="out", help="Directorio de salida")
    parser.add_argument("--topk", type=int, default=3, help="Top-k por defecto por DP")
    parser.add_argument("--n_test", type=int, default=60, help="Número de queries de test a muestrear")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo")
    parser.add_argument("--bootstrap_B", "--B", type=int, default=5000, help="Número de réplicas bootstrap para calibración (por defecto 5000)")
    parser.add_argument("--p_threshold", type=float, default=0.95, help="Percentil para umbral bootstrap (por defecto 0.95)")
    parser.add_argument("--m", type=int, default=2, help="Número de componentes excedidas para declarar 'alert' (no usado aquí)")
    parser.add_argument("--sample_mode", choices=["same","different"], default="same")
    parser.add_argument("--ks", action="store_true", help="Si se dispone de scipy, ejecutar KS test")
    parser.add_argument("--calib_sample", type=int, default=None)
    parser.add_argument("--max_search_k", type=int, default=300)
    parser.add_argument("--near_threshold_global", type=float, default=None)
    parser.add_argument("--N_calib", type=int, default=60)
    parser.add_argument("--p_for_dp", type=int, default=10)
    parser.add_argument("--partial_k_map", type=str, default=None)
    parser.add_argument("--no_data_action", choices=["ignore","alert","fallback"], default="ignore")
    parser.add_argument("--debug_ids", type=str, default=None)
    parser.add_argument("--report_topk_l1", type=float, default=0.2)
    parser.add_argument("--phi_boot_B", type=int, default=2000, help="Bootstrap B for phi delta_norm threshold (smaller)")
    # único parámetro de degradación solicitado (drop por fragment)
    parser.add_argument("--degrade_p", type=float, default=0.0,
                        help="Probabilidad (0..1) de descartar cada fragment proveniente del proveedor degradado (aplica solo en experiments 'partial').")
    args = parser.parse_args()

    # ---------- load queries ----------
    with open(args.queries, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)

    # embeddings optional
    embeddings_map = None
    embeddings_list = None
    faiss_cache = None
    if args.embeddings:
        if not HAS_FAISS:
            raise SystemExit("Para usar --embeddings necesitas faiss instalado (faiss-cpu).")
        embeddings_list, embeddings_map = load_embeddings_map(args.embeddings)
        faiss_cache = FaissIndexCache()
        print(f"[INFO] embeddings loaded: {len(embeddings_map)} vectors")

    # normalize queries_raw to list of dicts
    queries = []
    for i, item in enumerate(queries_raw):
        if isinstance(item, dict):
            qid = item.get("id") or item.get("qid") or f"Q{i+1:04d}"
            text = item.get("text") or item.get("query") or ""
            calib = item.get("calibration_set", False)
            queries.append({"id": str(qid), "text": text, "calibration_set": calib})
        elif isinstance(item, str):
            qid = f"Q{i+1:04d}"
            queries.append({"id": qid, "text": item, "calibration_set": False})
        else:
            qid = f"Q{i+1:04d}"
            queries.append({"id": qid, "text": str(item), "calibration_set": False})

    calib_qs_all = [q for q in queries if q.get("calibration_set")]
    if not calib_qs_all:
        raise SystemExit("No se encontraron queries de calibración (calibration_set=true) en el JSON.")

    # Deterministic: take first N_calib from file order
    N_calib = int(args.N_calib)
    calib_qs = calib_qs_all[:min(N_calib, len(calib_qs_all))]
    if args.calib_sample is not None:
        k = int(args.calib_sample)
        if k <= 0:
            raise SystemExit("calib_sample debe ser > 0")
        if k > len(calib_qs):
            raise SystemExit(f"calib_sample ({k}) > número de queries de calibración disponibles ({len(calib_qs)})")
        random.seed(args.seed)
        calib_qs = random.sample(calib_qs, k)
        print(f"[INFO] Muestreo aleatorio de calibración: seleccionado {k} queries (seed={args.seed}).")
    pool_qs = [q for q in queries if not q.get("calibration_set")]

    if not calib_qs:
        raise SystemExit("No se encontraron queries de calibración tras aplicar selección.")
    if not pool_qs:
        raise SystemExit("No quedan queries para test (todas están marcadas como calibración).")

    print(f"Calibración: {len(calib_qs)} queries. Pool de test: {len(pool_qs)} queries.")

    # dp_list desde indexes dir
    idx_base = Path(args.indexes)
    if not idx_base.exists():
        raise SystemExit(f"Directorio de índices no existe: {args.indexes}")
    dp_list = [p.name for p in sorted(idx_base.iterdir()) if p.is_dir()]
    if not dp_list:
        raise SystemExit(f"No se encontraron subcarpetas de DP en {args.indexes}")
    print("DPs detectados:", dp_list)

    # Prepare per-dp scores collector (to compute per-DP near thresholds)
    per_dp_scores = {dp: [] for dp in dp_list}

    # -----------------------
    # 1) CALIBRACIÓN - PASO A: recolectar scores por DP (sin aplicar thresholds)
    # -----------------------
    print("[STEP] Calibración (PASO A): recolectando scores por DP (sin thresholds)...")
    per_query_meta = []
    for idx, q in enumerate(calib_qs):
        qid = q.get("id")
        qtext = q.get("text")
        qobj = {"id": qid, "text": qtext}
        start = time.time()
        frags = retrieve_from_indexes(args.indexes, qobj, {"__default__": args.topk}, dp_list,
                                      embeddings_map=embeddings_map, faiss_cache=faiss_cache,
                                      max_search_k=args.max_search_k,
                                      near_threshold_dp=None,
                                      near_threshold_global=None)
        end = time.time()
        latency_ms = (end - start) * 1000.0

        for fr in frags:
            src = fr.get("source")
            sc = fr.get("score")
            try:
                if src in per_dp_scores and sc is not None:
                    s = float(sc)
                    if not (math.isnan(s) or math.isinf(s)):
                        per_dp_scores[src].append(s)
            except Exception:
                pass

        counts = Counter(fr.get("source", "unknown") for fr in frags)
        dist_counts = {dp: counts.get(dp, 0) for dp in dp_list}
        total = sum(dist_counts.values())
        normalized = {dp: (dist_counts[dp]/total) if total>0 else 0.0 for dp in dp_list}
        redundancy = compute_redundancy(frags)
        H = shannon_entropy_from_counts(counts)
        N_eff = effective_N_from_entropy(H)
        HHI_v = hhi_from_counts(counts)
        per_item = {
            "id": qid,
            "query": qtext,
            "pos": idx,
            "counts_by_dp": dict(dist_counts),
            "dist": dict(normalized),
            "n_frag": len(frags),
            "redundancy": redundancy,
            "H": H,
            "N_eff": N_eff,
            "HHI": HHI_v,
            "latency_ms": round(latency_ms, 2)
        }
        per_query_meta.append(per_item)

    # 1b) calcular near_threshold_dp (percentil p_for_dp)
    p_for_dp = int(args.p_for_dp)
    near_threshold_dp = {}
    for dp in dp_list:
        scores = per_dp_scores.get(dp, []) or []
        if not scores:
            near_threshold_dp[dp] = None
            continue
        try:
            import numpy as _np
            thr = float(_np.percentile(_np.array(scores), p_for_dp))
        except Exception:
            vals = sorted(scores)
            idxp = max(0, min(len(vals)-1, math.ceil((p_for_dp/100.0)*len(vals)) - 1))
            thr = float(vals[idxp])
        near_threshold_dp[dp] = thr

    print("[INFO] near_threshold per DP (p{}):".format(p_for_dp))
    for dp, thr in near_threshold_dp.items():
        print(f" - {dp}: {thr}")

    # ---------- JSD-based calibration (PASO C) - apply per-dp thresholds to build filtered calibration set ----------
    print("[STEP] Calibración (PASO C): reconstruyendo per_query_meta aplicando near_threshold_dp...")
    per_query_meta_filtered = []
    phi_list = []
    for idx, q in enumerate(calib_qs):
        qid = q.get("id")
        qtext = q.get("text")
        qobj = {"id": qid, "text": qtext}

        frags = retrieve_from_indexes(args.indexes, qobj, {"__default__": args.topk}, dp_list,
                                      embeddings_map=embeddings_map, faiss_cache=faiss_cache,
                                      max_search_k=args.max_search_k,
                                      near_threshold_dp=near_threshold_dp,
                                      near_threshold_global=args.near_threshold_global)

        counts = Counter(fr.get("source", "unknown") for fr in frags)
        dist_counts = {dp: counts.get(dp, 0) for dp in dp_list}
        total = sum(dist_counts.values())
        normalized = {dp: (dist_counts[dp]/total) if total>0 else 0.0 for dp in dp_list}
        redundancy = compute_redundancy(frags)
        H = shannon_entropy_from_counts(counts)
        N_eff = effective_N_from_entropy(H)
        HHI_v = hhi_from_counts(counts)
        latency_ms = 0.0
        per_item = {
            "id": qid,
            "query": qtext,
            "pos": idx,
            "counts_by_dp": dict(dist_counts),
            "dist": dict(normalized),
            "n_frag": len(frags),
            "redundancy": redundancy,
            "H": H,
            "N_eff": N_eff,
            "HHI": HHI_v,
            "latency_ms": round(latency_ms, 2)
        }
        per_query_meta_filtered.append(per_item)
        # build phi for this item (order preserved)
        phi = build_phi_from_fragments(per_item["dist"], per_item["counts_by_dp"], per_item["n_frag"],
                                       per_item["redundancy"], per_item["latency_ms"], dp_list)
        phi_list.append(phi)

        if idx < 10:
            print(f"[DEBUG CALIB] idx={idx} id={qid} n_frag={per_item['n_frag']} dist={per_item['dist']}")

    zeros_after = sum(1 for it in per_query_meta_filtered if (it.get("n_frag") or 0) == 0)
    print(f"[INFO] queries con n_frag == 0 tras aplicar thresholds (filtered out): {zeros_after} / {len(per_query_meta_filtered)}")

    # ---------------------------
    # Bootstrap sobre JSD: construir distribution of JSDs under baseline (compare each v_calib vs global_ref)
    # ---------------------------
    print(f"[STEP] Estimando umbral JSD por bootstrap (B={args.bootstrap_B}) comparando cada vector de calib contra global_ref...")

    def make_vec_from_item(it):
        if "dist" in it and isinstance(it["dist"], dict):
            return [float(it["dist"].get(dp, 0.0)) for dp in dp_list]
        if "counts_by_dp" in it:
            vals = []
            total = sum(it["counts_by_dp"].get(dp, 0) for dp in dp_list)
            if total == 0:
                return [0.0]*len(dp_list)
            for dp in dp_list:
                vals.append(float(it["counts_by_dp"].get(dp, 0))/float(total))
            return vals
        return [0.0]*len(dp_list)

    vecs = [make_vec_from_item(it) for it in per_query_meta_filtered if (it.get("n_frag",0) > 0)]
    N_calib_effective = len(vecs)
    if N_calib_effective == 0:
        raise SystemExit("No hay vectores de calibración con n_frag>0 para estimar JSD bootstrap.")

    # global_ref: mean of calib vectors (used to compute providers_jsd on tests)
    if np is not None:
        arr = np.vstack([np.asarray(v, dtype=np.float64) for v in vecs])
        global_ref = arr.mean(axis=0).tolist()
        s = float(sum(global_ref)) or 1.0
        if s != 0:
            global_ref = [x/s for x in global_ref]
    else:
        agg = [0.0]*len(dp_list)
        for v in vecs:
            for j,val in enumerate(v):
                agg[j] += val
        global_ref = [v/len(vecs) for v in agg]
        s = sum(global_ref) or 1.0
        global_ref = [v/s for v in global_ref]

    print("[DBG] global_ref (rounded):", [round(x,6) for x in global_ref])

    def _jsd_vec(p, q):
        return jensen_shannon_divergence({str(i): p[i] for i in range(len(p))},
                                         {str(i): q[i] for i in range(len(q))})

    rng = random.Random(int(args.seed))
    pooled_jsd = []
    B = int(args.bootstrap_B)

    print(f"[DBG] N_calib_effective={N_calib_effective}, B={B}")
    for b in range(B):
        idxs = [rng.randrange(0, N_calib_effective) for _ in range(N_calib_effective)]
        if np is not None:
            mu_b = np.mean(np.vstack([vecs[i] for i in idxs]), axis=0)
            s = float(mu_b.sum()) or 1.0
            if s != 0:
                mu_b = (mu_b / s).tolist()
            else:
                mu_b = (np.ones_like(mu_b)/len(mu_b)).tolist()
        else:
            agg = [0.0]*len(dp_list)
            for i in idxs:
                for j,val in enumerate(vecs[i]):
                    agg[j] += val
            mu_b = [v/len(idxs) for v in agg]
            s = sum(mu_b) or 1.0
            mu_b = [v/s for v in mu_b]
        all_idxs = set(range(N_calib_effective))
        sampled_set = set(idxs)
        oob_candidates = list(all_idxs - sampled_set)
        if oob_candidates:
            o_idx = rng.choice(oob_candidates)
        else:
            o_idx = rng.choice(idxs)
        v = vecs[o_idx]
        jsd_val = _jsd_vec(v, mu_b)
        pooled_jsd.append(jsd_val)

    try:
        import numpy as _np
        arr_p = _np.array(pooled_jsd)
        mean_p = float(arr_p.mean()) if arr_p.size else None
        min_p = float(arr_p.min()) if arr_p.size else None
        max_p = float(arr_p.max()) if arr_p.size else None
        p95_p = float(_np.percentile(arr_p, float(args.p_threshold)*100.0 if args.p_threshold <= 1.0 else float(args.p_threshold))) if arr_p.size else None
    except Exception:
        mean_p = statistics.mean(pooled_jsd) if pooled_jsd else None
        min_p = min(pooled_jsd) if pooled_jsd else None
        max_p = max(pooled_jsd) if pooled_jsd else None
        vals_sorted = sorted(pooled_jsd)
        perc = args.p_threshold if args.p_threshold > 1 else (args.p_threshold*100.0)
        idxp = max(0, min(len(vals_sorted)-1, math.ceil((perc/100.0)*len(vals_sorted)) - 1)) if vals_sorted else None
        p95_p = float(vals_sorted[idxp]) if idxp is not None else None

    try:
        import numpy as _np
        tau_jsd = float(_np.percentile(_np.array(pooled_jsd), float(args.p_threshold)*100.0 if args.p_threshold <= 1.0 else float(args.p_threshold)))
    except Exception:
        perc = args.p_threshold if args.p_threshold > 1 else (args.p_threshold*100.0)
        vals = sorted(pooled_jsd)
        idxp = max(0, min(len(vals)-1, math.ceil((perc/100.0)*len(vals)) - 1))
        tau_jsd = float(vals[idxp])

    print(f"[INFO] Umbral JSD (bootstrap/empírico p={args.p_threshold}): {tau_jsd:.6f}")

    # ---------------------------
    # Now compute mu/sigma for phi and bootstrap tau_delta (OOB-like)
    # ---------------------------
    print("[STEP] Calibrando representación φ: calculando mu_phi, sigma_phi y tau_delta (bootstrap)...")
    # filter phi_list to those with n_frag>0 (we preserved order)
    phi_filtered = [phi for i,phi in enumerate(phi_list) if per_query_meta_filtered[i].get("n_frag",0) > 0]
    mu_phi, sigma_phi = compute_mu_sigma_from_phi(phi_filtered)
    # compute Delta_norm for each calib item (to get distribution)
    deltas = []
    for phi in phi_filtered:
        info = per_query_vector_and_deltas(np.asarray(phi, dtype=np.float64) if np is not None else phi, mu_phi, sigma_phi)
        deltas.append(info["Delta_norm"])

    # bootstrap for tau_delta
    rng = random.Random(int(args.seed))
    pooled_delta = []
    Bphi = int(args.phi_boot_B)
    Nphi = len(phi_filtered)
    if Nphi == 0:
        raise SystemExit("No hay vectores φ de calibración para estimar tau_delta.")
    for b in range(Bphi):
        idxs = [rng.randrange(0, Nphi) for _ in range(Nphi)]
        # mu_b_phi:
        if np is not None:
            mu_b_phi = np.mean(np.vstack([phi_filtered[i] for i in idxs]), axis=0)
            sigma_b_phi = np.std(np.vstack([phi_filtered[i] for i in idxs]), axis=0, ddof=1)
            sigma_b_phi = np.where(sigma_b_phi > 0.0, sigma_b_phi, 1e-9)
            # pick OOB
            all_idxs = set(range(Nphi)); sampled_set=set(idxs); oob_candidates=list(all_idxs - sampled_set)
            if oob_candidates:
                o_idx = rng.choice(oob_candidates)
            else:
                o_idx = rng.choice(idxs)
            v = phi_filtered[o_idx]
            # compute Delta_norm for v vs mu_b_phi
            delta = np.asarray(v) - mu_b_phi
            delta_norm = delta / sigma_b_phi
            Delta_norm = float(np.linalg.norm(delta_norm))
            pooled_delta.append(Delta_norm)
        else:
            # fallback simple
            agg = [0.0]*len(phi_filtered[0])
            for i in idxs:
                for j,val in enumerate(phi_filtered[i]):
                    agg[j] += val
            mu_b_phi = [v/len(idxs) for v in agg]
            # pick oob
            all_idxs = set(range(Nphi)); sampled_set=set(idxs); oob_candidates=list(all_idxs - sampled_set)
            if oob_candidates:
                o_idx = rng.choice(oob_candidates)
            else:
                o_idx = rng.choice(idxs)
            v = phi_filtered[o_idx]
            delta = [a-b for a,b in zip(v, mu_b_phi)]
            # approximate sigma as 1 to avoid division issues
            Delta_norm = math.sqrt(sum((d/ (1.0 if 1.0!=0 else 1e-9))**2 for d in delta))
            pooled_delta.append(Delta_norm)

    try:
        import numpy as _np
        tau_delta = float(_np.percentile(_np.array(pooled_delta), float(args.p_threshold)*100.0 if args.p_threshold <= 1.0 else float(args.p_threshold)))
    except Exception:
        vals = sorted(pooled_delta)
        idxp = max(0, min(len(vals)-1, math.ceil((args.p_threshold*100.0/100.0)*len(vals)) - 1))
        tau_delta = float(vals[idxp])

    print(f"[INFO] tau_delta (Phi Δ-norm p={args.p_threshold}): {tau_delta:.6f}")

    # ---------------------------
    # 2) Prepare test samples (same / different)
    # ---------------------------
    if args.sample_mode == "same":
        random.seed(args.seed)
        sample_queries = random.sample(pool_qs, min(args.n_test, len(pool_qs)))
        baseline_sample = sample_queries
        omit_sample = sample_queries
        partial_sample = sample_queries
    else:
        random.seed(args.seed)
        baseline_sample = random.sample(pool_qs, min(args.n_test, len(pool_qs)))
        random.seed(args.seed+1)
        omit_sample = random.sample(pool_qs, min(args.n_test, len(pool_qs)))
        random.seed(args.seed+2)
        partial_sample = random.sample(pool_qs, min(args.n_test, len(pool_qs)))

    # RNG reproducible para degradaciones (semilla derivada)
    deg_rng = random.Random(args.seed + 999)

    # ---------------------------
    # 3) Run baseline tests (run_baseline-like)
    # ---------------------------
    print("[STEP] Ejecutando baseline (sin perturbación)...")
    # reusing run_* logic inline for clarity
    def run_sample(sample_queries, per_dp_topk, degrade_dp=None, degrade_p=0.0, deg_rng=None):
        out = []
        for q in sample_queries:
            qid = q.get("id")
            qtext = q.get("text")
            qobj = {"id": qid, "text": qtext}
            start = time.time()
            frags = retrieve_from_indexes(args.indexes, qobj, per_dp_topk, dp_list,
                                          embeddings_map=embeddings_map, faiss_cache=faiss_cache,
                                          max_search_k=args.max_search_k,
                                          near_threshold_dp=near_threshold_dp,
                                          near_threshold_global=args.near_threshold_global)
            end = time.time()
            latency_ms = (end - start) * 1000.0

            # ---- APLICAR degradación por-FRAGMENT SOLO si degrade_dp pasado (opción A) ----
            if degrade_dp is not None and (degrade_p is not None) and (degrade_p > 0.0) and (deg_rng is not None):
                filtered_frags = []
                for fr in frags:
                    src = fr.get("source")
                    if src == degrade_dp:
                        # con probabilidad degrade_p descartamos el fragment (drop por fragment)
                        if deg_rng.random() < float(degrade_p):
                            continue
                    filtered_frags.append(fr)
                frags = filtered_frags
            # ------------------------------------------------------------------------------

            counts = Counter(fr.get("source","unknown") for fr in frags)
            dist_counts = {d: counts.get(d,0) for d in dp_list}
            total = sum(dist_counts.values())
            norm = {d: (dist_counts[d]/total) if total>0 else 0.0 for d in dp_list}
            redundancy = compute_redundancy(frags)
            H = shannon_entropy_from_counts(counts)
            N_eff = effective_N_from_entropy(H)
            HHI_v = hhi_from_counts(counts)
            item = {
                "query": qtext,
                "id": qid,
                "dist_counts": dict(dist_counts),
                "dist": norm,
                "n_frag": len(frags),
                "redundancy": redundancy,
                "H": H,
                "N_eff": N_eff,
                "HHI": HHI_v,
                "latency_ms": round(latency_ms, 2)
            }
            # build phi and deltas for this test item
            phi = build_phi_from_fragments(item["dist"], item["dist_counts"], item["n_frag"], item["redundancy"], item["latency_ms"], dp_list)
            if np is not None:
                phi_arr = np.asarray(phi, dtype=np.float64)
                delta_info = per_query_vector_and_deltas(phi_arr, mu_phi, sigma_phi)
            else:
                delta_info = per_query_vector_and_deltas(phi, mu_phi, sigma_phi)
            item["phi"] = delta_info["phi"]
            item["Delta_norm"] = delta_info["Delta_norm"]
            out.append(item)
        return out

    baseline_results = run_sample(baseline_sample, {"__default__": args.topk})

    for it in baseline_results:
        if (it.get("n_frag",0) == 0):
            if args.no_data_action == "ignore":
                it["providers_jsd"] = None
                it["decision"] = "no_data"
            elif args.no_data_action == "alert":
                it["providers_jsd"] = float(max(0.0, tau_jsd + 1e-6))
                it["decision"] = "alert"
            else:
                it["providers_jsd"] = float(tau_jsd)
                it["decision"] = "ok"
            # Delta_norm handling
            it["Delta_decision"] = "no_data" if it["Delta_norm"] is None else ("alert" if (it["Delta_norm"] > tau_delta) else "ok")
            continue
        if "dist" in it and isinstance(it["dist"], dict):
            v = [float(it["dist"].get(dp,0.0)) for dp in dp_list]
        elif "dist_counts" in it:
            total = sum(it["dist_counts"].get(dp,0) for dp in dp_list)
            if total == 0:
                v = [0.0]*len(dp_list)
            else:
                v = [float(it["dist_counts"].get(dp,0))/float(total) for dp in dp_list]
        else:
            v = [0.0]*len(dp_list)
        it["providers_jsd"] = _jsd_vec(v, global_ref)
        it["decision"] = "alert" if it["providers_jsd"] > tau_jsd else "ok"
        it["Delta_decision"] = "alert" if (it.get("Delta_norm") is not None and it.get("Delta_norm") > tau_delta) else "ok"

    # 4) Omit DP experiments (SIN degradación; omit simula proveedor eliminado completamente)
    print("[STEP] Ejecutando omit DP (simulación de indisponibilidad TOTAL del proveedor: dp_k=0)...")
    def run_omit_dp_local(sample_queries):
        out_map = {}
        for dp in dp_list:
            cfg = {"__default__": args.topk, dp: 0}
            items = run_sample(sample_queries, cfg)  # no degrade applied here
            # attach providers_jsd decisions
            for it in items:
                if (it.get("n_frag",0) == 0):
                    if args.no_data_action == "ignore":
                        it["providers_jsd"] = None
                        it["decision"] = "no_data"
                    elif args.no_data_action == "alert":
                        it["providers_jsd"] = float(max(0.0, tau_jsd + 1e-6))
                        it["decision"] = "alert"
                    else:
                        it["providers_jsd"] = float(tau_jsd)
                        it["decision"] = "ok"
                    it["Delta_decision"] = "no_data" if it["Delta_norm"] is None else ("alert" if (it["Delta_norm"] > tau_delta) else "ok")
                    continue
                if "dist" in it and isinstance(it["dist"], dict):
                    v = [float(it["dist"].get(dp2,0.0)) for dp2 in dp_list]
                else:
                    total = sum(it["dist_counts"].get(dp2,0) for dp2 in dp_list)
                    if total == 0:
                        v = [0.0]*len(dp_list)
                    else:
                        v = [float(it["dist_counts"].get(dp2,0))/float(total) for dp2 in dp_list]
                it["providers_jsd"] = _jsd_vec(v, global_ref)
                it["decision"] = "alert" if it["providers_jsd"] > tau_jsd else "ok"
                it["Delta_decision"] = "alert" if (it.get("Delta_norm") is not None and it.get("Delta_norm") > tau_delta) else "ok"
            out_map[f"omit_{dp}"] = items
        return out_map

    omit_results_map = run_omit_dp_local(omit_sample)

    # 5) Partial DP experiments (AQUÍ aplicamos degrade_p por-fragment tal como quieres)
    print("[STEP] Ejecutando partial DP (simulación de lectura parcial; APLICANDO degrade_p por-fragment)...")
    if args.partial_k_map:
        try:
            partial_k_map = json.loads(args.partial_k_map)
        except Exception:
            print("[WARN] partial_k_map no parseable, usando default reduction")
            partial_k_map = None
    else:
        partial_k_map = None

    if partial_k_map is None:
        partial_k_map = {"__default__": args.topk}
        for dp in dp_list:
            partial_k_map[dp] = max(1, args.topk // 3)

    def run_partial_dp_local(sample_queries):
        out_map = {}
        for dp in dp_list:
            cfg = {"__default__": partial_k_map.get("__default__", args.topk)}
            cfg[dp] = partial_k_map.get(dp, max(1, args.topk//3)) if (partial_k_map is not None) else max(1, args.topk//3)
            # PASAMOS degrade params aquí para que la degradación actúe en el experimento partial
            items = run_sample(sample_queries, cfg, degrade_dp=dp, degrade_p=args.degrade_p, deg_rng=deg_rng)
            for it in items:
                if (it.get("n_frag",0) == 0):
                    if args.no_data_action == "ignore":
                        it["providers_jsd"] = None
                        it["decision"] = "no_data"
                    elif args.no_data_action == "alert":
                        it["providers_jsd"] = float(max(0.0, tau_jsd + 1e-6))
                        it["decision"] = "alert"
                    else:
                        it["providers_jsd"] = float(tau_jsd)
                        it["decision"] = "ok"
                    it["Delta_decision"] = "no_data" if it["Delta_norm"] is None else ("alert" if (it["Delta_norm"] > tau_delta) else "ok")
                    continue
                if "dist" in it and isinstance(it["dist"], dict):
                    v = [float(it["dist"].get(dp2,0.0)) for dp2 in dp_list]
                else:
                    total = sum(it["dist_counts"].get(dp2,0) for dp2 in dp_list)
                    if total == 0:
                        v = [0.0]*len(dp_list)
                    else:
                        v = [float(it["dist_counts"].get(dp2,0))/float(total) for dp2 in dp_list]
                it["providers_jsd"] = _jsd_vec(v, global_ref)
                it["decision"] = "alert" if it["providers_jsd"] > tau_jsd else "ok"
                it["Delta_decision"] = "alert" if (it.get("Delta_norm") is not None and it.get("Delta_norm") > tau_delta) else "ok"
            out_map[f"partial_{dp}"] = items
        return out_map

    partial_results_map = run_partial_dp_local(partial_sample)

    # 6) Gather test results
    test_results = {"baseline_sample": baseline_results}
    test_results.update(omit_results_map)
    test_results.update(partial_results_map)

    # 7) Optional KS tests
    ks_summary = None
    if args.ks:
        try:
            from scipy.stats import ks_2samp
            print("[STEP] Ejecutando KS tests (baseline vs cada perturbación) sobre providers_jsd...")
            baseline_vals = [it.get("providers_jsd") for it in baseline_results if (it.get("providers_jsd") is not None)]
            ks_summary = {}
            for cfg, items in test_results.items():
                if cfg == "baseline_sample":
                    continue
                other = [it.get("providers_jsd") for it in items if (it.get("providers_jsd") is not None)]
                try:
                    stat, p = ks_2samp(baseline_vals, other)
                except Exception:
                    stat, p = None, None
                ks_summary[cfg] = {"ks_stat": stat, "p_value": p, "n": len(other)}
            print("KS summary:", ks_summary)
        except Exception:
            print("[WARN] scipy no disponible; omitiendo KS tests.")

    # 8) Summaries
    def summarize_jsd_map(results_map, key="providers_jsd"):
        s = {}
        for cfg, items in results_map.items():
            vals = [it.get(key) for it in items if (it.get(key) is not None)]
            no_data = sum(1 for it in items if (it.get("providers_jsd") is None or it.get("decision")=="no_data"))
            s[cfg] = {
                "n": len(items),
                "n_no_data": no_data,
                "mean": statistics.mean(vals) if vals else None,
                "median": statistics.median(vals) if vals else None,
                "p95": (sorted(vals)[max(0, min(len(vals)-1, math.ceil(0.95*len(vals)) - 1))] if vals else None),
                "alerts": sum(1 for it in items if it.get("decision")=="alert"),
                "borderlines": sum(1 for it in items if it.get("decision")=="borderline")
            }
        return s

    summaries = summarize_jsd_map(test_results, key="providers_jsd")
    print("Resumen (providers_jsd):")
    for cfg, st in summaries.items():
        extra = f", no_data={st['n_no_data']}" if st.get("n_no_data") else ""
        print(f" - {cfg}: n={st['n']}, mean={st['mean']}, p95={st['p95']}, alerts={st['alerts']}, borderline={st['borderlines']}{extra}")

    # 9) Debug: compare specific IDs
    if args.debug_ids:
        ids = [s.strip() for s in args.debug_ids.split(",") if s.strip()]
        print("[DEBUG COMP] comparaciones para IDs:", ids)
        for qid in ids:
            b = next((it for it in baseline_results if it["id"]==qid), None)
            print(" QID:", qid)
            if b is None:
                print("  - no encontrado en baseline sample")
            else:
                print(f"  - baseline n_frag={b.get('n_frag')} dist={b.get('dist')} providers_jsd={b.get('providers_jsd')} Delta_norm={b.get('Delta_norm')}")
            for cfg_name, items in test_results.items():
                if cfg_name == "baseline_sample":
                    continue
                o = next((it for it in items if it["id"]==qid), None)
                if o:
                    print(f"   {cfg_name}: n_frag={o.get('n_frag')} dist={o.get('dist')} providers_jsd={o.get('providers_jsd')} Delta_norm={o.get('Delta_norm')}")
            print("")

    # 10) Extra diagnostics: top offenders and L1 changes
    def l1_dist(a,b):
        return sum(abs(a.get(dp,0.0)-b.get(dp,0.0)) for dp in dp_list)

    top_offenders = []
    for cfg, items in test_results.items():
        for it in items:
            jsd = it.get("providers_jsd")
            nfrag = it.get("n_frag", 0)
            if jsd is None:
                continue
            if jsd > tau_jsd:
                top_offenders.append((cfg, it["id"], jsd, nfrag, it.get("dist", {})))

    print(f"[DBG] Top offenders (providers_jsd > tau_to_use={tau_jsd}): n={len(top_offenders)} (showing up to 20)")
    for t in top_offenders[:20]:
        print(" ", "cfg=%s id=%s jsd=%.6f n_frag=%d dist=%s" % (t[0], t[1], t[2], t[3], t[4]))

    l1_changes = []
    baseline_map = {it["id"]: it for it in baseline_results}
    for cfg, items in test_results.items():
        if cfg == "baseline_sample":
            continue
        for it in items:
            bid = baseline_map.get(it["id"])
            if not bid:
                continue
            l1v = l1_dist(bid.get("dist", {}), it.get("dist", {}))
            if l1v > args.report_topk_l1:
                l1_changes.append((cfg, it["id"], round(l1v,4), bid.get("dist", {}), it.get("dist", {})))

    print("[DBG] n queries with L1>{}: {}".format(args.report_topk_l1, len(l1_changes)))
    if l1_changes:
        print(" sample L1 changes (up to 20):")
        for c in l1_changes[:20]:
            print(" ", c)

    # 11) Save outputs (JSON, CSV, figures)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"structural_double_end2end_{ts}.json"

    result_bundle = {
        "dp_list": dp_list,
        "near_threshold_dp": near_threshold_dp,
        "near_threshold_global": args.near_threshold_global,
        "tau_jsd": tau_jsd,
        "tau_delta": tau_delta,
        "pooled_jsd_sample_count": len(pooled_jsd),
        "pooled_jsd_sample": pooled_jsd[:1000],
        "pooled_delta_sample": pooled_delta[:1000],
        "global_ref": global_ref,
        "n_calibration": len(per_query_meta_filtered),
        "calibration_meta": per_query_meta_filtered,
        "phi_mu": (mu_phi.tolist() if hasattr(mu_phi,"tolist") else mu_phi),
        "phi_sigma": (sigma_phi.tolist() if hasattr(sigma_phi,"tolist") else sigma_phi),
        "per_dp_scores_sample_counts": {dp: len(per_dp_scores.get(dp, [])) for dp in dp_list},
        "test_results": test_results,
        "_ks_summary": ks_summary,
        "summaries": summaries,
        "top_offenders_sample": top_offenders[:200],
        "l1_changes_sample": l1_changes[:200],
        "args": vars(args)
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_bundle, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] Results JSON: {out_json}")

    # CSV flattening (simple)
    out_csv = out_dir / f"structural_double_flat_{ts}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fc:
        writer = csv.writer(fc)
        header = ["cfg","id","providers_jsd","providers_decision","Delta_norm","Delta_decision","n_frag"]
        writer.writerow(header)
        for cfg, items in test_results.items():
            for it in items:
                row = [cfg, it.get("id"), it.get("providers_jsd"), it.get("decision"), it.get("Delta_norm"), it.get("Delta_decision"), it.get("n_frag")]
                writer.writerow(row)
    print(f"[SAVED] Results CSV: {out_csv}")

    # Figures (if matplotlib available)
    if plt is not None:
        # 1) hist of calibration pooled_jsd vs baseline providers_jsd
        fig1 = plt.figure(figsize=(6,4))
        try:
            plt.hist(pooled_jsd, bins=50, alpha=0.7, label=f"calibration_jsd (n={len(pooled_jsd)})")
        except Exception:
            pass
        baseline_vals = [it.get("providers_jsd") for it in baseline_results if it.get("providers_jsd") is not None]
        if baseline_vals:
            plt.hist(baseline_vals, bins=20, alpha=0.9, label=f"baseline_jsd (n={len(baseline_vals)})")
        plt.axvline(tau_jsd, color="k", linestyle="--", label=f"tau_jsd={tau_jsd:.6f}")
        plt.xlabel("providers_jsd")
        plt.ylabel("count")
        plt.legend()
        p1 = out_dir / f"hist_jsd_{ts}.png"
        plt.tight_layout(); plt.savefig(p1); plt.close()

        # 2) scatter providers_jsd vs Delta_norm (baseline)
        fig2 = plt.figure(figsize=(6,6))
        xs = [it.get("providers_jsd") for it in baseline_results if it.get("providers_jsd") is not None]
        ys = [it.get("Delta_norm") for it in baseline_results if it.get("providers_jsd") is not None]
        labels = [it.get("id") for it in baseline_results if it.get("providers_jsd") is not None]
        if xs and ys:
            plt.scatter(xs, ys)
            plt.axvline(tau_jsd, color="k", linestyle="--")
            plt.axhline(tau_delta, color="k", linestyle="--")
            for i,lab in enumerate(labels):
                if i % max(1,len(labels)//10) == 0:
                    plt.text(xs[i], ys[i], lab, fontsize=6)
            plt.xlabel("providers_jsd")
            plt.ylabel("delta_norm")
        p2 = out_dir / f"scatter_jsd_delta_{ts}.png"
        plt.tight_layout(); plt.savefig(p2); plt.close()

        # 3) top offenders bar (providers_jsd)
        sorted_off = sorted(top_offenders, key=lambda x: x[2], reverse=True)[:30]
        if sorted_off:
            fig3 = plt.figure(figsize=(8,10))
            names = [f"{t[0]}:{t[1]}" for t in sorted_off]
            vals = [t[2] for t in sorted_off]
            plt.barh(names[::-1], vals[::-1])
            plt.xlabel("providers_jsd")
            plt.title("Top offenders by providers_jsd")
            p3 = out_dir / f"top_offenders_{ts}.png"
            plt.tight_layout(); plt.savefig(p3); plt.close()

        print(f"[SAVED] figures: {p1}, {p2}{' ,' + str(p3) if 'p3' in locals() else ''}")

    print("[DONE] outputs and figures saved in", out_dir)

if __name__ == "__main__":
    main()