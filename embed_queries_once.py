#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed_queries_openai_v1.py

Genera embeddings con la API de OpenAI (openai>=1.0.0) para un fichero de queries
y escribe un JSON con la misma estructura de entrada pero añadiendo
el campo "embedding" (lista de floats) por cada query.

USO:
 python embed_queries_openai_v1.py --queries queries.json --out queries_with_embeddings.json --model text-embedding-3-small --batch 16

Requisitos:
 - pip install python-dotenv openai>=1.0.0
 - fichero .env en la raíz con OPENAI_API_KEY=sk-...
"""
import os
import json
import time
import argparse
from pathlib import Path

# cargar .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# new OpenAI client
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("Instala la librería openai>=1.0.0 (pip install openai).") from e

def normalize_queries(raw):
    out = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            qid = f"Q{i+1:04d}"
            out.append({"id": qid, "text": item, "_orig_is_str": True})
        elif isinstance(item, dict):
            qid = item.get("id") or item.get("qid") or f"Q{i+1:04d}"
            text = item.get("text") or item.get("query") or ""
            preserved = {k: v for k, v in item.items() if k not in ("id", "qid", "text", "query")}
            out.append({"id": str(qid), "text": text, "_orig_is_str": False, "_preserved": preserved})
        else:
            qid = f"Q{i+1:04d}"
            out.append({"id": qid, "text": str(item), "_orig_is_str": True})
    return out

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True, help="Fichero JSON de entrada (array)")
    p.add_argument("--out", default="queries_with_embeddings.json", help="Fichero JSON de salida")
    p.add_argument("--model", default="text-embedding-3-small", help="Modelo de embeddings OpenAI")
    p.add_argument("--batch", type=int, default=16, help="Tamaño de batch para las llamadas a la API")
    p.add_argument("--delay", type=float, default=0.5, help="Segundos a esperar entre batches")
    p.add_argument("--max_retries", type=int, default=2, help="Reintentos en caso de fallo por batch")
    args = p.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY no encontrado en el entorno. Crea un .env con OPENAI_API_KEY=sk-... o exporta la variable.")

    client = OpenAI(api_key=api_key)

    qpath = Path(args.queries)
    if not qpath.exists():
        raise SystemExit(f"Fichero no encontrado: {qpath}")

    with open(qpath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized = normalize_queries(raw)
    print(f"Total queries a procesar: {len(normalized)}")

    results = []
    for bidx, batch in enumerate(chunked(normalized, args.batch), start=1):
        texts = [q["text"] for q in batch]
        # reintentos simples
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=args.model, input=texts)
                # resp.data is a list with embeddings per input
                embeddings = [item.embedding for item in resp.data]
                break
            except Exception as e:
                attempt += 1
                print(f"[WARN] Falló llamada a OpenAI en batch {bidx} (intento {attempt}): {e}")
                if attempt > args.max_retries:
                    raise SystemExit(f"[FATAL] Batch {bidx} falló después de {args.max_retries} reintentos.")
                time.sleep(2.0 + attempt*1.0)

        for qitem, emb in zip(batch, embeddings):
            # emb is a list of floats (or similar)
            emb_list = list(map(float, emb))
            if qitem["_orig_is_str"]:
                out_obj = {"id": qitem["id"], "text": qitem["text"], "embedding": emb_list}
            else:
                base = {"id": qitem["id"], "text": qitem["text"]}
                if isinstance(qitem.get("_preserved"), dict):
                    base.update(qitem["_preserved"])
                base["embedding"] = emb_list
                out_obj = base
            results.append(out_obj)

        print(f"[BATCH {bidx}] Procesadas {len(batch)} queries.")
        time.sleep(args.delay)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Embeddings generados y guardados en {out_path}")

if __name__ == "__main__":
    main()