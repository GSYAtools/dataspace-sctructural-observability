# Evaluación operativa de la estabilidad estructural en espacios de datos federados

## Repositorio experimental asociado al artículo

**Autores:**  
Carlos Mario Braga Ortuno  
Miguel Ángel Serrano  
Eduardo Fernández-Medina  

**Evento:** JISBD / SISTEDES 2026  
**Año:** 2026  

**Contacto:** carlosmario.braga1@alu.uclm.es  

---

## Descripción general

Este repositorio contiene el pipeline experimental completo utilizado en el artículo:

> *Evaluación operativa de la estabilidad estructural en espacios de datos federados*

El objetivo del trabajo es proponer y validar una metodología operativa para evaluar la estabilidad estructural en arquitecturas RAG federadas mediante métricas estructurales (JSD, H, HHI), representación φ y bootstrap para estimación de umbrales (τ_JSD y τ_Δ).

---

## Relación con el trabajo previo (Repositorio Padre)

Reutilizamos el corpus documental, la arquitectura federada y los índices FAISS del framework experimental publicado en:

**Guided and Federated RAG: Architectural Models for Trustworthy AI in Data Spaces**  
DOI: https://doi.org/10.1007/978-3-032-10489-2_31  
Repositorio original: https://github.com/GSYAtools/DSRAG

> Nota: este repositorio **no depende de `core.py`** del repositorio padre. Los scripts aquí son autocontenidos para las tareas descritas.

---

## Flujo experimental (paso a paso con explicación)

### 1️⃣ Generación de embeddings — `embed_queries_once.py`

**Qué hace:** genera vectores (embeddings) para las queries (preguntas) usando la API de OpenAI.  
**Motivo de aislarlo:** separamos este paso para **optimizar el consumo de la API** (evitar repetir llamadas) y facilitar el reuso de vectores en múltiples ejecuciones del runner estructural (por ejemplo, al probar distintas configuraciones de bootstrap o degradación). También permite usar búsquedas vectoriales locales (FAISS) sin contactar a la API cada vez.  
**Entrada:** `queries.json` (lista de queries).  
**Salida:** `queries_with_embeddings.json` (igual que input, añadiendo campo `embedding` por query).

**Ejemplo de ejecución:**
```bash
python embed_queries_once.py   --queries queries.json   --out queries_with_embeddings.json
```

---

### 2️⃣ Runner estructural — `structural_double_end2end.py`

**Qué hace:** evalúa, por cada query, la estabilidad estructural del sistema de retrieval federado. El script realiza:  
- Recuperación (por proveedor) desde índices (FAISS o matching clásico).  
- Cálculo de distribuciones por proveedor para cada query.  
- Medidas estructurales: JSD frente a referencia global, entropía (H), HHI, redundancia, nº fragments, latencia.  
- Construcción de la representación doble φ (componentes por proveedor + escalares).  
- Bootstrap OOB para calibrar el umbral τ_JS D (comparando vectores de calibración con referencias).  
- Cálculo de mu/sigma de φ y bootstrap para τ_Δ (umbral sobre Δ_norm).  
- Simulaciones experimentales: baseline, omit (omisión total de un proveedor) y partial (degradación por fragmento).  
- Genera JSONs de resultados, CSVs planos y figuras diagnósticas.

**Entradas:**
- `queries.json` (raw) — para determinar las queries y cuáles son de calibración.  
- `queries_with_embeddings.json` (opcional) — para búsqueda vectorial local sin llamar a la API.  
- `indexes/` — carpeta con subcarpetas por proveedor conteniendo índices FAISS y metadatos (proporcionados por el experimento padre).

**Salidas por ejecución (en la carpeta `--out` que especifiques):**
- `structural_double_end2end_<run>_<ts>.json` (bundle con thresholds, global_ref, pooled_jsd, phi_mu/sigma, test_results...)  
- `structural_double_flat_<run>_<ts>.csv` (CSV plano con filas por query/config) — usado por `collect_runs.py`.  
- Figuras de control (opcional).

**Ejemplo de ejecución:**
```bash
python structural_double_end2end.py   --queries queries.json   --embeddings queries_with_embeddings.json   --indexes indexes   --out testQ1_KS   --bootstrap_B 5000   --phi_boot_B 2000   --seed 42
```

---

### 3️⃣ Agregación de ejecuciones — `collect_runs.py`

**Qué hace:** reúne resultados de múltiples ejecuciones (cada ejecución en su carpeta) y produce tablas consolidadas para análisis comparativo. Para cada run:
- localiza el CSV plano (`structural_double_flat_*.csv`) y el JSON de resultados (`structural_double_end2end_*.json`), tomando la versión más reciente si hay varias.  
- extrae y aplana los `args` (parámetros) y métricas top-level (τ_jsd, τ_Δ, n_calibration...).  
- concatena filas de todos los CSVs añadiendo columnas `run_*` con metadatos del run.  
- produce `combined_flat.csv`, `runs_metadata.csv` y `runs_summary.csv` en la carpeta `--out` (por defecto `summary/`).

**Ejemplo de ejecución:**
```bash
python collect_runs.py   --dirs testQ1_KS testQ2_KS testQ3_KS testQ3_3000 test_Q3ensbp10   --out summary
```

---

### 4️⃣ Figuras principales del artículo — `make_figure_from_test.py`

**Qué hace:** carga recursivamente los JSONs de resultados (generados por `structural_double_end2end.py` o ubicados en `summary/`) y genera las figuras usadas en el artículo:  
- `fig3_indicators_vs_p.png` (H y HHI vs p) — **Figura 3** del artículo.  
- `fig2a_mean_jsd_vs_p.png` (media JSD vs p) — **Figura 2a**.  
- `fig2b_pct_alerts_vs_p.png` (% activaciones vs p) — **Figura 2b**.  

Además construye un CSV resumen por valor de `p` si se solicita (`--save-csv`) para reproducibilidad.

**Ejemplo de ejecución:**
```bash
python make_figure_from_test.py   --input-dir summary   --outdir figures   --save-csv
```

---

### 5️⃣ Histograma comparativo (Figura 1) — `plot_hist_jsd.py`

**Qué hace:** genera un histograma comparativo (Calibración vs Omisión total) promediando porcentajes por bin **por fichero** (cada ejecución pesa igual). Esto evita que ejecuciones con distinto número de réplicas (por ejemplo B=5000) dominen la forma de la distribución. Este histograma se corresponde con **Figura 1** del artículo.

**Ejemplo de ejecución:**
```bash
python plot_hist_jsd.py   --input-dir summary   --out figures   --bins 30
```

---

## Estructura del repositorio (resumen)

```
.
├── embed_queries_once.py             # Genera embeddings (OpenAI) para queries.json
├── structural_double_end2end.py      # Runner estructural + calibración + simulaciones
├── collect_runs.py                   # Consolida múltiples runs en summary/
├── make_figure_from_test.py          # Genera Figuras 2a, 2b y 3 (usadas en el artículo)
├── plot_hist_jsd.py                  # Genera histograma comparativo (Figura 1)
├── queries.json                      # Fichero raw de queries
├── queries_with_embeddings.json      # Salida del paso 1
├── indexes/                          # Índices FAISS por proveedor (del repositorio padre)
├── summary/                          # Salida de collect_runs.py
├── figures/                          # Figuras finales (para el artículo)
└── README.md
```

---

## Licencia

Este repositorio se publica bajo licencia: **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**  
(https://creativecommons.org/licenses/by-nc-nd/4.0/)

---
