# DHB-XR Motion Database: Implementation & Storytelling Plan

**Date:** 2026-04-11
**Status:** Planning
**Goal:** Build compelling motion database demos and integrate into project page

---

## Vision

DHB-XR is not just a tokenizer — it's infrastructure for scalable robot learning.
The invariant representation enables a **motion-shape search engine**: encode once, retrieve anywhere.

The pipeline: collect → deduplicate in invariant space → augment to fill gaps → index → retrieve → adapt.

---

## Three Compelling Use Cases

### 1. Cross-Workspace Motion Retrieval
**Scenario:** 1000 demos recorded in Lab A. Lab B has different table/object layout. Query the invariant DB from Lab B to retrieve and adapt the closest motion — no retraining needed.

**Metrics:** Top-1/top-5 retrieval accuracy, EE error after adaptation (mm), wall-clock time vs fine-tuning

**Visualization:** Two-panel figure: Lab A → Lab B with arrows connecting query to retrieved match. Inset shows invariant signatures overlapping despite different absolute positions.

**Implementation:** Already have `MotionDatabase`, `similarity.py`, `libero_full_demo.py`. Need to:
- Build DB from all 10 LIBERO-Spatial tasks (50 demos each = 500 total)
- Query from offset positions and measure retrieval accuracy
- Compare invariant retrieval vs Euclidean nearest-neighbor

### 2. Dataset Compression via Deduplication
**Scenario:** Large robot datasets contain massive redundancy — same reaching motion at slightly different positions. Cluster by invariant distance, keep only centroids.

**Metrics:** Compression ratio, policy success on compressed vs full dataset, invariant distance threshold vs performance

**Visualization:** Scatter plot of trajectories in invariant space colored by cluster. Bar chart: dataset size vs task success with a "knee."

**Implementation:** Need to:
- Compute invariant signatures for all LIBERO-Spatial demos
- Cluster with HDBSCAN or k-means on invariant features
- Measure how many geometrically unique motions exist
- Train SmolVLA on compressed dataset, compare performance

### 3. Few-Shot Skill Library
**Scenario:** 50 motion primitives (reach, grasp, pour, insert, wipe) indexed by invariant signature. New task → decompose → retrieve nearest primitive per segment → chain with SE(3) adaptation.

**Metrics:** Task success with N library primitives vs N raw demos, retrieval latency

**Visualization:** Periodic-table-style grid of skills, each cell = trajectory glyph + invariant signature. Below: task decomposition → retrieved primitives diagram.

**Implementation:** Need to:
- Segment LIBERO demos into reusable sub-trajectories
- Build invariant-indexed library
- Retrieve + adapt + chain for new tasks

---

## Existing Codebase (inventoried)

### Core (`src/dhb_xr/database/`)
- `motion_db.py` — `MotionDatabase` class: add/retrieve by invariant distance. **Functional prototype.**
- `similarity.py` — `invariant_distance` (L2), `quaternion_geodesic_distance`, `soft_dtw_distance`. **Prototype.**
- `retrieval.py` — Thin re-export stub.

### Examples
- `examples/motion_database.py` — Clean 4-trajectory synthetic demo. **Production-ready.**
- `examples/integration/test_libero_retrieval.py` — Real LIBERO cross-task retrieval. **Well-structured.**

### Notebooks
- `demo_dhb_features_and_use_cases.ipynb` — 15-entry toy DB with custom retrieval (NOT using library class).

### Tests
- `tests/test_database.py` — Single happy-path test. **Minimal.**

### Key Bugs/Gaps
1. **FAISS dead code:** `add()` populates index but `retrieve()` never queries it (always brute-force)
2. **No persistence:** No save/load to disk
3. **Soft-DTW is actually hard DTW:** `gamma` parameter ignored in fallback
4. **No variable-length L2:** Asserts equal shapes; only DTW handles mismatch
5. **Thin tests:** 1 test, no edge cases, no DTW/FAISS testing
6. **Notebook decoupled:** Feature showcase builds own retrieval instead of using MotionDatabase

## Action Items

### Phase 1: Fix Core Bugs (TDD)
- [ ] Fix FAISS retrieval (query the index in `retrieve()`)
- [ ] Add save/load persistence (npz or pickle)
- [ ] Fix soft-DTW gamma parameter
- [ ] Add variable-length support to L2 distance (interpolate to match)
- [ ] Expand test suite: DTW, FAISS, persistence, edge cases

### Phase 2: Build LIBERO Motion Database
- [ ] Load all 10 LIBERO-Spatial tasks × 50 demos = 500 trajectories
- [ ] Encode all to DHB invariants and index in MotionDatabase
- [ ] Save database to disk for reuse

### Phase 3: Cross-Workspace Retrieval Benchmark
- [ ] Query from offset positions (0, 3, 5, 10 cm on each axis)
- [ ] Measure top-1/top-5 retrieval accuracy
- [ ] Compare invariant retrieval vs Euclidean nearest-neighbor
- [ ] Plot retrieval accuracy vs offset magnitude

### Phase 4: Dataset Compression
- [ ] Cluster 500 demos by invariant distance (HDBSCAN)
- [ ] Measure number of geometrically unique motions
- [ ] Plot compression ratio vs downstream task success

### Phase 5: Visualization + Project Page
- [ ] Cross-workspace retrieval figure (Lab A → Lab B)
- [ ] Invariant-space scatter plot with clusters
- [ ] Skill library "periodic table" visualization
- [ ] New project page section: "Motion Database"

## Connection to Existing Work

The motion DB closes the loop with augmentation:
- **Augmentation** generates synthetic variations (fill gaps in invariant space)
- **Motion DB** indexes what you have (organize by invariant signature)  
- **Retrieval** adapts stored motions to new configurations (zero-shot transfer)
- **Deduplication** compresses datasets (remove geometric redundancy)

Together: a complete geometric data management pipeline for VLA training at scale.
