# Identification and Prioritization of Plastic Additives in Chinese Agricultural Soils Using High-Resolution Mass Spectrometry and Network Annotation Propagation

<p align="left">
  <img src="https://img.shields.io/badge/R-HRMS%20processing-276DC3?logo=r&logoColor=white" alt="R">
  <img src="https://img.shields.io/badge/Python-Benchmark%20%26%20mNAP-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
</p>

---

## 🧭 Overview

This repository contains the data-processing and computational workflows supporting the study:

**Identification and Prioritization of Plastic Additives in Chinese Agricultural Soils Using High-Resolution Mass Spectrometry and Network Annotation Propagation**

The repository is organized into three main modules:

| Module | Description | Language |
|---|---|---|
| 🔬 `HRMS_data_processing/` | HRMS feature processing, suspect screening, transformation-product generation, diagnostic MS/MS filtering, and structural annotation | ![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white) |
| 📊 `Benchmark_evaluation/` | Benchmarking of MetFrag, ChemWalker, and modified Network Annotation Propagation (mNAP) | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) |
| 🕸️ `mnap_application/` | Application of mNAP to a GNPS molecular network for candidate re-ranking | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) |

---

## 📁 Repository Structure

```text
.
├── 🔬 HRMS_data_processing/
│   └── HRMS_data_processing.R
│
├── 📊 Benchmark_evaluation/
│   ├── benchmark_nist.py
│   ├── chemwalker_official_worker.py
│   └── run_benchmark.ipynb
│
└── 🕸️ mnap_application/
    └── mNAP_GNPS_application.ipynb
```

---

# 🔬 1. HRMS_data_processing

![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)
![patRoon](https://img.shields.io/badge/patRoon-HRMS%20workflow-4C78A8)
![ProteoWizard](https://img.shields.io/badge/ProteoWizard-mzML%20conversion-6B7280)

`HRMS_data_processing.R` provides the HRMS data-processing workflow implemented using the **patRoon** R package.

### Main workflow

```text
Raw HRMS data
      ↓
Feature detection and alignment
      ↓
Feature filtering
      ↓
Isotope/adduct componentization
      ↓
Suspect screening
      ↓
Transformation-product generation
      ↓
MS/MS processing
      ↓
Diagnostic MS/MS filtering
      ↓
Formula and structural annotation
```

The workflow includes:

- conversion of vendor raw files to centroided mzML using ProteoWizard;
- feature detection, grouping, alignment, and blank filtering;
- isotope and adduct componentization;
- suspect screening and transformation-product generation;
- MS/MS peak-list processing;
- diagnostic MS/MS filtering based on characteristic fragment ions and neutral losses;
- molecular formula generation using **GenForm**; and
- structural candidate generation using **MetFrag** and PubChemLite.

The diagnostic MS/MS filtering section uses organophosphorus compounds as an example. The diagnostic fragment ions and neutral losses can be replaced with class-specific features for other chemical groups.

**Main script**

```text
HRMS_data_processing/HRMS_data_processing.R
```

---

# 📊 2. Benchmark_evaluation

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-00A6D6)
![MetFrag](https://img.shields.io/badge/MetFrag-Candidate%20ranking-7B61A8)

This module provides a reproducible comparison of:

- 🧩 **MetFrag**
- 🚶 **ChemWalker**
- 🧠 **mNAP**

using the NIST tandem mass spectra selected for the published network-annotation benchmark.

The three methods are evaluated using the **same target spectra, seed assignments, and MetFrag candidate pools**, allowing direct comparison of candidate re-ranking performance.

### Main files

- **`benchmark_nist.py`** – main benchmark implementation.
- **`run_benchmark.ipynb`** – stepwise Jupyter interface for benchmark execution.
- **`chemwalker_official_worker.py`** – subprocess wrapper for running the original ChemWalker `cand_pair` and `random_walk` functions safely under Windows/Jupyter.

### Benchmark workflow

```text
NIST MS/MS spectra
        ↓
Molecular network
        ↓
MetFrag candidate generation
        ↓
Seed allocation
        ↓
┌───────────┬────────────┬──────────┐
│  MetFrag  │ ChemWalker │   mNAP   │
└───────────┴────────────┴──────────┘
        ↓
Candidate ranking evaluation
```

The molecular network is constructed using modified cosine similarity with:

- cosine similarity ≥ **0.60**;
- at least **2 matched product ions**; and
- mutual Top-K filtering.

Approximately **10% of nodes in each connected component** are assigned as seeds, with repeated seed allocations used to evaluate robustness.

For the controlled benchmark, MetFrag candidate lists are standardized to enable consistent comparison among methods.

### ▶️ Recommended execution

```python
import benchmark_nist as bm

# Validate inputs and software
bm.run_step(0)

# Build the molecular network
bm.run_step(1, force=True)

# Validate MetFrag output parsing
bm.run_step(2, force=True, max_nodes=5)

# Generate complete candidate lists
bm.run_step(2, force=False)

# Run the benchmark
bm.run_step(3, force=False)

# Evaluate ranking performance
outputs = bm.run_step(4)
```

The benchmark outputs include candidate coverage, Top-k accuracy, mean reciprocal rank, pairwise comparisons, bootstrap analyses, and publication-quality figures.

---

# 🕸️ 3. mnap_application

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![GNPS](https://img.shields.io/badge/GNPS-Molecular%20Networking-2F855A)
![RDKit](https://img.shields.io/badge/RDKit-Tanimoto%20similarity-00A6D6)
![MetFrag](https://img.shields.io/badge/MetFrag-Candidates-7B61A8)

`mNAP_GNPS_application.ipynb` demonstrates the application of mNAP to an experimental **GNPS feature-based molecular network**.

### Required inputs

```text
clusterinfo_summary.tsv
networking_pairs.tsv
spectra.mgf
seed_nodes.csv
PubChemLite.csv
MetFrag-CL.jar
```

Verified structures supplied in `seed_nodes.csv` are used as **seed nodes**, whereas the remaining nodes in the selected molecular-network component are treated as unknown features.

### Application workflow

```text
GNPS molecular network
        ↓
Select network component
        ↓
Identify verified seeds
        ↓
MetFrag candidates for unknown nodes
        ↓
Spectral similarity
        +
Structural similarity
        +
MetFrag score
        ↓
mNAP propagation
        ↓
Candidate re-ranking
```

The notebook first summarizes all available GNPS network components before a component is selected for analysis.

For the example dataset:

```python
component_id = 6164
```

The verified seeds and unknown nodes within this component are then identified automatically.

One unknown node can subsequently be selected for detailed inspection:

```python
target_node = 88874
```

The target node is selected **after** the complete component has been processed and does not alter the mNAP propagation network.

### 🧠 mNAP candidate re-ranking

mNAP integrates:

- MS/MS spectral similarity between connected features;
- structural similarity between candidate molecules;
- normalized MetFrag candidate scores; and
- structural information propagated from verified seed nodes.

Structural similarity is calculated using the **Tanimoto coefficient** based on RDKit molecular fingerprints.

Unlike the controlled benchmark, the experimental application retains **all valid MetFrag candidates** without applying a Top-50 cutoff.

### 📤 Outputs

Two result modes are provided:

**Single target node**

```text
mNAP_node_88874.csv
```

**All unknown nodes in the selected component**

```text
mNAP_all_unknown_nodes.csv
```

Candidate structures before and after mNAP re-ranking can additionally be exported as publication-quality **SVG** and **PDF** figures.

---

# ⚙️ Software Requirements

### ![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white) R workflow

- R
- patRoon
- ProteoWizard
- external software required by the selected patRoon annotation workflow

### ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) Python workflow

- pandas
- numpy
- networkx
- RDKit
- matchms
- pyteomics
- scipy
- matplotlib
- ChemWalker
- Jupyter

### ☕ Additional software

- Java
- MetFrag command-line JAR
- CairoSVG *(optional; PDF structure export)*

---

# 🚀 Workflow Summary

```text
🔬 HRMS data processing
        │
        ▼
HRMS_data_processing.R
        │
        ▼
📊 Independent mNAP benchmark
        │
        ▼
run_benchmark.ipynb
        │
        ▼
🕸️ Application to GNPS molecular networks
        │
        ▼
mNAP_GNPS_application.ipynb
```

---

# 📖 Citation

If you use this repository, please cite the associated manuscript once it becomes available:

> **Su, W. et al. Identification and Prioritization of Plastic Additives in Chinese Agricultural Soils Using High-Resolution Mass Spectrometry and Network Annotation Propagation.**

**DOI:** *To be added after publication.*

### Related methods and software

> 1. **Yu, J. S.; Kwak, Y. B.; Kee, K. H.; Wang, M.; Kim, D. H.; Dorrestein, P. C.; Kang, K. B.; Yoo, H. H., A versatile toolkit for drug metabolism studies with GNPS2: From drug development to clinical monitoring. Nat. Protoc. 2026, 21, (4), 1265-1299.**  

> 2. **Borelli, T. C.; Arini, G. S.; Feitosa, L. G. P.; Dorrestein, P. C.; Lopes, N. P.; da Silva, R. R., Improving annotation propagation on molecular networks through random walks: introducing ChemWalker. Bioinformatics 2023, 39, (3), btad078.**  

> 3. **Silva, R. R. d.; Wang, M.; Nothias, L.-F.; Hooft, J. J. J. v. d.; Caraballo-Rodríguez, A. M.; Fox, E.; Balunas, M. J.; Klassen, J. L.; Lopes, N. P.; Dorrestein, P. C., Propagating annotations of molecular networks using in silico fragmentation. PLOS Comput. Biol. 2018, 14, (4), e1006089.**  

> 4. **Wang, X.; Li, C.; Li, Z.; Qi, Y.; Zhang, X.; Zhao, X.; Zhao, C.; Lin, X.; Lu, X.; Xu, G., A structure-guided molecular network strategy for global untargeted metabolomics data annotation. Anal. Chem. 2023, 95, (31), 11603-11612.**  

> 5. **Zhang, Z.; Pedrycz, W., Intuitionistic multiplicative group analytic hierarchy process and its use in multicriteria group decision-making. IEEE Trans. Cybern. 2018, 48, (7), 1950-1962.**  

---

## 📌 Notes

- Local file paths should be adjusted according to the user's environment.
- Large raw MS files, PubChemLite databases, NIST spectra, and third-party software may need to be obtained separately.
- The controlled benchmark and experimental application use different candidate-retention strategies and should be treated as separate workflows.
- Parameter changes should be documented when adapting the workflow to other datasets.
