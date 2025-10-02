# private_sampling

This repository contains the official code implementation for the experiments in:

**Locally Optimal Private Sampling: Beyond the Global Minimax**  
Hrad Ghoukasian · Bonwoo Lee · Shahab Asoodeh  
*NeurIPS 2025*

---

## 📂 Repository Structure

- `experiments/` — scripts to run the experiments and generate data.  
- `plotting/` — scripts to visualize results from the saved experiment data.  
- `utils/` — core implementations of mechanisms and measures used in the experiments.
- 'figure/' — generated figures used in the paper  
- `requirements.txt` — Python package dependencies.  

All scripts should be run from the **repository root** using the `-m` flag (e.g., `python -m experiments.exp_...`).

---

## 🔹 Figure 1 — Laplace Mixture Visualization

To generate the output distributions and produce Figure 1:

```bash
python -m experiments.exp_LapMixture_visual
python -m plotting.plot_LapMixture_visual
```
---

## 🔹 Finite Space Results

Six figures are generated, grouped into two categories: pure LDP and ν-GLDP.

### Pure LDP (Figures 3, 5, 6)

```bash
python -m plotting.plot_finite_pure --k 10
python -m plotting.plot_finite_pure --k 20
python -m plotting.plot_finite_pure --k 100
```

### ν-GLDP (Figures 7, 8, 9)

```bash
python -m plotting.plot_finite_GLDP --k 10
python -m plotting.plot_finite_GLDP --k 20
python -m plotting.plot_finite_GLDP --k 100

---

## 🔹 Continuous Space Results

### 1D Laplace Mixture — Pure LDP (Figure 4)

```bash
python -m experiments.exp_1DLaplaceMix_pure --eps 0.1 --scale 1 --seed 1
python -m experiments.exp_1DLaplaceMix_pure --eps 0.5 --scale 1 --seed 2
python -m experiments.exp_1DLaplaceMix_pure --eps 1.0 --scale 1 --seed 3
python -m experiments.exp_1DLaplaceMix_pure --eps 2.0 --scale 1 --seed 4

python -m plotting.plot_1DLaplaceMix_pure
```
---
### 1D Laplace Mixture — ν-GLDP (Figure 10)

```bash
python -m experiments.exp_1DLaplaceMix_GLDP --nu 0.1 --scale 1 --seed 1
python -m experiments.exp_1DLaplaceMix_GLDP --nu 0.5 --scale 1 --seed 2
python -m experiments.exp_1DLaplaceMix_GLDP --nu 1.0 --scale 1 --seed 3
python -m experiments.exp_1DLaplaceMix_GLDP --nu 2.0 --scale 1 --seed 4

python -m plotting.plot_1DLaplaceMix_GLDP
```
---

### 2D Laplace Mixture — Pure LDP (Figure 11)

```bash
python -m experiments.exp_nDLaplaceMix_pure --eps 0.1 --seed 1 --dim 2
python -m experiments.exp_nDLaplaceMix_pure --eps 0.5 --seed 2 --dim 2
python -m experiments.exp_nDLaplaceMix_pure --eps 1.0 --seed 3 --dim 2
python -m experiments.exp_nDLaplaceMix_pure --eps 2.0 --seed 4 --dim 2

python -m plotting.plot_nDLaplaceMix_pure --dim 2
```
---

## 📊 Output

Each plotting script produces a .pdf (and sometimes .eps) file containing the reproduced figure.
Make sure the corresponding experiment scripts are run first to generate the required .npz files.

---

## ⚙️ System Requirements

Dependencies listed in requirements.txt

Install dependencies with:

```bash
pip install -r requirements.txt
```
---



