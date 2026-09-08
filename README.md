# TeNNet-SAC

TeNNet-SAC (Thermodynamics-Embedded Neural Network for Segment Activity Coefficients) is a machine learning framework designed to predict molecular activity coefficients in multicomponent systems using only molecular SMILES strings, composition, and temperature as input. 

![TeNNet-SAC](architecture.png)

This project provides:

1. **σ-profile prediction model**, including surface area and molecular volume estimation  
2. **Activity coefficient prediction** with two versions:  
   - **Base model**: trained on synthetic data generated from the COSMO-SAC model  
   - **Fine-tuned model**: further optimized using high-quality experimental data  

## Features

- Predicts activity coefficients beyond binary systems
- Requires only SMILES strings, mole fractions, and temperature
- Hard-constraint architecture ensures thermodynamic consistency  
- Modular design: supports use of σ-profiles from QC calculations  
- Robust and generalizable via two-stage training: synthetic COSMO-SAC pretraining followed by experimental fine-tuning, preserving physical consistency

## Installation via PyPI

If you only need to use TeNNet-SAC, install the published package from PyPI.

Create a new environment and install the package:

```bash
conda create -n tsac_env python=3.10 -y
conda activate tsac_env
pip install tennetsac
```
Once installed, import the public API directly:

```python
from tennetsac import profile, binary_lng, multi_lng

# Example 1: Generate σ-profile
s_profile, area, volume = profile("CCO")  # ethanol

# Example 2: Default binary prediction: ensemble mean and population std
mean_1, mean_2, std_1, std_2 = binary_lng(
    ["CCO", "ClCCCl"], 298.15, [0.0, 0.25, 0.5, 0.75, 1.0]
)

# The default binary result returns four lists in this order: mean_1, mean_2,
# std_1, and std_2; mean_1/std_1 describe component 1, mean_2/std_2 describe
# component 2, and both std lists are population standard deviations.

# Example 3: Default multicomponent prediction: ensemble mean and population std
mean, std = multi_lng(["CCO", "ClCCCl", "CCN"], 298.15, [0.3, 0.4])

# Explicit mean-only compatibility mode
mean = multi_lng(["CCO", "ClCCCl", "CCN"], 298.15, [0.3, 0.4], return_std=False)

# Select one numbered fine-tuned member (strings "1" through "10")
member_7 = multi_lng(
    ["CCO", "ClCCCl", "CCN"], 298.15, [0.3, 0.4], version="7", return_std=False
)
```
You can fit temperature-dependent NRTL parameters based on TeNNet-SAC predictions.

### Ensemble statistics and v0.2.0 API change

Starting in **v0.2.0**, `binary_lng` and `multi_lng` default to the ten-member
fine-tuned ensemble and return population standard deviations (`ddof=0`) in
addition to their means. The population standard deviation is calculated over
all ten final outputs. Pass `return_std=False` for the historical mean-only
return shapes.

Statistics are available only for `version="tuned"`. `version="base"` and
numbered members `version="1"` through `version="10"` require
`return_std=False` because each selects one model rather than an ensemble.

The project-owned bundled ensemble asset is
`fine-tuned/gamma-ensemble-v1.safetensors`. Its approximately 5.2 MB file
replaces approximately 37.3 MB of ten duplicated fine-tuned checkpoint files.
Its conversion record and parity command are in
[`docs/model-assets/gamma-ensemble-v1.md`](docs/model-assets/gamma-ensemble-v1.md).

```python
from tennetsac import fit_nrtl, plot_nrtl_fitting

nrtl_results = fit_nrtl(
    "CCO",
    "ClCCCl",
    alpha=0.3,
    temp_range=[300, 350, 400],  # Kelvin
    x_points=21
)

print(nrtl_results)
```
example output:
```
{
  'parameters': {
      'AIJ': 0.0111,
      'AJI': -1.0671,
      'BIJ': 170.582,
      'BJI': 685.029,
      'Alpha': 0.3
  },
  'fitting_metrics': {
      'RMSE': 0.027034,
      'Max_Abs_Error': 0.134795,
      'Success': True
  },
  'input_info': {
      'Component_i': 'CCO',
      'Component_j': 'ClCCCl',
      'Temp_Range_K': [300, 350, 400]
  }
}
```
To evaluate the fitting quality visually:

```python
plot_nrtl_fitting("CCO", "ClCCCl", nrtl_results)
```
**Note**: The non-randomness parameter (α) must be selected based on the thermodynamic characteristics of the system (default: 0.3).

### Quick Tutorial in Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1plBLQIqTAEglNpi3GqOs5TYYZHp5tLny?usp=sharing)


## Installation from Source

If you want to modify or develop TeNNet-SAC locally, clone the release source on GitHub and install its development extras:

```bash
git clone https://github.com/yueyue2299/TeNNet-SAC.git
cd TeNNet-SAC
pip install -e ".[dev]"
```

Alternatively, create the supplied CPU/GPU-ready conda environment, which installs this local package through pip:

```bash
conda env create -f TeNNet-SAC.yml
conda activate TeNNet-SAC
```

## Usage

You can use the [`examples/TeNNetSAC.ipynb`](./examples/TeNNetSAC.ipynb) notebook locally, or try it directly on Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xh1BT-ok73La7AQbjjVSwsRjf6q3JiGx?usp=sharing)

This notebook demonstrates how to:

- Predict **σ-profiles** and molecular geometry from SMILES strings  
- Predict **activity coefficients** for both binary and multicomponent mixtures  
  - You can choose between the **Base model** (trained on COSMO-SAC data) and the **Fine-tuned model** (refined with experimental data)

Importing `tennetsac` does not initialize the models. The packaged models and external embedders initialize on the first prediction call.

### SMI-TED model asset

The default SMI-TED encoder is TeNNet-SAC's inference-only derivative from the
immutable GitHub Release tag `model-smi-ted-light-v1`. It is not bundled in the
wheel or source distribution. On the first prediction call, TeNNet-SAC verifies
the cached asset and downloads it from that immutable release only when the
cache is missing; later calls reuse the verified cache.

To prefetch the asset before a prediction, then verify it without network
access, run:

```bash
python -m tennetsac.model_assets download smi-ted-light
python -m tennetsac.model_assets verify smi-ted-light
```

`TENNETSAC_CACHE_DIR` changes the cache root. Set
`TENNETSAC_SMI_TED_CHECKPOINT` to use an explicit local checkpoint instead of
the cache; the path remains user-owned and is never changed by TeNNet-SAC. For
an offline workflow, run `python -m tennetsac.model_assets download smi-ted-light`
while online, then set `TENNETSAC_OFFLINE=1` before a prediction or `verify`
command. Alternatively, provide the explicit checkpoint override.

The temporary `.pt` override is supported only for the exact pinned legacy IBM
checkpoint. It emits a `FutureWarning`, uses more memory, and will be removed
in the next major package version. A model-release tag is immutable: if v1 is
defective, TeNNet-SAC will publish a new release such as
`model-smi-ted-light-v2`, update the manifest in a package patch, and leave v1
unchanged—published model assets are never moved or replaced.

ChemBERTa2 remains an external dependency. The future offline ChemBERTa2
location, `src/tennetsac/assets/chemberta2/`, is reserved and intentionally
empty; adding a future bundle requires a separate redistribution review.
Package versions are derived from Git tags; GitHub is the release source.

## Project Structure

| File/Folder        | Description                                              |
|--------------------|----------------------------------------------------------|
| `TeNNet-SAC.yml`     | Conda environment configuration                          |
| `requirements.txt` | Development install entry point (`-e .[dev]`)            |
| `examples/TeNNetSAC.ipynb` | Example notebook using the public package API     |
| `src/tennetsac`   | Installable package, three checkpoints, the bundled gamma ensemble, and utilities |
| `src/tennetsac/ckpt_files/fine-tuned/gamma-ensemble-v1.safetensors` | Project-owned bundled ten-member gamma ensemble asset |
| `src/tennetsac/assets/chemberta2/` | Reserved, currently empty offline ChemBERTa2 location |
| `README.md`        | Project readme                                           |

## Citation

Yue Yang, Shiang-Tai Lin. *Physics-Embedded Machine Learning Model for Phase Equilibrium Prediction in Multicomponent Systems*. *Journal of Chemical Information and Modeling*, 2025. [DOI: 10.1021/acs.jcim.5c01804](https://doi.org/10.1021/acs.jcim.5c01804)

- [BibTeX (CITATION.bib)](./CITATION.bib)  
- [RIS (CITATION.ris)](./CITATION.ris)

## References

This project builds upon the following foundational models. If you use this project in your research, we encourage you to cite them as well:

- **ChemBERTa-2**  
Ahmad, W.; Simon, E.; Chithrananda, S.; Grand, G.; Ramsundar, B. Chemberta-2: Towards chemical foundation models. arXiv preprint arXiv:2209.01712 2022.
[https://arxiv.org/abs/2209.01712](https://arxiv.org/abs/2209.01712)

- **SMI-TED**  
Soares, E.; Shirasuna, V.; Brazil, E. V.; Cerqueira, R.; Zubarev, D.; Schmidt, K. A large encoder-decoder family of foundation models for chemical language. arXiv preprint arXiv:2407.20267 2024.
[https://arxiv.org/abs/2407.20267](https://arxiv.org/abs/2407.20267)

### External Code Acknowledgment

The packaged `src/tennetsac/smi_ted_light/` support code contains modified
Apache-2.0 code from IBM's SMI-TED implementation and MIT-licensed Python code
from Idiap's fast-transformers project. The SMI-TED model weights themselves
remain revision-pinned external downloads. Exact repositories, immutable source
commits, copied scope, local modifications, and license files are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Original TeNNet-SAC code is licensed under the MIT License; see
[`LICENSE`](LICENSE). Bundled third-party code remains under its applicable
Apache-2.0 or MIT terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
and the verbatim texts in [`licenses/`](licenses/).

---

Maintained by **Yue Yang** ([@yueyue2299](https://github.com/yueyue2299)).

COMET, Department of Chemical Engineering, National Taiwan University  
