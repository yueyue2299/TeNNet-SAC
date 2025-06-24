# PINN-SAC

PINN-SAC (Physics-Informed Neural Network for Segment Activity Coefficients) is a machine learning framework designed to predict molecular activity coefficients in multicomponent systems using only molecular SMILES strings, composition, and temperature as input. 

This project provides:

1. **σ-profile prediction model**, including surface area and molecular volume estimation  
2. **Activity coefficient prediction** with two versions:  
   - **Base model**: trained on synthetic data generated from the COSMO-SAC model  
   - **Fine-tuned model**: further optimized using high-quality experimental data  

## Features

- Predicts activity coefficients beyond binary systems
- Requires only SMILES strings, mole fractions, and temperature
- Physics-informed architecture ensures thermodynamic consistency  
- Modular design: supports use of σ-profiles from QC calculations  
- Robust and generalizable via two-stage training: synthetic COSMO-SAC pretraining followed by experimental fine-tuning, preserving physical consistency

## Installation

Clone the repository:

```bash
git clone https://github.com/yueyue2299/PINN-SAC.git
cd PINN-SAC
```

Create and activate the conda environment:

```bash
conda env create -f PINN-SAC.yml
conda activate PINN-SAC
```

## Usage

You can use the [`PINNSAC.ipynb`](./PINNSAC.ipynb) notebook, which demonstrates how to:

- Predict **σ-profiles** and molecular geometry from SMILES strings  
- Predict **activity coefficients** for both binary and multicomponent mixtures  
  - You can choose between the **Base model** (trained on COSMO-SAC data) and the **Fine-tuned model** (refined with experimental data)

## Project Structure

| File/Folder        | Description                                              |
|--------------------|----------------------------------------------------------|
| `PINN-SAC.yml`     | Conda environment configuration                          |
| `requirements.txt` | pip-style dependency list                                |
| `PINNSAC.ipynb`    | Example notebook demonstrating model usage               |
| `ckpt_files`       | Directory for check point files                          |
| `models`           | Directory containing σ-profile, geometry and Γ predictor |
| `smi_ted_light`    | SMI-TED project (external module)                        |
| `utils`            | Utility functions used across the project                |
| `README.md`        | Project readme                                           |



## License

This repository is currently private. Licensing will be added upon publication or release.

## References

This project builds upon the following foundational models. If you use this project in your research, we encourage you to cite them as well:

- **ChemBERTa-2**  
Ahmad, W.; Simon, E.; Chithrananda, S.; Grand, G.; Ramsundar, B. Chemberta-2: Towards chemical foundation models. arXiv preprint arXiv:2209.01712 2022.
[https://arxiv.org/abs/2209.01712](https://arxiv.org/abs/2209.01712)

- **SMI-TED**  
Soares, E.; Shirasuna, V.; Brazil, E. V.; Cerqueira, R.; Zubarev, D.; Schmidt, K. A large encoder-decoder family of foundation models for chemical language. arXiv preprint arXiv:2407.20267 2024.
[https://arxiv.org/abs/2407.20267](https://arxiv.org/abs/2407.20267)

### External Code Acknowledgment

The folder `smi_ted_light/` is adapted from the [SMI-TED](https://github.com/IBM/materials/tree/main/models/smi_ted) repository by Soares et al., with only minimal modifications. The core implementation remains unchanged. Full credit goes to the original authors.

---

Maintained by [@yueyue2299](https://github.com/yueyue2299).
