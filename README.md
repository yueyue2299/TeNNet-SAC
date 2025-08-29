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
- Physics-informed architecture ensures thermodynamic consistency  
- Modular design: supports use of σ-profiles from QC calculations  
- Robust and generalizable via two-stage training: synthetic COSMO-SAC pretraining followed by experimental fine-tuning, preserving physical consistency

## Installation

Clone the repository:

```bash
git clone https://github.com/yueyue2299/TeNNet-SAC.git
cd TeNNet-SAC
```

Create and activate the conda environment:

```bash
conda env create -f TeNNet-SAC.yml
conda activate TeNNet-SAC
```

## Usage

You can use the [`TeNNetSAC.ipynb`](./TeNNetSAC.ipynb) notebook locally, or try it directly on Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xh1BT-ok73La7AQbjjVSwsRjf6q3JiGx?usp=sharing)

This notebook demonstrates how to:

- Predict **σ-profiles** and molecular geometry from SMILES strings  
- Predict **activity coefficients** for both binary and multicomponent mixtures  
  - You can choose between the **Base model** (trained on COSMO-SAC data) and the **Fine-tuned model** (refined with experimental data)

## Project Structure

| File/Folder        | Description                                              |
|--------------------|----------------------------------------------------------|
| `TeNNet-SAC.yml`     | Conda environment configuration                          |
| `requirements.txt` | pip-style dependency list                                |
| `TeNNetSAC.ipynb`    | Example notebook demonstrating model usage               |
| `ckpt_files`       | Directory for check point files                          |
| `models`           | Directory containing σ-profile, geometry and Γ predictor |
| `smi_ted_light`    | SMI-TED project (external module)                        |
| `utils`            | Utility functions used across the project                |
| `README.md`        | Project readme                                           |

## Citation

If you use this project in your research, please cite:

- **TeNNet-SAC** 
We plan to release a related manuscript. Citation information will be added here when available.

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

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Maintained by **Yue Yang** ([@yueyue2299](https://github.com/yueyue2299)).

COMET, Department of Chemical Engineering, National Taiwan University  
