import matplotlib.pyplot as plt
import numpy as np

def plot_sigma_prf(smiles, prf_tensor, savepath=None):
    prf = prf_tensor.squeeze().detach().cpu().numpy()
    sigma = np.linspace(-0.025, 0.025, 51)
    plt.figure(figsize=(6, 4))
    plt.plot(sigma, prf, linestyle='-', linewidth=2)
    
    plt.xlim(-0.025, 0.025)
    plt.xlabel("σ (e/Å²)", fontsize=12)
    plt.ylabel("P(σ)", fontsize=12)
    plt.title(f"σ-Profile: {smiles}", fontsize=13)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()
    

def plot_sigma_prf_pair(smiles_1, prf_tensor_1, smiles_2, prf_tensor_2, savepath=None):
    prf_1 = prf_tensor_1.squeeze().detach().cpu().numpy()
    prf_2 = prf_tensor_2.squeeze().detach().cpu().numpy()
    sigma = np.linspace(-0.025, 0.025, 51)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    axs[0].plot(sigma, prf_1, linestyle='-', linewidth=2)
    axs[0].set_title(f"σ-Profile: {smiles_1}", fontsize=13)
    axs[0].set_xlabel("σ (e/Å²)", fontsize=12)
    axs[0].set_ylabel("P(σ)", fontsize=12)
    axs[0].tick_params(labelsize=10)

    axs[1].plot(sigma, prf_2, linestyle='-', linewidth=2, color='tab:orange')
    axs[1].set_title(f"σ-Profile: {smiles_2}", fontsize=13)
    axs[1].set_xlabel("σ (e/Å²)", fontsize=12)
    axs[1].tick_params(labelsize=10)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

def plot_binary_lng(x1_list, temperature, ln_gamma_1, ln_gamma_2, smiles_1, smiles_2, savepath=None):
    plt.figure(figsize=(6, 4))
    plt.plot(x1_list, ln_gamma_1, label='ln γ₁', linewidth=2)
    plt.plot(x1_list, ln_gamma_2, label='ln γ₂', linewidth=2)
    
    plt.xlabel("x₁", fontsize=12)
    plt.ylabel("ln γ", fontsize=12)
    plt.title(f"Binary Mixture, T = {temperature} K:\n1: {smiles_1}, 2: {smiles_2}", fontsize=13)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()