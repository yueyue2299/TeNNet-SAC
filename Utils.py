import torch
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from transformers import RobertaTokenizer, RobertaModel

import numpy as np

# === Embedding Models ===
class ChemBERTaEmbedder:
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM", max_length=128, device="cpu"):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(device)
        self.max_length = max_length
        self.device = device

    def __call__(self, smiles):
        tokens = self.tokenizer(
            smiles, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            output = self.model(tokens["input_ids"], tokens["attention_mask"])
            hidden = output.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            avg_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return avg_emb

# SMILES function
def canonicalize_smiles(smi, isomeric=False, canonical=True):
    try:
        can_smi = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        print("Error: Unparseable SMILES string")
        can_smi = None
    return can_smi

def compute_mol_weight(smiles):
    """
    Calculate molecular weight based on SMILES string
    Print error if it happened, and return None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Error: Unparseable SMILES string")
            return None
        weight = Descriptors.ExactMolWt(mol)
        return weight
    except Exception as e:
        print("Error:", e)
        return None

# Model
def load_model(model, ckpt_file, device='cpu'):
    model.to(torch.device(device))
    model.load_state_dict(torch.load(ckpt_file, map_location=torch.device(device)))
    
    return model.eval()

# Ensemble model
def load_all_Gamma_models(model_class, ckpt_dir, num_models=10):
    models = []
    for i in range(1, num_models + 1):
        ckpt_file = f'{ckpt_dir}\\{i}.ckpt'
        model = model_class()
        model.to(torch.device('cpu'))
        model.load_state_dict(torch.load(ckpt_file, map_location=torch.device('cpu')))
        model.eval()
        models.append(model)
    return models

def ensemble_segac(models, sigma, temperature):
    sigma = sigma.clone().detach().requires_grad_(True)
    segac_list = []

    for model in models:
        _, segac_i = model(sigma, torch.tensor([temperature]))
        segac_list.append(segac_i.numpy())

    segac_mean = np.mean(segac_list, axis=0)
    return segac_mean

# Get embeddings
def get_ChemBERTa2_embedding(smiles, tokenizer, model):
    tokens = tokenizer(smiles, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    with torch.no_grad():
        output = model(tokens["input_ids"], tokens["attention_mask"])
        hidden = output["last_hidden_state"]
        mask = tokens["attention_mask"].unsqueeze(-1)
        avg_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return avg_emb

def get_SMI_TED_embedding(smiles, model):
    with torch.no_grad():
        encode_embeddings = model.encode(smiles, return_torch=True)
    return encode_embeddings.cpu()

def get_input_embeddings(smiles, cb_model, st_model):
    isomeric_cansmi = canonicalize_smiles(smiles, isomeric=True)
    nonisomeric_cansmi = canonicalize_smiles(smiles, isomeric=False)

    cb_emb = get_ChemBERTa2_embedding(isomeric_cansmi, cb_model)
    st_emb = get_SMI_TED_embedding(nonisomeric_cansmi, st_model)
    
    return st_emb, cb_emb

# sigma profile
def get_sigma_profile(smiles, prf_model, geometry_model):
    st_emb, cb_emb = get_input_embeddings(smiles)
    mw_tensor = torch.tensor(compute_mol_weight(smiles)).unsqueeze(0)
    with torch.no_grad():
        normalized_sigma_profile = prf_model(st_emb, cb_emb)
        geometry = geometry_model(st_emb, cb_emb, mw_tensor)
        
    area, volume = geometry[0][0].item(), geometry[0][1].item()
    
    return normalized_sigma_profile * area, area, volume

# ln gamma calculation 
def compute_SG_combinatorial_term(x, areas, volumes):
    """
    Compute the Staverman-Guggenheim combinatorial term based on areas and volumes.

    Parameters:
        x (list[float]): List of mole fractions.
        areas (list): List of area.
        volumes (list): List of volume.

    Returns:
        - lngc (list[float]): The logarithmic correction for each component.
    """
    ncomp = len(x) # Number of components
    refarea = 79.531954 # reference area, unit: A2
    z = 5.0 # coordination number, no unit
    totq = 0.0
    totr = 0.0

    # compute total weighted area and total weighted volume.
    for i in range(ncomp):
        totq += x[i] * areas[i]
        totr += x[i] * volumes[i]

    # Initialize arrays.
    lngc = [0.0] * ncomp

    # compute each component's lngc.
    for i in range(ncomp):
        r = volumes[i] / totr # normalized volume
        q = areas[i] / totq # normalized area
        lngc[i] = 1 - r + np.log(r) - z * (areas[i] / refarea) * (1 - r/q + np.log(r/q))

    return lngc

# 
def calc_ln_gamma(smiles_list, mole_fraction_list, temperature, Gamma_models):
    aeff = 5.8447 # A2
    # the last mole_fraction
    mole_fraction_list.append(1.0 - sum(mole_fraction_list))

    num_components = len(smiles_list)
    assert num_components == len(mole_fraction_list)

    sigma_profiles = []
    areas = []
    volumes = []
    segacs_pure = []

    # Calculate Pure profile and segment Gamma
    for smiles in smiles_list:
        sigma, area, volume = get_sigma_profile(smiles)
        sigma = sigma.clone().detach()
        # segac = Gamma_model(sigma, torch.tensor([temperature]))[1]
        segac = ensemble_segac(Gamma_models, sigma, temperature)
        sigma_profiles.append(sigma)
        areas.append(area)
        volumes.append(volume)
        segacs_pure.append(segac)
    
    ln_gamma_comb_list = compute_SG_combinatorial_term(mole_fraction_list, areas, volumes)
    
    # mixture sigma profile = Σ xi * σi
    sigma_mix = sum(xi * sigma for xi, sigma in zip(mole_fraction_list, sigma_profiles))
    # print(sigma_profiles[1]-sigma_profiles[0])
    segac_mix = ensemble_segac(Gamma_models, sigma_mix, temperature)
    # segac_mix = Gamma_model(sigma_mix, torch.tensor([temperature]))[1]  # Γ_S(σ)

    # plot_sigma_profile('Mixture', torch.tensor(sigma_mix))
    # plot_segment_activity('Mixture', temperature, segac_mix)

    # calculate ln γ^res
    ln_gamma_res_list = []
    for i in range(num_components):
        p_i = sigma_profiles[i] / areas[i]  # normalized σ profile
        delta_ln = segac_mix - segacs_pure[i]
        ln_gamma_res = areas[i] / aeff * torch.sum(p_i * delta_ln).item()  # n_i * Σ p_i(σ) * (ln Γ_S - ln Γ_i)
        ln_gamma_res_list.append(ln_gamma_res)


    ln_gamma_comb_array = np.array(ln_gamma_comb_list)
    ln_gamma_res_arrary = np.array(ln_gamma_res_list)
    ln_gamma_array = ln_gamma_comb_array + ln_gamma_res_arrary
    
    return ln_gamma_comb_array, ln_gamma_res_arrary, ln_gamma_array