import torch
import numpy as np
from utils.embedding import get_input_embeddings
from utils.smiles import compute_mol_weight

def get_sigma_profile(smiles, prf_model, geometry_model, cb_embedder, st_embedder):
    st_emb, cb_emb = get_input_embeddings(smiles, cb_embedder, st_embedder)
    mw_tensor = torch.tensor(compute_mol_weight(smiles)).unsqueeze(0)
    with torch.no_grad():
        normalized_sigma_profile = prf_model(st_emb, cb_emb)
        geometry = geometry_model(st_emb, cb_emb, mw_tensor)
        
    area, volume = geometry[0][0].item(), geometry[0][1].item()
    
    return normalized_sigma_profile * area, area, volume

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

def ensemble_segac(models, sigma, temperature):
    sigma = sigma.clone().detach().requires_grad_(True)
    segac_list = []

    for model in models:
        _, segac_i = model(sigma, torch.tensor([temperature]))
        segac_list.append(segac_i.numpy())

    segac_mean = np.mean(segac_list, axis=0)
    return segac_mean

def calc_ln_gamma(smiles_list, mole_fraction_list, temperature, gamma_predictor, get_sigma_profile_fn):
    aeff = 5.8447  # A2
    num_components = len(smiles_list)

    if len(mole_fraction_list) == num_components - 1:
        last_mf = 1.0 - sum(mole_fraction_list)
        if last_mf < 0 or last_mf > 1:
            raise ValueError(f"Invalid mole fractions: sum exceeds 1.0 → sum = {sum(mole_fraction_list):.4f}")
        mole_fraction_list = mole_fraction_list + [last_mf]
        # print(f"[Info] Mole fraction auto-completed: last component = {last_mf:.4f}")
    elif len(mole_fraction_list) != num_components:
        raise ValueError(f"Length mismatch: {num_components} SMILES vs {len(mole_fraction_list)} mole fractions.\n"
                        f"Expected mole_fraction_list to have {num_components - 1} or {num_components} items.")
    elif len(mole_fraction_list) == num_components:
        mf_sum = sum(mole_fraction_list)
        if abs(mf_sum - 1.0) > 1e-6:
            raise ValueError(f"Mole fractions must sum to 1.0, but got {mf_sum:.6f}")
    
    sigma_profiles = []
    areas = []
    volumes = []
    segacs_pure = []

    for smiles in smiles_list:
        sigma, area, volume = get_sigma_profile_fn(smiles)
        sigma = sigma.clone().detach()
        segac = gamma_predictor(sigma, temperature)
        sigma_profiles.append(sigma)
        areas.append(area)
        volumes.append(volume)
        segacs_pure.append(segac)

    ln_gamma_comb_list = compute_SG_combinatorial_term(mole_fraction_list, areas, volumes)

    sigma_mix = sum(xi * sigma for xi, sigma in zip(mole_fraction_list, sigma_profiles))
    segac_mix = gamma_predictor(sigma_mix, temperature)

    ln_gamma_res_list = []
    for i in range(num_components):
        p_i = sigma_profiles[i] / areas[i]
        delta_ln = segac_mix - segacs_pure[i]
        ln_gamma_res = areas[i] / aeff * torch.sum(p_i * delta_ln).item()
        ln_gamma_res_list.append(ln_gamma_res)

    ln_gamma_comb_array = np.array(ln_gamma_comb_list)
    ln_gamma_res_array = np.array(ln_gamma_res_list)
    ln_gamma_array = ln_gamma_comb_array + ln_gamma_res_array
    
    # return ln_gamma_comb_array, ln_gamma_res_array, ln_gamma_array
    return ln_gamma_array

def calc_ln_gamma_binary(smiles_1, smiles_2, x1_list, temperature,
                         gamma_predictor, get_sigma_profile_fn):
    aeff = 5.8447  # A²

    # Get sigma profiles, areas, volumes
    sigma_1, area_1, vol_1 = get_sigma_profile_fn(smiles_1)
    sigma_2, area_2, vol_2 = get_sigma_profile_fn(smiles_2)
    sigma_1 = sigma_1.clone().detach()
    sigma_2 = sigma_2.clone().detach()

    # Predict segment activity coefficients
    segac_1 = gamma_predictor(sigma_1, temperature)
    segac_2 = gamma_predictor(sigma_2, temperature)

    ln_gamma_1_list = []
    ln_gamma_2_list = []

    for x1 in x1_list:
        x2 = 1.0 - x1

        # combinatorial part
        ln_gamma_comb_1, ln_gamma_comb_2 = compute_SG_combinatorial_term(
            [x1, x2], [area_1, area_2], [vol_1, vol_2]
        )

        # mixture sigma profile
        sigma_mix = x1 * sigma_1 + x2 * sigma_2
        segac_mix = gamma_predictor(sigma_mix, temperature)

        # residual part
        p_1 = sigma_1 / area_1
        p_2 = sigma_2 / area_2
        delta_ln_1 = segac_mix - segac_1
        delta_ln_2 = segac_mix - segac_2
        ln_gamma_res_1 = area_1 / aeff * torch.sum(p_1 * delta_ln_1).item()
        ln_gamma_res_2 = area_2 / aeff * torch.sum(p_2 * delta_ln_2).item()

        # total ln gamma
        ln_gamma_1 = ln_gamma_comb_1 + ln_gamma_res_1
        ln_gamma_2 = ln_gamma_comb_2 + ln_gamma_res_2

        ln_gamma_1_list.append(ln_gamma_1)
        ln_gamma_2_list.append(ln_gamma_2)

    return np.array(ln_gamma_1_list), np.array(ln_gamma_2_list)
