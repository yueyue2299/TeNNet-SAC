# === Standard Library ===
import os
import warnings
from typing import List, Tuple

# === Computing & Visualization ===
import torch

# === Warning Suppression ===
from tqdm import TqdmWarning
warnings.simplefilter("ignore", category=TqdmWarning)
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")
from transformers.utils import logging
logging.set_verbosity_error()

# === Model Architectures ===
from .models.Emb2Profile import SigmaProfileGenerator
from .models.Emb2Geometry import GeometryGenerator
from .models.Prf2Gamma import Prf_to_Seg_Model

# === Model Loading ===
from .utils.model_io import load_model, load_all_Gamma_models

# === Embedding Extraction ===
from .utils.embedding import ChemBERTaEmbedder, SMITEDEmbedder

# === Computation ===
from .utils.property import get_sigma_profile, calc_ln_gamma, ensemble_segac, calc_ln_gamma_binary

# === Fitting ===
from scipy.optimize import least_squares
import numpy as np

# === Plotting ===
import matplotlib.pyplot as plt

# === Embedding models ===
cb_emb = ChemBERTaEmbedder()
st_emb = SMITEDEmbedder()

# === Load checkpoints ===
here = os.path.dirname(__file__)
ckpt_path = os.path.join(here, "ckpt_files")

prf_model = load_model(SigmaProfileGenerator(), os.path.join(ckpt_path, "prf.ckpt"))
geometry_model = load_model(GeometryGenerator(), os.path.join(ckpt_path, "geo.ckpt"))
Gamma_base_model = load_model(Prf_to_Seg_Model(), os.path.join(ckpt_path, "base.ckpt"))

Gamma_finetuned_models = load_all_Gamma_models(Prf_to_Seg_Model, os.path.join(ckpt_path, "fine-tuned"))

# === Define functions ===
def sigma_profile_wrapper(smiles):
    return get_sigma_profile(smiles, prf_model, geometry_model, cb_emb, st_emb)

def single_model_predictor(sigma, temperature):
    return Gamma_base_model(sigma, torch.tensor([temperature]))[1]

def ensemble_predictor(sigma, temperature):
    return ensemble_segac(Gamma_finetuned_models, sigma, temperature)

def select_gamma_predictor(model_type: str):
    if model_type == "base":
        return single_model_predictor
    elif model_type == "tuned":
        return ensemble_predictor
    else:
        raise ValueError("Invalid model_type. Choose 'base' or 'tuned'.")

def profile(smiles: str) -> Tuple[List[float], float, float]:
    """
    Generate the sigma profile, surface area, and volume for a given SMILES string.

    Parameters
    ----------
    smiles : str
        Input SMILES string of the molecule.

    Returns
    -------
    tuple
        (sigma_profile: list[float], area: float, volume: float)
    """
    s_prf, area, volume = sigma_profile_wrapper(smiles)
    return s_prf.squeeze().tolist(), area, volume

def binary_lng(smiles: List[str], temperature: float, molefraction: List[float],
               version: str = "tuned") -> Tuple[List[float], List[float]]:
    """
    Calculate the natural logarithm of activity coefficients (ln γ) for a binary mixture.

    Parameters
    ----------
    smiles : list[str]
        List containing exactly two SMILES strings for the binary components.
    temperature : float
        System temperature (K).
    molefraction : list[float]
        Mole fraction of the first component (x1), second is implicitly 1 - x1.
    version : str, optional
        Model type, "base" or "tuned". Default is "tuned".

    Returns
    -------
    tuple
        (ln_gamma_1: list[float], ln_gamma_2: list[float])
    """

    if not isinstance(smiles, list) or len(smiles) != 2:
        raise ValueError(f"'smiles' must be a list of exactly two SMILES strings, got {smiles}")

    gamma_predictor = select_gamma_predictor(version)

    ln_gamma_1, ln_gamma_2 = calc_ln_gamma_binary(smiles[0], smiles[1], molefraction, temperature,
                         gamma_predictor=gamma_predictor,
                         get_sigma_profile_fn=sigma_profile_wrapper)
    return ln_gamma_1.tolist(), ln_gamma_2.tolist()

def multi_lng(smiles: List[str], temperature: float, composition: List[float], version: str = "tuned") -> List[float]:
    """
    Calculate the natural logarithm of activity coefficients (ln γ) for a multicomponent mixture.

    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings for the components (must have length >= 2).
    temperature : float
        System temperature (K).
    composition : list[float]
        Mole fractions of all components (must match the length of smiles, and sum close to 1).
    version : str, optional
        Model type, "base" or "tuned". Default is "tuned".

    Returns
    -------
    list[float]
        ln_gamma values for each component.
    """

    gamma_predictor = select_gamma_predictor(version)

    lng_array = calc_ln_gamma(
    smiles,
    composition,
    temperature,
    gamma_predictor=gamma_predictor,
    get_sigma_profile_fn=sigma_profile_wrapper
    )

    return lng_array.tolist()

def fit_nrtl(smiles1, smiles2,
            alpha=0.3,
            temp_range=None,
            x_points=21):
    """
    Use tennetsac to generate data and fit NRTL parameters.

    NRTL: tau_ij = A_ij + B_ij / T

    Parameters:
    -----------
    smiles1, smiles2 : str

    alpha : float
        NRTL non-random parameter
    temp_range : list of float, optional
        List of fitted temperature (Kelvin)
    x_points : int
        Number of concentration points

    Returns:
    --------
    dict
        NRTL parameters in Kelvin (A12, A21, B12, B21) and error matrices (RMSE, Max_Error)。
    """

    if temp_range is None:
        temp_range = [298.15, 313.15, 328.15, 343.15] # 25, 40, 55, 70 °C

    x1_eval = np.linspace(0.0, 1.0, x_points)

    t_data = []
    x1_data = []
    lng1_obs = []
    lng2_obs = []

    smiles_pair = [smiles1, smiles2]

    for T in temp_range:
        l1, l2 = binary_lng(smiles_pair, T, x1_eval.tolist())
        lng1_obs.extend(l1)
        lng2_obs.extend(l2)
        x1_data.extend(x1_eval)
        t_data.extend([T] * len(x1_eval))

    x1_data = np.array(x1_data)
    x2_data = 1.0 - x1_data
    t_data = np.array(t_data)
    y_obs = np.concatenate([lng1_obs, lng2_obs])

    # 3. NRTL (Aspen Form)
    def calculate_nrtl(params, x1, x2, T):
        A12, A21, B12, B21 = params

        # Aspen form: tau = A + B/T
        tau12 = A12 + B12 / T
        tau21 = A21 + B21 / T

        G12 = np.exp(-alpha * tau12)
        G21 = np.exp(-alpha * tau21)

        # NRTL Equations
        # ln(gamma1)
        term1_1 = tau21 * (G21 / (x1 + x2 * G21))**2
        term1_2 = (tau12 * G12) / (x2 + x1 * G12)**2
        lng1_calc = x2**2 * (term1_1 + term1_2)

        # ln(gamma2)
        term2_1 = tau12 * (G12 / (x2 + x1 * G12))**2
        term2_2 = (tau21 * G21) / (x1 + x2 * G21)**2
        lng2_calc = x1**2 * (term2_1 + term2_2)

        return lng1_calc, lng2_calc

    #  Residuals
    def residuals(params):
        lng1_calc, lng2_calc = calculate_nrtl(params, x1_data, x2_data, t_data)
        y_calc = np.concatenate([lng1_calc, lng2_calc])
        # prevent NaN of Inf
        diff = y_obs - y_calc
        return np.nan_to_num(diff, nan=10.0, posinf=10.0, neginf=-10.0)

    #  Fitting
    #  [A12, A21, B12, B21]
    initial_guess = [0.0, 0.0, 0.0, 0.0]

    result = least_squares(residuals, initial_guess, method='lm')

    # error
    final_params = result.x
    final_resid = residuals(final_params)
    rmse = np.sqrt(np.mean(final_resid**2))
    max_error = np.max(np.abs(final_resid))

    return {
        "parameters": {
            "AIJ": round(final_params[0], 4), # A12
            "AJI": round(final_params[1], 4), # A21
            "BIJ": round(final_params[2], 4), # B12
            "BJI": round(final_params[3], 4), # B21
            "Alpha": alpha
        },
        "fitting_metrics": {
            "RMSE": round(rmse, 6),
            "Max_Abs_Error": round(max_error, 6),
            "Success": result.success
        },
        "input_info": {
            "Component_i": smiles1,
            "Component_j": smiles2,
            "Temp_Range_K": temp_range
        }
    }


def plot_nrtl_fitting(smiles1, smiles2, fit_result):
    """
    Two graphs showing the NRTL fitting results:
    1. Gamma vs Composition (fitting curves at different temperatures)
    2. Parity Plot (prediction accuracy analysis)
    """

    # 1. extract fitting parameters
    p = fit_result['parameters']
    params_vec = [p['AIJ'], p['AJI'], p['BIJ'], p['BJI']]
    alpha = p['Alpha']
    temp_range = fit_result['input_info']['Temp_Range_K']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Parity Plot
    all_obs = []
    all_calc = []

    colors = plt.cm.viridis(np.linspace(0, 1, len(temp_range)))

    def get_nrtl_gamma(x1, T, A12, A21, B12, B21, alpha):
        x2 = 1.0 - x1
        tau12 = A12 + B12 / T
        tau21 = A21 + B21 / T
        G12 = np.exp(-alpha * tau12)
        G21 = np.exp(-alpha * tau21)

        # ln(gamma1)
        term1_1 = tau21 * (G21 / (x1 + x2 * G21))**2
        term1_2 = (tau12 * G12) / (x2 + x1 * G12)**2
        lng1 = x2**2 * (term1_1 + term1_2)

        # ln(gamma2)
        term2_1 = tau12 * (G12 / (x2 + x1 * G12))**2
        term2_2 = (tau21 * G21) / (x1 + x2 * G21)**2
        lng2 = x1**2 * (term2_1 + term2_2)

        return lng1, lng2

    for i, T in enumerate(temp_range):
        x_discrete = np.linspace(0.0, 1.0, 11)
        lng1_obs_list, lng2_obs_list = binary_lng([smiles1, smiles2], T, x_discrete.tolist())

        lng1_obs = np.array(lng1_obs_list)
        lng2_obs = np.array(lng2_obs_list)

        x_smooth = np.linspace(0, 1, 100)
        lng1_calc_smooth, lng2_calc_smooth = get_nrtl_gamma(
            x_smooth, T, *params_vec, alpha
        )

        lng1_calc_pts, lng2_calc_pts = get_nrtl_gamma(
            x_discrete, T, *params_vec, alpha
        )

        ax1.plot(x_smooth, lng1_calc_smooth, '-', color=colors[i], label=f'T={T}K')
        ax1.scatter(x_discrete, lng1_obs, color=colors[i], marker='o', s=30, alpha=0.7)

        ax1.plot(x_smooth, lng2_calc_smooth, '--', color=colors[i], alpha=0.5)
        ax1.scatter(x_discrete, lng2_obs, color=colors[i], marker='^', s=30, alpha=0.5)
        all_obs.extend(lng1_obs)
        all_obs.extend(lng2_obs)
        all_calc.extend(lng1_calc_pts)
        all_calc.extend(lng2_calc_pts)

    ax1.set_title(f'NRTL Fit: {smiles1} / {smiles2}', fontsize=14)
    ax1.set_xlabel(f'Mole Fraction of {smiles1} (x1)', fontsize=12)
    ax1.set_ylabel('ln(Activity Coefficient)', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.text(0.02, 0.02, "Solid/Circle: Comp 1\nDashed/Triangle: Comp 2",
             transform=ax1.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    min_val = min(min(all_obs), min(all_calc))
    max_val = max(max(all_obs), max(all_calc))

    ax2.plot([min_val, max_val], [min_val, max_val], 'k-', lw=2, alpha=0.3)

    ax2.scatter(all_obs, all_calc, c='blue', alpha=0.5, edgecolors='k')

    rmse = np.sqrt(np.mean((np.array(all_obs) - np.array(all_calc))**2))
    ax2.text(0.05, 0.9, f"RMSE = {rmse:.4f}", transform=ax2.transAxes, fontsize=12,
             bbox=dict(facecolor='wheat', alpha=0.5))

    ax2.set_title('Parity Plot (Goodness of Fit)', fontsize=14)
    ax2.set_xlabel('Tennetsac Values', fontsize=12)
    ax2.set_ylabel('NRTL Calculated Values', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()
