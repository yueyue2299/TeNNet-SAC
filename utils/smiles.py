from rdkit import Chem
from rdkit.Chem import Descriptors

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