from transformers import RobertaTokenizer, RobertaModel
from utils.smiles import canonicalize_smiles, compute_mol_weight
import torch

# === ChemBERTa2 Embedding  ===
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

# === SMI-TED Embedding  ===
from smi_ted_light.load import load_smi_ted

class SMITEDEmbedder:
    def __init__(self, model_dir="smi_ted_light", ckpt_name="smi-ted-Light_40.pt", device="cpu"):
        self.model = load_smi_ted(folder=model_dir, ckpt_filename=ckpt_name).to(device)
        self.device = device

    def __call__(self, smiles):
        with torch.no_grad():
            emb = self.model.encode(smiles, return_torch=True).to("cpu")  # always return CPU tensor
        return emb

# === function  ===
def get_input_embeddings(smiles, cb_embedder, st_embedder):
    isomeric_cansmi = canonicalize_smiles(smiles, isomeric=True)
    nonisomeric_cansmi = canonicalize_smiles(smiles, isomeric=False)

    cb_emb = cb_embedder(isomeric_cansmi)
    st_emb = st_embedder(nonisomeric_cansmi)
    
    return st_emb, cb_emb