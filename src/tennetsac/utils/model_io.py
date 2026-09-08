import torch

def load_model(model, ckpt_file, device='cpu'):
    model.to(torch.device(device))
    model.load_state_dict(torch.load(ckpt_file, map_location=torch.device(device)))
    
    return model.eval()
