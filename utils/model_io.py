import torch

def load_model(model, ckpt_file, device='cpu'):
    model.to(torch.device(device))
    model.load_state_dict(torch.load(ckpt_file, map_location=torch.device(device)))
    
    return model.eval()

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