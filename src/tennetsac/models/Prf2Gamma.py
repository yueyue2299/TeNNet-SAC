import torch
import torch.nn as nn
import torch.autograd as autograd

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, activation=nn.GELU):
        """
        in_features: input feature
        hidden_features: hidden feature, if None, use in_features
        """
        super(ResidualBlock, self).__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out += identity  # add residual
        out = self.act(out)
        return out

class Prf_to_Seg_Model(nn.Module):
    def __init__(self, sig_dim=51, temp_hidden_dim=16, sig_hidden_dim=512, hidden_2_dim=256):
        """
        sig_dim: int, dimension of sigma profile input
        """
        super(Prf_to_Seg_Model, self).__init__()
        # sigma model
        self.model_sigma = nn.Sequential(
            nn.Linear(sig_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, sig_hidden_dim),
            nn.GELU(),
        )

        self.bn_sig = nn.BatchNorm1d(num_features=sig_hidden_dim)

        self.temp_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, temp_hidden_dim),
            nn.ReLU()
        )

        self.bn_t = nn.BatchNorm1d(num_features=temp_hidden_dim)

        self.model_combined = nn.Sequential(
            nn.Linear(sig_hidden_dim + temp_hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_2_dim),
            nn.GELU(),
        )

        self.bn2 = nn.BatchNorm1d(num_features=hidden_2_dim)

        # Resudual Block
        self.res_block = ResidualBlock(in_features=hidden_2_dim, hidden_features=hidden_2_dim, activation=nn.GELU)

        self.model_final = nn.Sequential(
            nn.Linear(hidden_2_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, sigs, t):
        """
        sigs: (batch_size, sig_dim)
        t   : (batch_size,) or (batch_size, 1)
        """
        # Ensure sigs require gradients
        sigs = sigs.requires_grad_(True)
        # sigs.requires_grad = True
        t = t.requires_grad_(False)

        # sigma profile
        sum_sigs = sigs.sum(dim=1, keepdim=True)  # (batch_size, 1)
        # Send normalized profile into sigma model.
        sigma_emb = self.model_sigma(sigs/sum_sigs)               # (batch_size, hidden_1_dim)
        sigma_emb = self.bn_sig(sigma_emb)

        # temperature
        t_inv = 1.0 / t.view(-1, 1)   # (batch_size, 1)
        t_emb = self.temp_embedding(t_inv) # (batch_size, temp_hidden_dim)
        t_emb = self.bn_t(t_emb)

        # Concatenate sigma_emb and t_emb
        combined_in = torch.cat([sigma_emb, t_emb], dim=-1)
        combined_out = self.model_combined(combined_in)

        # Batch Normal 2
        combined_out = self.bn2(combined_out)

        # Residual connection
        combined_out = self.res_block(combined_out)

        # combined_out_t = torch.cat([combined_out, t_inv], dim=-1)

        # Final FNN
        gchg_part = self.model_final(combined_out)

        # gchg = sum(sigs) * gchg_part 
        gchg = sum_sigs * gchg_part                 # (batch_size, 1)

        # Summing gchg and compute the gradient
        grad = autograd.grad(
            gchg.sum(),  # -> scalar
            sigs,
            create_graph=self.training
          )[0]  # shape: (batch_size, sig_dim)
        segs = grad
        return gchg, segs

