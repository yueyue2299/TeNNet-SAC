import torch
import torch.nn as nn

from .Prf2Gamma import ResidualBlock, _new_final_head


class GammaTrunk(nn.Module):
    def __init__(
        self,
        sig_dim=51,
        temp_hidden_dim=16,
        sig_hidden_dim=512,
        hidden_2_dim=256,
    ):
        super().__init__()
        self.model_sigma = nn.Sequential(
            nn.Linear(sig_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, sig_hidden_dim),
            nn.GELU(),
        )
        self.bn_sig = nn.BatchNorm1d(sig_hidden_dim)
        self.temp_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, temp_hidden_dim),
            nn.ReLU(),
        )
        self.bn_t = nn.BatchNorm1d(temp_hidden_dim)
        self.model_combined = nn.Sequential(
            nn.Linear(sig_hidden_dim + temp_hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_2_dim),
            nn.GELU(),
        )
        self.bn2 = nn.BatchNorm1d(hidden_2_dim)
        self.res_block = ResidualBlock(
            in_features=hidden_2_dim,
            hidden_features=hidden_2_dim,
            activation=nn.GELU,
        )

    def forward(self, sigs, temperature):
        sum_sigs = sigs.sum(dim=1, keepdim=True)
        sigma_emb = self.bn_sig(self.model_sigma(sigs / sum_sigs))

        t = torch.as_tensor(temperature, dtype=sigs.dtype, device=sigs.device)
        if t.numel() == 1:
            t = t.reshape(1).expand(sigs.shape[0])
        elif t.numel() != sigs.shape[0]:
            raise ValueError("temperature must be scalar or match sigma batch size")
        t_emb = self.bn_t(self.temp_embedding((1.0 / t).reshape(-1, 1)))

        combined = self.bn2(
            self.model_combined(torch.cat([sigma_emb, t_emb], dim=-1))
        )
        return sum_sigs, self.res_block(combined)


class GammaEnsemble(nn.Module):
    def __init__(self, member_count: int = 10):
        super().__init__()
        if type(member_count) is not int or member_count < 1:
            raise ValueError("member_count must be a positive integer")
        self.trunk = GammaTrunk()
        self.heads = nn.ModuleList(
            _new_final_head() for _ in range(member_count)
        )

    def predict_segac(
        self,
        sigma,
        temperature,
        *,
        member_index=None,
        return_members=False,
    ) -> torch.Tensor:
        if self.training:
            raise RuntimeError("GammaEnsemble prediction requires eval mode")
        if return_members and member_index is not None:
            raise ValueError("member_index and return_members are mutually exclusive")
        if member_index is not None and (
            type(member_index) is not int
            or not 0 <= member_index < len(self.heads)
        ):
            raise IndexError("member_index out of range")

        sigs = sigma.clone().detach().requires_grad_(True)
        sum_sigs, shared = self.trunk(sigs, temperature)
        if member_index is not None:
            gibbs = sum_sigs * self.heads[member_index](shared)
            return torch.autograd.grad(gibbs.sum(), sigs)[0]

        gibbs = torch.stack([sum_sigs * head(shared) for head in self.heads])
        if return_members:
            count, batch = gibbs.shape[:2]
            basis = torch.eye(count, dtype=gibbs.dtype, device=gibbs.device)
            basis = basis.reshape(count, count, 1, 1).expand(
                count, count, batch, 1
            )
            return torch.autograd.grad(
                gibbs,
                sigs,
                grad_outputs=basis,
                is_grads_batched=True,
            )[0]
        return torch.autograd.grad(gibbs.mean(dim=0).sum(), sigs)[0]
