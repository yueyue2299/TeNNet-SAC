import torch
import torch.nn as nn

class ResidualBlock(nn.Module): # ResNet v2
    def __init__(self, in_features, hidden_features=None, activation=nn.GELU):
        """
        in_features: input feature
        hidden_features: hidden feature, if None, use in_features
        """
        super(ResidualBlock, self).__init__()
        hidden_features = hidden_features or in_features
        self.act = activation()
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.act(out)
        out = self.fc1(out)

        out = self.bn2(out)
        out = self.act(out)
        out = self.fc2(out)

        out += identity  # add residual
        return out

class SigmaProfileGenerator(nn.Module):
    def __init__(self, STemb_dim=768, CBemb_dim=384, CBout_dim=32,
                 prf_dim=51, hidden_dim=256, dropout_rate=0.3):
        """
        sig_dim: int, dimension of sigma profile input
        """
        super(SigmaProfileGenerator, self).__init__()

        self.isomerism_encoder = nn.Sequential(
          nn.Linear(CBemb_dim, 256),
          nn.GELU(),
          nn.Linear(256, CBout_dim)
        )

        self.backbone = nn.Sequential(
          nn.Linear(STemb_dim + CBout_dim, 512),
          nn.GELU(),
          nn.Linear(512, 256),
          nn.GELU(),
          nn.Linear(256, hidden_dim),
          nn.GELU(),
          ResidualBlock(in_features=hidden_dim, hidden_features=hidden_dim, activation=nn.GELU),
          nn.GELU(),
          ResidualBlock(in_features=hidden_dim, hidden_features=hidden_dim, activation=nn.GELU),
          nn.GELU(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.GELU()
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, prf_dim),
            nn.ReLU()
        )

    def forward(self, STemb, CBemb):
        conf_feat = self.isomerism_encoder(CBemb)
        x = torch.cat([STemb, conf_feat], dim=1)
        x = self.backbone(x)
        output = self.head(x)
        area = output.sum(dim=1, keepdim=True)
        prf = output / area

        return prf
