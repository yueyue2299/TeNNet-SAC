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

class GeometryGenerator(nn.Module):
    def __init__(self, STemb_dim=768, CBemb_dim=384, CBout_dim=32, av_dim=2,
                 hidden_dim=128, mw_dim=128, dropout_rate=0.1):
        """
        sig_dim: int, dimension of sigma profile input
        """
        super(GeometryGenerator, self).__init__()

        self.isomerism_encoder = nn.Sequential(
          nn.Linear(CBemb_dim, 256),
          nn.GELU(),
          nn.Linear(256, CBout_dim)
        )

        self.backbone = nn.Sequential(
          nn.Linear(STemb_dim + CBout_dim, 512),
          nn.GELU(),
          nn.Linear(512, hidden_dim),
          nn.GELU(),
        )

        self.mw_projector = nn.Sequential(
          nn.Linear(1, 64),
          nn.GELU(),
          nn.Linear(64, mw_dim),
          nn.GELU(),
        )

        self.comb_layers = nn.Sequential(
          nn.Linear(hidden_dim + mw_dim, hidden_dim),
          nn.GELU(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.GELU(),
        )

        self.res_block = ResidualBlock(in_features=hidden_dim, hidden_features=hidden_dim, activation=nn.GELU)

        self.area_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            # nn.Dropout(p=0.1),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self.volume_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            # nn.Dropout(p=0.1),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, STemb, CBemb, mw):
        mw = mw.unsqueeze(1)
        conf_feat = self.isomerism_encoder(CBemb)

        emb = torch.cat([STemb, conf_feat], dim=1)

        hidden_emb = self.backbone(emb)
        hidden_mw = self.mw_projector(mw)

        hidden_comb = torch.cat([hidden_emb, hidden_mw], dim=1)
        hidden = self.comb_layers(hidden_comb)
        hidden = self.res_block(hidden)

        area = self.area_head(hidden)
        volume = self.volume_head(hidden)

        output = torch.cat([area, volume], dim=1)

        return output