# pointnet_delta.py
import torch, torch.nn as nn, torch.nn.functional as F

class PointNetDelta(nn.Module):
    def __init__(self, in_dim=8, feat=128):
        super().__init__()
        # per-point encoder
        self.phi = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, feat), nn.ReLU()
        )
        # per-point decoder (uses global feature too)
        self.dec = nn.Sequential(
            nn.Linear(feat*2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3)   # Δx,Δy,Δz
        )

    def forward(self, x):           # x: (B,N,in_dim)
        h = self.phi(x)             # (B,N,F)
        g, _ = torch.max(h, dim=1)  # (B,F)
        g = g.unsqueeze(1).expand_as(h)  # (B,N,F)
        out = self.dec(torch.cat([h, g], dim=-1))  # (B,N,3)
        return out


class SimpleGraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nei  = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):  # x:(B,N,F), adj: (B,N,K) neighbor indices
        B,N,F = x.shape
        K = adj.shape[-1]
        # gather neighbors
        nei = []
        for k in range(K):
            nei.append(torch.gather(x, 1, adj[:,:,k].unsqueeze(-1).expand(B,N,F)))
        nei = torch.stack(nei, dim=2)               # (B,N,K,F)
        nei_feat = nei.mean(dim=2)                  # (B,N,F)
        return F.relu(self.lin_self(x) + self.lin_nei(nei_feat))
