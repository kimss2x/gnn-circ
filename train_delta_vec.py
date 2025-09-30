# train_delta_vec.py
import os, glob, json, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def build_vertex_features(org_xyz):
    """
    org_xyz: (N,3) Basis vertex positions.
    만들 피처(예시, 견고/경량):
      - 중심정렬 xyz (3)
      - PCA 평면 투영 P=(px,py) (2)
      - 반지름 r (1)
      - 각도 theta의 cos/sin (2)
    총 8D 피처 -> 점별 학습용
    """
    X = org_xyz.astype(np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)           # (N,3)

    # PCA: 2D 평면 기저
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)  # Vt: (3,3)
    u, v = Vt[0], Vt[1]
    P = np.stack([Xc @ u, Xc @ v], axis=-1)          # (N,2)
    r = np.linalg.norm(P, axis=1, keepdims=True)     # (N,1)
    theta = np.arctan2(P[:,1], P[:,0]).reshape(-1,1) # (N,1)
    cs = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)  # (N,2)

    feats = np.concatenate([Xc, P, r, cs], axis=1).astype(np.float32)  # (N,8)
    return feats

class NPZVertexDataset(Dataset):
    """
    여러 파일의 (org(N,3), delta(N,3))를 로드해서
    '점 단위'로 펼쳐 하나의 데이터셋으로 만듭니다.
    X: (M, 8)  /  Y: (M, 3)   (M = 모든 샘플의 정점 총합)
    """
    def __init__(self, paths):
        feats_list, target_list = [], []
        for p in paths:
            d = np.load(p, allow_pickle=True)
            org   = d["org"].astype(np.float32)   # (N,3)
            delta = d["delta"].astype(np.float32) # (N,3)
            feats = build_vertex_features(org)    # (N,8)
            feats_list.append(feats)
            target_list.append(delta)
        X = np.concatenate(feats_list, axis=0)    # (M,8)
        Y = np.concatenate(target_list, axis=0)   # (M,3)

        # 표준화
        self.mean = X.mean(axis=0, keepdims=True)
        self.std  = X.std(axis=0, keepdims=True) + 1e-8
        self.Xn   = (X - self.mean) / self.std
        self.Y    = Y

    def __len__(self):  return self.Xn.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.Xn[i]), torch.from_numpy(self.Y[i])

class DeltaMLP(nn.Module):
    def __init__(self, in_dim=8, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3)   # Δx, Δy, Δz
        )
    def forward(self, x):  # x: (B,in_dim)
        return self.net(x) # (B,3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir",  default=".")
    ap.add_argument("--epochs",   type=int, default=80)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--bs",       type=int, default=2048)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "sample_*_vec.npz")))
    if not paths:
        raise SystemExit("No *_vec.npz files found.")

    ds = NPZVertexDataset(paths)
    dl = DataLoader(ds, batch_size=min(args.bs, len(ds)), shuffle=True, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeltaMLP(in_dim=ds.Xn.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss(beta=0.01)  # L1/L2 사이, 좌표 회귀에 무난

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for X,Y in dl:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            pred = model(X)          # (B,3)
            loss = loss_fn(pred, Y)  # Δ 회귀
            loss.backward()
            opt.step()
            running += float(loss) * X.size(0)
        print(f"[{epoch:03d}] loss={running/len(ds):.6e}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "delta_regressor.pth"))
    scaler = {"mean": ds.mean.flatten().tolist(),
              "std":  ds.std.flatten().tolist(),
              "in_dim": int(ds.Xn.shape[1])}
    with open(os.path.join(args.out_dir, "delta_regressor_scaler.json"), "w") as f:
        json.dump(scaler, f, indent=2)

    print("Saved:",
          os.path.join(args.out_dir, "delta_regressor.pth"),
          os.path.join(args.out_dir, "delta_regressor_scaler.json"))

if __name__ == "__main__":
    main()
