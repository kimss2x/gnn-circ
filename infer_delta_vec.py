import json, numpy as np, torch, torch.nn as nn

def build_vertex_features(org_xyz):
    X = org_xyz.astype(np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    u,v = Vt[0], Vt[1]
    P = np.stack([Xc @ u, Xc @ v], axis=-1)
    r = np.linalg.norm(P, axis=1, keepdims=True)
    theta = np.arctan2(P[:,1], P[:,0]).reshape(-1,1)
    cs = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    feats = np.concatenate([Xc, P, r, cs], axis=1).astype(np.float32)  # (N,8)
    return feats

class DeltaMLP(nn.Module):
    def __init__(self, in_dim=8, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3)
        )
    def forward(self, x): return self.net(x)

def load_model(model_path, scaler_path):
    with open(scaler_path, "r") as f:
        sc = json.load(f)
    d = int(sc["in_dim"])
    mean = np.array(sc["mean"], dtype=np.float32)
    std  = np.array(sc["std"],  dtype=np.float32)
    model = DeltaMLP(in_dim=d)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state); model.eval()
    return model, mean, std

def predict_delta(org_xyz, model, mean, std):
    feats = build_vertex_features(org_xyz)  # (N,8)
    Xn = (feats - mean) / (std + 1e-8)
    with torch.no_grad():
        delta = model(torch.from_numpy(Xn)).cpu().numpy()  # (N,3)
    return delta

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--org_npy", required=True)
    ap.add_argument("--model",   required=True)
    ap.add_argument("--scaler",  required=True)
    ap.add_argument("--out_npy", required=True)
    args = ap.parse_args()

    org = np.load(args.org_npy).astype(np.float32)        # (N,3)
    model, mean, std = load_model(args.model, args.scaler)
    delta = predict_delta(org, model, mean, std)           # (N,3)
    move  = org + delta                                    # (N,3)
    np.save(args.out_npy, move)
    print("Saved:", args.out_npy)
