import numpy as np, torch
from infer_delta_vec import load_model, build_vertex_features, normalize_by_radius, make_ring_adj, PointNetGNNDelta
org = np.load(r"C:\circ_dataset\out_vec\org_flw.npy").astype(np.float32)
model, mean, std = load_model(r"C:\circ_dataset\out_vec\delta_regressor.pth",
                              r"C:\circ_dataset\out_vec\delta_regressor_scaler.json",
                              device="cpu")
org_n, s = normalize_by_radius(org)
feats = build_vertex_features(org_n)
x = torch.from_numpy((feats-mean)/(std+1e-8)).unsqueeze(0)
mask = torch.ones(1, x.shape[1], dtype=torch.bool)
with torch.no_grad():
    delta_n = model(x, mask).squeeze(0).numpy()
print("pred Δ_n mean abs:", np.abs(delta_n).mean(), "max:", np.abs(delta_n).max())
