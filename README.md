아래 내용을 `README.md`로 저장하면 됩니다.

# gnn-circ

**gnn-circ**는 3D 메시의 **루프(폐곡선) 정점**들을 입력으로 받아, 원형에 가깝게 만드는 **정점 Δ(Delta) 회귀** 모델입니다.
PointNet + 간단한 GNN(Loop GraphSAGE) 하이브리드 구조, **스케일 정규화**, **형상 보존 손실**, **Blender 파이프라인**을 제공합니다.

* 입력: Basis 정점 좌표 `org (N,3)`
* 출력: `move = org + Δ_pred (N,3)` → Blender Shape Key로 적용
* 데이터: `sample_*_vec.npz` (`org`, `delta`)
* 학습: `train_delta_vec.py`
* 추론: `infer_delta_vec.py`

---

## ✨ 특징

* **PointNet + GNN**: 전역 특징(max-pool)과 루프 이웃(i±1, i±2) 맥락을 동시에 사용
* **스케일 정규화**: 샘플 간 크기 차이를 줄여 안정적 회귀
* **형상 보존 손실**(옵션): 엣지 길이 보존, 원형성(반지름 분산) 최소화
* **데이터 증강**: 평면 회전/스케일/지터
* **TTA**(추론 시 회전 앙상블) 지원
* **Blender 연동**: org/vec 데이터 export & 예측 좌표를 Shape Key로 바로 적용

---

## 📁 리포 구조

```
gnn-circ/
├─ train_delta_vec.py       # 학습 스크립트
├─ infer_delta_vec.py       # 추론 스크립트
├─ pointnet_delta.py        # (선택) 모델 구성 분리 시 사용
├─ data/
│  ├─ sample_flw_vec.npz    # 예시 (org, delta)
│  ├─ sample_qrd_vec.npz
│  └─ sample_tri_vec.npz
├─ out_vec/
│  ├─ delta_regressor.pth
│  ├─ delta_regressor_scaler.json
│  ├─ org_flw.npy           # Blender에서 export한 Basis 좌표
│  └─ move_pred_flw.npy     # 예측 결과
└─ README.md
```

---

## 🛠️ 환경 설정 (Windows / Conda / Python 3.10)

```cmd
conda create -n tr310 python=3.10 -y
conda activate tr310

:: PyTorch 설치 (CUDA 버전에 맞게 설치)
:: 공식 가이드: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: 기타 필수
pip install numpy
```

> CUDA 버전에 맞는 PyTorch 인덱스를 사용하세요. CPU만이면 기본 `pip install torch torchvision torchaudio` 로도 동작합니다.

---

## 📦 데이터 형식

`data/sample_*_vec.npz` 파일에 다음 키가 들어갑니다.

* `org`   : `(N,3)` Basis 좌표 (메시 정점 순서 그대로)
* `delta` : `(N,3)` 목표 이동량 (예: “원형화” 타겟 포즈 − Basis)

> **중요**: 추론/적용할 메시의 **정점 개수와 순서**는 `org`와 반드시 동일해야 합니다.

---

## 🧠 학습 (Training)

기본 학습 예:

```cmd
conda activate tr310
python train_delta_vec.py ^
  --data_dir data ^
  --out_dir  out_vec ^
  --epochs   250 ^
  --lambda_edge 0.4 ^
  --lambda_circle 0.06
```

출력:

* `out_vec/delta_regressor.pth` : 학습된 가중치
* `out_vec/delta_regressor_scaler.json` : 피처 표준화 통계

### 주요 옵션

* `--lambda_edge` : 이웃 엣지 길이 보존 가중치 (0.0~0.6 권장)
* `--lambda_circle` : 원형성(반지름 분산) 최소화 가중치 (0.0~0.1 권장)
* `--no_aug` : 데이터 증강 비활성화 (디버그/오버핏 테스트에 유용)
* `--bs`, `--epochs`, `--lr` : 배치/에폭/학습률

> 처음에는 `--no_aug`와 규제(λ)를 0으로 시작해 **기본 회귀가 잘 되는지** 확인한 뒤, 점진적으로 규제를 켜는 것을 권장합니다.

---

## 🔮 추론 (Inference)

Basis 좌표 `org_*.npy (N,3)`가 준비되어 있어야 합니다. (Blender에서 export 스니펫은 아래 참고)

```cmd
conda activate tr310
python infer_delta_vec.py ^
  --org_npy out_vec\org_flw.npy ^
  --model   out_vec\delta_regressor.pth ^
  --scaler  out_vec\delta_regressor_scaler.json ^
  --out_npy out_vec\move_pred_flw.npy ^
  --device  cuda ^
  --tta     8
```

* `--tta`: 회전 앙상블 횟수 (0=off, 4~16 권장)
* 출력 `move_pred_*.npy`는 `(N,3)` 좌표이며, 그대로 Shape Key에 적용합니다.

---

## 🧩 Blender 연동

### A. Basis 좌표 export (선택 객체 1개)

```python
# Blender Text Editor에서 실행
import bpy, numpy as np, os
obj = bpy.context.active_object
assert obj and obj.type == 'MESH'
base = r"C:\circ_dataset\out_vec"
os.makedirs(base, exist_ok=True)

verts = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float32)
np.save(os.path.join(base, f"org_{obj.name}.npy"), verts)
print("saved:", os.path.join(base, f"org_{obj.name}.npy"), verts.shape)
```

### B. 예측 좌표 적용 (Shape Key 생성/재사용)

```python
import bpy, numpy as np
obj = bpy.context.active_object
move = np.load(r"C:\circ_dataset\out_vec\move_pred_flw.npy")  # (N,3)
assert len(obj.data.vertices) == move.shape[0], "vertex count mismatch!"

if not obj.data.shape_keys:
    obj.shape_key_add(name="Basis", from_mix=False)

kb_name = "PredictedCircle"
kb = obj.data.shape_keys.key_blocks.get(kb_name) or obj.shape_key_add(name=kb_name, from_mix=False)

for i in range(move.shape[0]):
    kb.data[i].co = move[i]

kb.value = 1.0
bpy.context.view_layer.update()
print(f"[{obj.name}] {kb_name} applied.")
```

> 여러 객체를 일괄 적용하려면 `move_pred_{오브젝트명}.npy` 규칙으로 저장해 루프를 돌리면 됩니다.

---

## 🎯 모델 개요

* 입력 피처(점당 8D): 중심정렬 xyz(3) + PCA 평면 투영(2) + 반지름(1) + (cosθ, sinθ)
* 인코더: Per-point MLP → GraphSAGE(이웃 i±1, i±2) × 2층 → 전역 max-pool
* 디코더: 전역 특징 concat 후 per-point Δ 회귀
* 손실: SmoothL1(좌표) + λ_edge·λ_circle(옵션)
* 스케일 정규화: 각 샘플의 PCA 평면에서 **median r = 1**이 되도록 정규화 후 학습/추론, 결과는 원스케일로 복원

---

## 🔍 문제 해결(FAQ)

* **Blender에서 변화가 거의 없음**

  * org와 move의 차이 크기를 수치로 확인:

    ```python
    import numpy as np
    org  = np.load(r"C:\circ_dataset\out_vec\org_flw.npy")
    move = np.load(r"C:\circ_dataset\out_vec\move_pred_flw.npy")
    r = np.linalg.norm(move-org, axis=1)
    print("mean|Δ| =", r.mean(), "max|Δ| =", r.max())
    ```
  * 작다면 학습 세팅이 보수적일 수 있습니다.

    * 규제 0으로 baseline 학습 → 이후 λ_edge, λ_circle 점진 적용
    * 모델 용량/에폭 증가
    * 데이터 증강은 baseline이 잘 된 후 켜기
* **정점 수/순서 불일치**

  * org 추출 후 메시 수정/모디파이어 변경 시 다시 export & 추론 필요
* **CUDA/AMP 관련 경고/오류**

  * 최신 PyTorch 버전에 맞춰 `torch.amp.autocast('cuda')`/`torch.amp.GradScaler('cuda')` 사용

---

## 📈 팁 (정확도 향상)

* **모델 용량↑**: hidden 256~512, 레이어 4~5
* **피처 확장**: 이웃 길이/각도, 간이 곡률, 포지셔널 인코딩 등
* **이웃 확장**: i±3까지 실험 (k-ring 증가)
* **스케줄러**: CosineAnnealingLR, gradient clipping
* **검증 지표**: Δ MSE, Edge-length Error, Radial Variance

---

## 🔒 라이선스

MIT (원하시는 라이선스로 변경 가능)

---

## 🙌 크레딧

* PointNet 아이디어(전역 풀링 기반 순열 불변)
* GraphSAGE 스타일의 간단 메시 패싱
* Blender 파이프라인 스크립트(Export/Apply)

---

## 🚀 빠른 시작 요약

1. `conda activate tr310`
2. `python train_delta_vec.py --data_dir data --out_dir out_vec`
3. Blender에서 `org_*.npy` export
4. `python infer_delta_vec.py --org_npy out_vec\org_flw.npy --model out_vec\delta_regressor.pth --scaler out_vec\delta_regressor_scaler.json --out_npy out_vec\move_pred_flw.npy --device cuda --tta 8`
5. Blender에서 `move_pred_*.npy`를 Shape Key로 적용

필요하시면 **배치(.bat) 스크립트**, **VSCode 런치 설정**, **샘플 노트북** 템플릿도 추가해드릴게요.
