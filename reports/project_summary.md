# Project Summary: Cross-View Ambient Diffusion for Multi-Acquisition Low-Field MRI

## Dataset and scope

- Dataset: **M4Raw 0.3 T** T2-weighted brain, multi-coil (4 coils), 256x256, 2D.
- This is **low-field (0.3 T)**, NOT ultra-low-field (<0.1 T).
- Preprocessing: test acceptance rate 99.7% (299 accepted / 1 rejected, 25 subjects); train/val/test slices: 708/239/299.
- Two evaluation sets: the **2-view set** (25 test subjects, 75 central slices) and the **4-view set** (20 subjects, 59 slices — requiring 4 mutually motion-consistent repetitions accepts 188/300 slices).
- Reference: the average of all 6 M4Raw test repetitions (RSS magnitude).
- All metrics are **brain-masked** (Otsu on the reference, identical mask for every method on a slice) and aggregated **per subject** with 95% bootstrap CIs.

## Mask design and acquisition budget

Cartesian column undersampling with a fully-sampled 16-line ACS block. `Lines` is what you pay for (summed over views); `unique cols` is what you actually measure.

| Condition | Views | Lines bought | Unique cols | Coverage |
|---|---|---|---|---|
| `single_r4` | 1 | 64 | 64 | 25% |
| `merge_fixed` / `joint_fixed` (fixed budget, 2xR=8) | 2 | 64 | 48 | 19% |
| `*_extra` (2x, duplicated mask) | 2 | 128 | 64 | 25% |
| `*_extra_comp` (2x, complementary, shared ACS) | 2 | 128 | 112 | 44% |
| `*_extra_fullcomp` (2x, fully-complementary, split ACS) | 2 | 128 | 128 | 50% |
| `*_quad` (4x, duplicated mask) | 4 | 256 | 64 | 25% |
| `*_quad_comp` (4x, complementary, shared ACS) | 4 | 256 | 208 | 81% |
| `*_quad_fullcomp` (4x, fully-complementary, split ACS) | 4 | 256 | 256 | 100% |
| `single_full` (one complete measurement) | 1 | 256 | 256 | 100% |
| `dual_full` (two complete measurements, NEX=2) | 2 | 512 | 256 | 100% |

Fixed-budget accounting (coefficients = columns x 256 rows):
- **train**: single_r4=16384, fixed_sum(dupACS)=16384 (within tolerance: True), fixed_union=12288, extra_sum=32768.
- **val**: single_r4=16384, fixed_sum(dupACS)=16384 (within tolerance: True), fixed_union=12288, extra_sum=32768.
- **test**: single_r4=16384, fixed_sum(dupACS)=16384 (within tolerance: True), fixed_union=12288, extra_sum=32768.

## Selected hyperparameters

- Diffusion inference: num_steps=100, l_ss=30.0, likelihood_type=DPS (tuned on validation held-out k-space error).
- L1-wavelet SENSE: lambda=0.03, iterations=50.
- Cross-view fine-tune: 0.02 Mimg, lr=1e-4, EMA half-life 0.002 Mimg.

## Results: 2-view set (25 subjects, 75 slices)

Subject-level mean [95% bootstrap CI].

### Classical baselines and fixed budget (64 lines)

| Method | Description | Views | Lines | Unique cols | Coverage | SSIM | NRMSE | Held-out k err |
|---|---|---|---|---|---|---|---|---|
| `classical_adjoint` | Classical adjoint | 2 | 64 | 48 | 19% | 0.5193 [0.5130, 0.5255] | 0.2258 [0.2208, 0.2313] | 0.9845 [0.9833, 0.9857] |
| `classical_l1wav` | Classical L1-wavelet | 2 | 64 | 48 | 19% | 0.5090 [0.5002, 0.5179] | 0.2215 [0.2169, 0.2264] | 0.9713 [0.9670, 0.9755] |
| `single_r4` | Single scan (R=4) | 1 | 64 | 64 | 25% | 0.6176 [0.5993, 0.6394] | 0.1827 [0.1764, 0.1892] | 0.8668 [0.8579, 0.8748] |
| `fixed_split_merge` | Fixed budget, merged | 2 | 64 | 48 | 19% | 0.5973 [0.5824, 0.6160] | 0.1909 [0.1865, 0.1950] | 0.8838 [0.8753, 0.8918] |
| `fixed_split_ft` | Fixed budget, xview-FT | 2 | 64 | 48 | 19% | 0.6050 [0.5900, 0.6235] | 0.1878 [0.1836, 0.1918] | 0.8781 [0.8696, 0.8862] |

### 2x budget (128 lines): duplicated vs complementary vs fully-complementary

| Method | Description | Views | Lines | Unique cols | Coverage | SSIM | NRMSE | Held-out k err |
|---|---|---|---|---|---|---|---|---|
| `dup2_merge` | 2x duplicated, merged | 2 | 128 | 64 | 25% | 0.6542 [0.6394, 0.6713] | 0.1668 [0.1622, 0.1715] | 0.8639 [0.8559, 0.8714] |
| `dup2_ft` | 2x duplicated, xview-FT | 2 | 128 | 64 | 25% | 0.6628 [0.6466, 0.6818] | 0.1648 [0.1599, 0.1698] | 0.8574 [0.8492, 0.8649] |
| `comp2_merge` | 2x complementary, merged | 2 | 128 | 112 | 44% | 0.7059 [0.6929, 0.7218] | 0.1493 [0.1447, 0.1541] | 0.8484 [0.8402, 0.8561] |
| `comp2_ft` | 2x complementary, xview-FT | 2 | 128 | 112 | 44% | 0.7042 [0.6908, 0.7202] | 0.1494 [0.1452, 0.1537] | 0.8480 [0.8400, 0.8556] |
| `fcomp2_merge` | 2x fully-complementary, merged | 2 | 128 | 128 | 50% | 0.7067 [0.6944, 0.7222] | 0.1486 [0.1446, 0.1527] | 0.8458 [0.8374, 0.8536] |
| `fcomp2_ft` | 2x fully-complementary, xview-FT | 2 | 128 | 128 | 50% | 0.7083 [0.6957, 0.7236] | 0.1485 [0.1447, 0.1525] | 0.8440 [0.8355, 0.8519] |

## Results: 4-view set (20 subjects, 59 slices)

Subject-level mean [95% bootstrap CI].

| Method | Description | Views | Lines | Unique cols | Coverage | SSIM | NRMSE | Held-out k err |
|---|---|---|---|---|---|---|---|---|
| `single_r4` | Single scan (R=4) | 1 | 64 | 64 | 25% | 0.6431 [0.6218, 0.6667] | 0.1671 [0.1635, 0.1706] | 0.8729 [0.8634, 0.8821] |
| `dup4_merge` | 4x duplicated, merged | 4 | 256 | 64 | 25% | 0.6563 [0.6360, 0.6773] | 0.1577 [0.1535, 0.1618] | 0.8833 [0.8736, 0.8932] |
| `dup4_ft` | 4x duplicated, xview-FT | 4 | 256 | 64 | 25% | 0.7032 [0.6845, 0.7236] | 0.1447 [0.1418, 0.1476] | 0.8578 [0.8495, 0.8661] |
| `comp4_merge` | 4x complementary, merged | 4 | 256 | 208 | 81% | 0.7656 [0.7513, 0.7816] | 0.1215 [0.1192, 0.1238] | 0.8370 [0.8251, 0.8489] |
| `comp4_ft` | 4x complementary, xview-FT | 4 | 256 | 208 | 81% | 0.7643 [0.7492, 0.7809] | 0.1205 [0.1185, 0.1224] | 0.8388 [0.8268, 0.8505] |
| `fcomp4_merge` | 4x fully-complementary, merged | 4 | 256 | 256 | 100% | 0.7650 [0.7508, 0.7809] | 0.1306 [0.1256, 0.1363] | n/a |
| `fcomp4_ft` | 4x fully-complementary, xview-FT | 4 | 256 | 256 | 100% | 0.7559 [0.7408, 0.7735] | 0.1406 [0.1334, 0.1484] | n/a |
| `full_diffusion` | Full scan, diffusion | 1 | 256 | 256 | 100% | 0.7577 [0.7436, 0.7733] | 0.1346 [0.1303, 0.1396] | n/a |
| `full_plain` | Full scan, plain recon | 1 | 256 | 256 | 100% | 0.7964 [0.7832, 0.8118] | 0.1198 [0.1165, 0.1240] | n/a |
| `dualfull_merge` | 2 full scans, classical avg (NEX=2) | 2 | 512 | 256 | 100% | 0.8778 [0.8689, 0.8870] | 0.0986 [0.0936, 0.1034] | n/a |
| `dualfull_ft` | 2 full scans, xview-FT | 2 | 512 | 256 | 100% | 0.7586 [0.7439, 0.7748] | 0.1425 [0.1361, 0.1496] | n/a |

## Figures

**Main results — reconstruction montages (per results table)**

- `figures/project/recon_montage_2view.png`
- `figures/project/recon_montage_4view.png`

**Results tables (visualized)**

- `figures/project/results_2view.png`
- `figures/project/results_4view.png`

**Cross-experiment**

- `figures/project/effect_sizes_forest.png`
- `figures/project/budget_ladder.png`
- `figures/project/finetuning_vs_coverage.png`

**Acquisition design and tuning**

- `figures/project/mask_design_all.png`
- `figures/project/lss_tier_tuning.png`

**Supporting — absolute error maps**

- `figures/project/error_maps.png`
- `figures/project/error_maps_quad.png`

