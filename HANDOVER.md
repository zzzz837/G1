# GTCRN Adaptive — 工作交接文档

## 仓库信息

| 项目 | 值 |
|------|----|
| GitHub 仓库 | `https://github.com/zzzz837/GTCRN` |
| 分支 | `main` (已推送完整代码) |
| 上游原始仓库 | `https://github.com/Xiaobin-Rong/gtcrn` |
| 官方文件 | `gtcrn.py`, `loss.py`, `infer.py` — **不得修改** |
| 预训练权重 | `checkpoints/model_trained_on_dns3.tar` |

---

## 快速开始（新窗口操作）

### 1. 克隆仓库

```bash
git clone https://github.com/zzzz837/GTCRN.git
cd GTCRN
```

### 2. 环境要求

```bash
# Python 3.9+
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy soundfile einops scipy matplotlib scikit-learn pytest
```

### 3. 设置环境变量（Windows）

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

### 4. 验证一切正常

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available())"
# 必须输出: cuda= False

python -m pytest work2/tests -v --ignore=work2/tests/test_baseline_output.py
# 应输出: 73 passed

python work2/scripts/run_baseline.py
python work2/tests/test_baseline_output.py
# 应输出: PASS + max error 0.0
```

---

## 项目结构

```
GTCRN/
├── gtcrn.py                    # 官方 GTCRN (勿改)
├── infer.py                    # 官方推理 (勿改)
├── loss.py                     # 官方损失 (勿改)
├── checkpoints/
│   └── model_trained_on_dns3.tar  # 预训练权重 (~10MB)
├── test_wavs/
│   ├── mix.wav                 # 测试音频
│   └── enh.wav                 # 基线增强输出
├── work2/
│   ├── data/                   # 基础工具层
│   │   ├── audio_utils.py      # 6音频工具 (mono/normalize/rms/snr/check)
│   │   ├── random_utils.py     # seed_everything()
│   │   ├── stft_utils.py       # PyTorch 2.x STFT兼容层
│   │   ├── degradation.py      # 5退化 + 4预设 + dataclass标签 ★核心★
│   │   ├── label_utils.py      # SNR归一化/标签转换
│   │   └── build_manifest.py   # JSONL清单构建器
│   │
│   ├── models/                 # 模型层
│   │   ├── frozen_gtcrn_extractor.py  # 冻结GTCRN + hook捕获dpgrnn2
│   │   ├── degradation_estimator.py   # 退化估计器 (1,985参数)
│   │   ├── adaptive_residual.py       # 自适应残差补偿 (2,184参数) ★创新★
│   │   └── gtcrn_adaptive.py          # 端到端管道
│   │
│   ├── losses/                 # 损失层
│   │   ├── degradation_estimator_loss.py  # 多任务masked loss
│   │   └── residual_loss.py              # 高频+残差联合loss
│   │
│   ├── datasets/
│   │   └── degradation_feature_dataset.py  # 缓存Dataset
│   │
│   ├── train/                  # 训练脚本
│   │   ├── train_degradation_estimator.py   # 训练退化估计器
│   │   └── train_adaptive.py               # 训练自适应残差
│   │
│   ├── evaluation/
│   │   └── evaluate_degradation_estimator.py  # 评测+混淆矩阵+绘图
│   │
│   ├── scripts/                # 工具脚本
│   │   ├── run_baseline.py           # 基线推理(已重构,用stft_utils)
│   │   ├── run_infer.py              # 官方infer.py兼容包装
│   │   ├── run_estimator_smoke_test.py  # 估计器冒烟测试
│   │   ├── run_adaptive_smoke.py        # 自适应残差冒烟测试
│   │   ├── run_pipeline_verify.py       # ★ 真实数据端到端验证
│   │   ├── resample_vctk.py             # VCTK 48k→16k重采样
│   │   ├── cache_degradation_features.py # 批量特征缓存
│   │   └── generate_degradation_demo.py  # 退化示例生成
│   │
│   ├── tests/                  # 单元测试
│   │   ├── test_degradation.py          # 退化(39 tests)
│   │   ├── test_frozen_gtcrn_extractor.py  # 冻结GTCRN(8 tests)
│   │   ├── test_degradation_estimator.py   # 估计器(4 tests)
│   │   ├── test_estimator_loss.py          # masked loss(2 tests)
│   │   ├── test_feature_cache.py           # 缓存(3 tests)
│   │   ├── test_adaptive_residual.py       # 残差(17 tests)
│   │   └── test_baseline_output.py         # 基线验证(1 test)
│   │
│   └── reports/                # 文档
│       ├── baseline_report.md
│       ├── degradation_report.md
│       ├── degradation_estimator_report.md
│       └── project_overview.md   # ★ 项目整体说明
│
├── outputs/                    # 输出(大文件已被gitignore)
│   ├── baseline/               # 基线增强结果
│   ├── degradation_demo/       # 退化示例音频+频谱图
│   ├── estimator_smoke/        # 估计器冒烟输出
│   ├── adaptive_smoke/         # 自适应残差冒烟输出
│   ├── pipeline_verify/        # 真实数据验证结果
│   └── degradation_estimator/  # 退化估计器训练输出
│
└── data/                       # 数据集(gitignore排除)
    ├── vctk_sample/            # VCTK样本(原始48kHz)
    └── vctk_16k/               # 重采样后16kHz
```

---

## 创新框架总览

```
带噪语音 → STFT
  → 冻结GTCRN (48K参数, 不可训练)
    ├→ enhanced_base (基础增强)
    └→ dpgrnn2 bottleneck → stats(32维)
        → DegradationEstimator (1,985参数)
          ├→ noise_logit → 是否有噪声
          ├→ snr_pred → SNR预测
          ├→ bw_logits → 带宽类别(全频/6k/4k)
          └→ bit_logits → 量化位数(16/12/10/8)
            → condition vector (9维)
              → AdaptiveResidualModule (2,184参数)
                → enhanced_final = base + α·residual
                  → iSTFT → 最终增强语音
```

**总可训练参数**: 4,169 (仅占GTCRN的8.6%)

---

## 完整工作流

### Step 1: 数据准备

```bash
# 如果需要VCTK数据集,先安装openxlab
pip install openxlab

# 配置AK/SK (一次性)
# 创建 ~/.openxlab/config.json:
# {"ak":"gva34kod0qjeo7yxqewl","sk":"omjg9q21bzzrypmw4vr0y6edadlk83o0b6xnldja"}

# 下载VCTK样本
openxlab dataset download --dataset-repo OpenDataLab/VCTK \
  --source-path /sample --target-path data/vctk_sample

# 重采样 48kHz→16kHz
python work2/scripts/resample_vctk.py \
  --input-dir data/vctk_sample/OpenDataLab___VCTK/sample/audio \
  --output-dir data/vctk_16k
```

### Step 2: 建清单

```bash
python -m work2.data.build_manifest \
  --clean-dir data/vctk_16k \
  --output outputs/manifest_vctk.jsonl \
  --seed 2026
```

### Step 3: 缓存GTCRN特征

```bash
python -m work2.scripts.cache_degradation_features \
  --clean-manifest outputs/manifest_vctk.jsonl \
  --checkpoint checkpoints/model_trained_on_dns3.tar \
  --output-dir outputs/estimator_cache \
  --samples-per-file 4 --seed 2026
```

### Step 4: 训练退化估计器

```bash
python -m work2.train.train_degradation_estimator \
  --cache-dir outputs/estimator_cache \
  --output-dir outputs/degradation_estimator \
  --epochs 50 --batch-size 64 --seed 2026
```

### Step 5: 训练自适应残差

```bash
python -m work2.train.train_adaptive \
  --gtcrn-checkpoint checkpoints/model_trained_on_dns3.tar \
  --estimator-checkpoint outputs/degradation_estimator/best_model.pt \
  --output-dir outputs/adaptive_residual \
  --epochs 50 --freeze-estimator --seed 2026
```

### Step 6: 验证

```bash
python -m work2.scripts.run_pipeline_verify \
  --clean-dir data/vctk_16k \
  --estimator-checkpoint outputs/degradation_estimator/best_model.pt \
  --output-dir outputs/pipeline_verify
```

---

## 常用命令速查

```bash
# 环境变量 (每次新窗口)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 运行全部测试 (73个)
python -m pytest work2/tests -v --ignore=work2/tests/test_baseline_output.py

# 基线验证
python work2/scripts/run_baseline.py
python work2/tests/test_baseline_output.py

# 冒烟测试 (纯合成数据, 验证管线)
python -m work2.scripts.run_estimator_smoke_test --seed 2026
python -m work2.scripts.run_adaptive_smoke --seed 2026

# 检查官方文件是否被修改
git diff -- gtcrn.py loss.py infer.py
# 应输出: (空)
```

---

## 当前进展总结

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | GTCRN CPU基线 + bit-exact验证 | ✅ 通过 |
| Phase 2 | 5种退化 + 4级预设 + 标签系统 | ✅ 39 tests |
| Phase 3 | 冻结GTCRN + 退化估计器 | ✅ 1,985参数 |
| Phase 4 | 自适应残差补偿模块 | ✅ 2,184参数 |
| Pipeline | VCTK真实数据端到端验证 | ✅ 跑通 |
| 缺失 | 大规模真实数据训练 | 🔲 待做 |
| 缺失 | 最终PESQ/STOI评测 | 🔲 待做 |
| 缺失 | 残差模块正式训练 | 🔲 仅冒烟 |

---

## 注意事项

1. **官方文件不可修改**: `gtcrn.py`, `loss.py`, `infer.py`
2. **基线必须 bit-exact**: 每次修改后运行 `test_baseline_output.py`
3. **KMP_DUPLICATE_LIB_OK**: OpenMP冲突,每次启动新终端要设置
4. **仅CPU**: `torch.cuda.is_available()` 必须为 `False`
5. **数据文件被gitignore**: `data/`, `*.wav`, `*.zip` 不上传Git
6. **degration mask位定义**: bit0=噪声, bit1=带宽, bit2=量化
7. **spect格式**: stft_utils使用(B,F,T,2), adaptive_residual内部转换为(B,2,T,F)
