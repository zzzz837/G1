# EDCR-Net: Explicit Degradation-Conditioned Residual Speech Enhancement

基于显式复合退化建模与条件化频带残差修复的语音增强方法

## 项目概述

本仓库实现了面向复合退化语音的轻量级增强框架 EDCR-Net。与传统"统一黑箱增强"不同，EDCR 首先通过显式退化估计器诊断输入语音的噪声、SNR、带宽和位深状态，随后利用条件化残差修复模块在频谱域对基础增强结果进行针对性补偿。整个扩展链路仅增加约 4K 参数，基础增强器全程保持冻结。

### 论文核心结论

1. **退化估计器成立**：基础增强器瓶颈统计特征中包含显著可分的复合退化信息，可被轻量多任务网络高精度解析（噪声精度 > 0.9999，SNR MAE < 1dB）
2. **残差修复有效**：条件化残差修复在 LSD/HF-LSD 上显著优于基础增强器（改善比例 76%~79%，p<0.001）
3. **条件利用有待增强**：条件有效性实验和无条件残差对照表明，当前增益主要来自残差模块结构本身，显式条件尚未被充分转化为差异化修复策略

---

## 目录结构

```
├── gtcrn.py / infer.py / loss.py    # 官方文件（未修改）
├── checkpoints/                     # 官方预训练权重
├── requirements.txt                 # Python 依赖
├── work2/                           # 全部核心代码
│   ├── data/                        # 基础工具层
│   │   ├── degradation.py           #   复合退化生成器（噪声/带宽/量化/低通）
│   │   ├── audio_utils.py           #   音频工具
│   │   ├── stft_utils.py            #   PyTorch 2.x STFT 兼容层
│   │   ├── label_utils.py           #   SNR 归一化与标签转换
│   │   └── random_utils.py          #   可复现随机种子
│   ├── models/                      # 模型层
│   │   ├── adaptive_residual.py     #   V1 / V2-initial / V2-Refined-A 残差模块
│   │   ├── frozen_gtcrn_extractor.py#   冻结基础增强器 + 瓶颈特征抽取
│   │   ├── degradation_estimator.py #   退化估计器（1.8K 参数）
│   │   └── gtcrn_adaptive.py        #   端到端推理管道
│   ├── losses/                      # 损失层
│   │   ├── residual_loss.py         #   残差损失（mag/hf/complex/res/protect/gate/ratio）
│   │   └── degradation_estimator_loss.py
│   ├── datasets/                    # 数据加载
│   │   └── degradation_feature_shard_dataset.py
│   ├── train/                       # 训练脚本
│   │   └── train_adaptive_cached.py #   残差模块 cached 训练（样本加权平均）
│   └── evaluation/                  # 评测脚本
│       ├── evaluate_enhancement_models.py       # 系统级增强评测
│       ├── evaluate_condition_effectiveness.py  # 条件有效性实验
│       ├── evaluate_unconditional_residual.py   # 无条件残差对照
│       ├── compute_statistical_tests.py         # 显著性检验
│       ├── generate_case_spectrograms.py        # 典型案例频谱图
│       └── generate_sentence_plots.py           # 句级箱线图
├── outputs/
│   └── 已打包_全部实验结果/          # 完整实验结果（10 类，中文命名）
│       ├── 01_退化估计器/            #   模型、指标、混淆矩阵、散点图
│       ├── 02_V1-Oracle模型/         #   V1 最佳 checkpoint
│       ├── 03_V2-initial模型/
│       ├── 04_V2-Refined-B模型/
│       ├── 05_系统级增强对比/        #   总体表、分组表、曲线图
│       ├── 06_条件有效性实验/        #   V1/V2 条件有效性
│       ├── 07_无条件残差对照/
│       ├── 08_显著性检验/
│       ├── 09_典型频谱图/
│       └── 10_论文文档/
└── README.md
```

---

## 环境要求

- Python 3.9+
- PyTorch 2.0+（CPU 版）
- 依赖包：`numpy scipy soundfile einops matplotlib scikit-learn pystoi`
- 可选（PESQ）：需要 `pesq` 包和 Microsoft Visual C++ Build Tools

### 安装

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy soundfile einops matplotlib scikit-learn pystoi
```

### 环境变量（Windows）
```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

---

## 复现步骤

### Step 1：准备数据集

下载 VCTK 数据集并重采样到 16kHz：

```powershell
# 通过 openxlab 下载
openxlab login
openxlab dataset get --dataset-repo OpenDataLab/VCTK --target-path data/vctk

# 解压
python -c "import zipfile; zipfile.ZipFile('data/vctk/.../VCTK-Corpus.zip').extractall('data/vctk_raw')"

# 重采样 48kHz -> 16kHz
python work2/scripts/resample_vctk.py --input-dir data/vctk_raw/VCTK-Corpus/wav48 --output-dir data/vctk_16k --target-rate 16000
```

### Step 2：构建数据清单

```powershell
python -m work2.data.build_manifest --clean-dir data/vctk_16k --output outputs/manifest_vctk.jsonl --seed 2026
```

### Step 3：缓存退化估计器特征

```powershell
python -m work2.scripts.cache_degradation_features_sharded --clean-manifest outputs/manifest_vctk.jsonl --output-dir outputs/estimator_cache_sharded --seed 2026
```

### Step 4：训练退化估计器

```powershell
python -m work2.train.train_degradation_estimator_fast --cache-dir outputs/estimator_cache_sharded --output-dir outputs/degradation_estimator_fast --epochs 50 --batch-size 256 --seed 2026
```

### Step 5：评估退化估计器

```powershell
python -m work2.evaluation.evaluate_degradation_estimator_fast --cache-dir outputs/estimator_cache_sharded --checkpoint outputs/degradation_estimator_fast/best_model.pt --output-dir outputs/degradation_estimator_fast/evaluation
```

### Step 6：缓存自适应残差训练数据

```powershell
python -m work2.scripts.cache_adaptive_training_sharded --clean-manifest outputs/manifest_vctk.jsonl --checkpoint checkpoints/model_trained_on_dns3.tar --output-dir outputs/adaptive_cache_sharded --shard-size 128 --seed 2026
```

### Step 7：训练条件残差模块（以 V1 为例）

```powershell
# 正式训练
python -m work2.train.train_adaptive_cached --cache-dir outputs/adaptive_cache_sharded --output-dir outputs/adaptive_residual_v1 --residual-version v1 --condition-source oracle --epochs 50 --mini-batch-size 32 --learning-rate 0.001 --seed 2026

# 中断恢复
python -m work2.train.train_adaptive_cached --cache-dir outputs/adaptive_cache_sharded --output-dir outputs/adaptive_residual_v1 --residual-version v1 --condition-source oracle --epochs 50 --mini-batch-size 32 --learning-rate 0.001 --seed 2026 --resume
```

### Step 8：系统级增强评测

```powershell
python -m work2.evaluation.evaluate_enhancement_models --cache-dir outputs/adaptive_cache_sharded --v1-checkpoint outputs/adaptive_residual_v1/best_model.pt --output-dir outputs/paper_results
```

### Step 9：条件有效性实验

```powershell
python -m work2.evaluation.evaluate_condition_effectiveness --cache-dir outputs/adaptive_cache_sharded --gtcrn-checkpoint checkpoints/model_trained_on_dns3.tar --estimator-checkpoint outputs/degradation_estimator_fast/best_model.pt --residual-checkpoint outputs/adaptive_residual_v1/best_model.pt --residual-version v1 --output-dir outputs/paper_results_condition --split test --max-shards 20 --cpu-threads 8
```

### Step 10：无条件残差对照

```powershell
python -m work2.evaluation.evaluate_unconditional_residual --cache-dir outputs/adaptive_cache_sharded --v1-checkpoint outputs/adaptive_residual_v1/best_model.pt --output-dir outputs/paper_results_unconditional --split test --max-shards 20
```

### Step 11：显著性检验

```powershell
python -m work2.evaluation.compute_statistical_tests --enhancement-csv outputs/paper_results/sentence_level_metrics.csv --output-dir outputs/paper_results_stats
```

### Step 12：生成典型频谱图

```powershell
python -m work2.evaluation.generate_case_spectrograms --cache-dir outputs/adaptive_cache_sharded --v1-checkpoint outputs/adaptive_residual_v1/best_model.pt --output-dir outputs/paper_results_cases --split test --cpu-threads 8
```

---

## 模型版本说明

| 版本 | 类名 | 结构特点 | 论文定位 |
|------|------|---------|---------|
| V1 | `AdaptiveResidualModule` | 单分支全局条件残差 | 最稳定版本，主模型结果 |
| V2-initial | `AdaptiveResidualModuleV2` | 三频带 expert + band gate | 早期探索，旧 checkpoint 兼容 |
| V2-Refined-A | `AdaptiveResidualModuleV2Formal` | expert + global gate + tanh 限幅 | 门控受控版 |

---

## 关键实验结果速览

### 退化估计器（全量测试集）

| 任务 | Accuracy/F1 | MAE (dB) |
|------|:----------:|:--------:|
| 噪声识别 | 0.9999 / 0.9999 | — |
| 带宽分类 | 0.9961 / 0.9949 | — |
| 位深分类 | 0.8693 / 0.8661 | — |
| SNR 回归 | — | 0.8968 |

### 系统级增强（V1 vs Base）

| 指标 | Base | V1 | 改善比例 | p-value |
|------|-----:|----:|--------:|--------:|
| LSD | 13.06 | 11.20 | 76.4% | <0.001 |
| HF-LSD | 13.88 | 11.44 | 78.9% | <0.001 |
| STOI | 0.735 | 0.733 | — | 持平 |

### 条件有效性（V1，四种条件对比）

Oracle ≈ Predicted ≈ Zero ≈ Shuffled，四种条件差异极小，条件利用有待增强。

### 无条件残差对照

V1-cond ≈ V1-uncond，当前增益主要来自残差模块结构本身。

---

## 引用

如使用本工作，请引用：
```
基于显式复合退化建模与条件化频带残差修复的语音增强方法
EDCR-Net: Explicit Degradation-Conditioned Residual Speech Enhancement
```

本仓库基于 GTCRN (ICASSP 2024) 开源实现：
https://github.com/Xiaobin-Rong/gtcrn
