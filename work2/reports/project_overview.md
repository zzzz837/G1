# GTCRN 自适应残差增强系统 — 项目整体说明

**仓库**: `https://github.com/Xiaobin-Rong/gtcrn.git`  
**分支**: `work2-adaptive-residual`  
**当前提交**: `561d191bf6dab0c738c088a8d40b48f80c7f1276`  
**总测试**: 73/73 passed  
**基线保持**: bit-exact (误差 0.0)

---

## 1. 原始 GTCRN 是什么

**GTCRN** (Grouped Temporal Convolutional Recurrent Network) 是 ICASSP 2024 发表的超低计算量语音增强模型。

| 指标 | 值 |
|------|----|
| 参数量 | 48.2K |
| 计算量 | 33.0 MMACs/s |
| 输入 | 16kHz 单声道带噪语音 |
| 输出 | 增强后语音 |
| 处理域 | STFT 频谱域 (512-FFT, 256-hop) |

### 原始架构

```
带噪语音 → STFT(257频点) 
  → ERB频带压缩(257→129)
  → SFE子带特征展开(3→9通道)
  → Encoder(2×ConvBlock + 3×GTConvBlock, 逐步下采样4倍)  → (B,16,T,33)
  → DPGRNN×2 (分组双向+单向GRU, 时间/频率双路径建模)
  → Decoder(对称+跳跃连接, 逐步上采样)  → (B,2,T,257)
  → Complex Ratio Mask → iSTFT → 增强语音
```

**关键组件**:
- **ERB**: 等效矩形带宽滤波器组, 低频直接保留, 高频压缩
- **GTConvBlock**: ShuffleNetV2 风格的组卷积 + 通道混洗
- **TRA**: 时间循环注意力 (GRU 计算帧能量作为软注意力)
- **DPGRNN**: 分组双路径 RNN, 分别在频率轴(双向)和时间轴(单向)建模

**原始应用场景**: 通用语音增强 — 输入任意带噪语音, 输出增强结果。不区分噪声类型和退化程度。

---

## 2. 我们的创新点

### 核心问题

原始 GTCRN 对**所有输入无差别处理**。但在实际场景中，语音可能遭受不同组合的退化:

| 退化类型 | 示例 |
|----------|------|
| 带宽限制 | 电话语音(8kHz)、网络通话(窄带) |
| 量化失真 | 低比特率编码 |
| 加性噪声 | 交通、风声、多人交谈 |
| 复合退化 | 同时存在以上多种 |

**GTCRN 对所有情况使用同一套参数，无法针对性优化。**

### 创新方案: 退化感知自适应残差

在冻结的 GTCRN 之后，增加一个**极轻量**的退化估计 → 自适应补偿链路:

```
原始方案 (GTCRN 原文):
  带噪音频 → GTCRN → 增强音频

我们的方案 (GTCRN Adaptive):
  带噪音频 → 冻结GTCRN → 基础增强 ─┬→ 最终输出
                    ↓              │
              dpgrnn2 bottleneck    │
                    ↓              │
              DegradationEstimator │
              (1,985 params)       │
                    ↓              │
         噪声? SNR? 带宽? 量化?     │
               condition vector    │
                    ↓              │
          AdaptiveResidualModule ←─┘
          (2,184 params)
                    ↓
              enhanced_final = base + α · residual
```

### 三个核心创新模块

#### (A) 退化估计器 (DegradationEstimator)
- **输入**: GTCRN bottleneck 的统计特征 (mean + std, 共32维)
- **输出**: 4个并行预测
  - 是否存在加性噪声 (二分类)
  - SNR 值回归 (0-40 dB)
  - 带宽类别 (全频/6k/4k, 三分类)
  - 量化位数 (16/12/10/8 bit, 四分类)
- **参数量**: 1,985
- **关键设计**: 
  - SNR 损失只在有噪声样本上计算 (masked loss)
  - 不接入干净语音，只看 GTCRN 瓶颈特征
  - 可独立训练

#### (B) 自适应残差模块 (AdaptiveResidualModule)
- **输入**: 
  - GTCRN 基础增强频谱 (B, F, T, 2)
  - 退化条件向量 (B, 9) = [noise_prob, snr_pred, bw_probs, bit_probs]
- **处理**: 
  - 条件嵌入: FC→LayerNorm→PReLU, 9→32维
  - 频域卷积: Conv1d 在频率轴上提取残差
  - 条件门控: 学习 α 系数控制残差注入量
- **输出**: enhanced_final = enhanced_base + α · residual
- **参数量**: 2,184
- **关键设计**:
  - 在频谱域操作，与 GTCRN 输出直接融合
  - 不同退化条件产生不同的 α 门控值
  - 带宽受限场景可对高频区域做更大补偿
  - V2 版本支持分频带独立处理 (低频/中频/高频)

#### (C) 残差联合损失 (AdaptiveResidualLoss)
```
L = L_mag + λ_hf·L_hf + λ_res·L_res

L_mag:  全频带幅度 L1 + MSE
L_hf:   高频区域 (>55% Nyquist) 额外约束 (仅带宽受限样本)
L_res:  残差 L2 正则化, 鼓励小修正
```

---

## 3. 完整工作流

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练流程                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: 数据准备                                                    │
│    干净语音 + 噪声                                                   │
│      ↓ build_manifest.py (JSONL 清单, 80/10/10 split)                │
│                                                                     │
│  Step 2: 复合退化生成                                                │
│    clean → degradation.py → {clean, light, medium, heavy}           │
│    5种退化: 重采样/低通/量化/加噪/组合                                 │
│    自动生成标签: bandwidth_class, bit_class, noise_present, SNR      │
│      ↓                                                              │
│                                                                     │
│  Step 3: 特征缓存                                                    │
│    degraded audio → STFT → FrozenGTCRN → stats + labels             │
│      ↓ cache_degradation_features.py                                │
│    缓存内容: stats(32维) + 标签, 不含音频                             │
│    按源文件划分 train/valid/test, 防止数据泄漏                       │
│                                                                     │
│  Step 4: 训练退化估计器                                              │
│    stats(32) → DegradationEstimator → 4 predictions                 │
│    Loss: BCE + MaskedSmoothL1 + 2×CrossEntropy                     │
│      ↓ train_degradation_estimator.py                               │
│                                                                     │
│  Step 5: 训练自适应残差 (可选冻结估计器)                               │
│    degraded → STFT → FrozenGTCRN + Estimator + ResidualModule       │
│    GTCRN 始终冻结, 估计器可选冻结                                      │
│      ↓ train_adaptive.py                                            │
│                                                                     │
│  Step 6: 评测                                                        │
│    对比 GTCRN base vs GTCRN adaptive:                               │
│    PESQ, STOI, SI-SDR, SNR improvement                              │
│    按 severity 分组分析                                               │
│      ↓ evaluate_degradation_estimator.py                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       推理流程                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  带噪语音 (任意退化组合)                                              │
│    ↓ STFT (257频点)                                                 │
│    ↓ Frozen GTCRN → enhanced_base + bottleneck stats                │
│    ↓ DegradationEstimator → 4 predictions → condition(9维)           │
│    ↓ AdaptiveResidualModule → residual correction                   │
│    ↓ enhanced_final = base + residual                               │
│    ↓ iSTFT → 最终增强语音                                            │
│                                                                     │
│  额外计算量: ~4,200 FLOPs (估算, 两个微型网络的前向计算)                │
│  额外参数: 4,169 (仅占 GTCRN 的 8.6%)                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 修改对比

### 官方文件: 完全未修改

| 文件 | 状态 |
|------|------|
| `gtcrn.py` | 未修改 — SHA256 保持一致 |
| `loss.py` | 未修改 |
| `infer.py` | 未修改 |
| `checkpoints/model_trained_on_dns3.tar` | 原始权重 |

### 新增文件: work2/ 目录下全部

```
work2/
├── data/
│   ├── audio_utils.py          # 音频工具 (mono, normalize, RMS, SNR, validate)
│   ├── random_utils.py         # 可复现随机种子
│   ├── stft_utils.py           # PyTorch 2.x 兼容 STFT/iSTFT (无 monkey-patch)
│   ├── degradation.py          # 5种退化 + 4级预设 + dataclass 标签
│   ├── label_utils.py          # SNR 归一化/反归一化 + 标签转换
│   └── build_manifest.py       # 数据清单构建器 (JSONL)
├── models/
│   ├── frozen_gtcrn_extractor.py  # 冻结 GTCRN + hook 捕获 dpgrnn2
│   ├── degradation_estimator.py   # 退化估计器 (1,985 params)
│   ├── adaptive_residual.py       # 自适应残差补偿模块 (2,184 params)
│   └── gtcrn_adaptive.py          # 端到端完整管道
├── losses/
│   ├── degradation_estimator_loss.py  # 多任务损失 (masked SNR)
│   └── residual_loss.py               # 高频+残差联合损失
├── datasets/
│   └── degradation_feature_dataset.py  # 缓存特征 PyTorch Dataset
├── train/
│   ├── train_degradation_estimator.py  # 估计器训练
│   └── train_adaptive.py               # 残差模块训练
├── evaluation/
│   └── evaluate_degradation_estimator.py  # 评测 + 混淆矩阵 + 散点图
├── scripts/
│   ├── run_baseline.py              # 基线推理 (refactored)
│   ├── run_infer.py                 # 官方 infer.py 兼容包装
│   ├── generate_degradation_demo.py  # 退化示例生成
│   ├── cache_degradation_features.py # 批量特征缓存
│   ├── run_estimator_smoke_test.py   # 估计器冒烟
│   └── run_adaptive_smoke.py         # 自适应残差冒烟
├── tests/
│   ├── test_baseline_output.py      # 基线输出一致性
│   ├── test_degradation.py          # 退化函数 (39 tests)
│   ├── test_degradation_estimator.py
│   ├── test_estimator_loss.py       # masked SNR loss
│   ├── test_feature_cache.py        # 缓存可复现 + 微型过拟合
│   ├── test_frozen_gtcrn_extractor.py
│   └── test_adaptive_residual.py    # 残差模块 (17 tests)
└── reports/
    ├── baseline_report.md
    ├── degradation_report.md
    └── degradation_estimator_report.md
```

---

## 5. 应用场景变化

| 维度 | 原始 GTCRN | GTCRN Adaptive (我们的方案) |
|------|-----------|---------------------------|
| **输入假设** | 任意带噪语音 | 可能遭受多种退化的语音 |
| **处理策略** | 统一模型参数 | 退化感知 + 自适应补偿 |
| **场景覆盖** | 仅加性噪声 | 噪声 + 带宽限制 + 量化失真 |
| **输出** | 增强语音 | 增强语音 + **退化诊断标签** |
| **额外参数** | 0 | +4,169 (8.6%) |
| **推理代价** | 33 MMACs | ~33 + 极微小开销 |
| **目标场景** | 通用语音增强 | **通信系统**: VoIP/电话/对讲机等带宽受限场景 |
| | | **低比特率编码**: 窄带+量化联合退化 |
| | | **录音取证**: 多退化源识别+增强 |

### 典型应用示例

```
场景 1: 网络电话 (VoIP)
  原始语音 → 8kHz重采样 → 低通4kHz → 10-bit量化 → 背景噪声
  GTCRN Adaptive 诊断: [带宽=窄带, 量化=10bit, 噪声=有, SNR≈15dB]
           针对性补偿: 中高频残差增强 + 量化噪声抑制

场景 2: 对讲机通信
  原始语音 → 窄带滤波 → 强噪声环境 → 低码率编码
  诊断: [带宽=4k, 量化=8bit, 噪声=有, SNR≈5dB]
  补偿: 频带重建 + 严重量化修复

场景 3: 高清录音 (无退化)
  原始语音 → 全频带, 16bit, 无噪声
  诊断: [带宽=全频, 量化=16bit, 噪声=无]
  补偿: α≈0, 几乎不修改 GTCRN 原始输出
```

---

## 6. 关键设计决策

| 决策 | 理由 |
|------|------|
| GTCRN 全部冻结 | 保持原始增强能力, 不破坏预训练权重 |
| 特征来自 dpgrnn2 而非 dpgrnn1 | dpgrnn2 是最终瓶颈, 语义信息最丰富 |
| 使用 mean+std 而非完整 bottleneck | 大幅降低缓存大小 (32维 vs 16×33×T) |
| 退化估计和残差补偿分离 | 可独立训练/评估/替换 |
| 残差在频谱域操作 | 与 GTCRN 输出直接融合, 无需额外 iSTFT |
| 条件向量使用概率而非 logits | softmax/sigmoid 后值域统一 [0,1], 训练更稳定 |
| SNR 归一化到 [0,1] | 与 sigmoid 输出匹配, 避免尺度问题 |
| 同一源文件不跨 split | 哈希分配确保无数据泄漏 |

---

## 7. 测试覆盖

| 测试组 | 数量 | 覆盖内容 |
|--------|------|----------|
| 退化函数 | 39 | 量化/SNR/重采样/低通/可复现/标签/异常输入 |
| 退化估计器 | 4 | 输出形状/参数量/反向传播/SNR范围 |
| 估计器损失 | 2 | 正常batch/无有效SNR batch (不NaN) |
| 特征提取器 | 8 | 冻结/形状/NaN/维度/hook不重复/基线一致性 |
| 特征缓存 | 3 | 可复现/无跨split泄漏/微型过拟合 |
| 自适应残差 | 17 | 形状/NaN/参数量/残差应用/训练模式/V2/损失/条件向量 |
| **合计** | **73** | |

---

## 8. 尚未完成

| 项目 | 状态 |
|------|------|
| 真实数据集 | 待获取 (DNS3 或 VCTK) |
| 大规模特征缓存 | 待数据到位后运行 |
| 退化估计器正式训练 | 代码就绪, 待缓存数据 |
| 自适应残差正式训练 | 代码就绪, 待估计器训练完成 |
| 最终评测 (PESQ/STOI/SI-SDR) | 待模型训练完成 |
| 消融实验 | 待定 |
| LADSPA 实时插件适配 | 待定 |
