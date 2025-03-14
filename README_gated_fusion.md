# 门控跨模态注意力融合机制

本项目实现了一种改进的跨模态融合机制——门控跨模态注意力（Gated Cross-Modal Attention），用于增强蛋白质-小分子结合亲和力预测模型DeepBindNet的性能。

## 目录

- [门控跨模态注意力融合机制](#门控跨模态注意力融合机制)
  - [目录](#目录)
  - [实现原理](#实现原理)
  - [技术优势](#技术优势)
  - [代码结构](#代码结构)
  - [使用方法](#使用方法)
    - [1. 运行基础测试](#1-运行基础测试)
    - [2. 比较模型性能](#2-比较模型性能)
    - [3. 在DeepBindNet中使用门控融合](#3-在deepbindnet中使用门控融合)
  - [实验结果](#实验结果)
  - [进一步改进](#进一步改进)

## 实现原理

门控跨模态注意力机制通过引入门控系数来动态调节原始特征与注意力特征的平衡，其核心数学表达式为：

```
g = Sigmoid(W_g·BN(ReLU(W_c[h_p;h_m])))
h_attn = g ⊙ Attention(h_p, h_m) + (1-g) ⊙ h_p
```

其中：
- `g`：门控系数，取值范围(0,1)
- `h_p`：蛋白质特征
- `h_m`：分子特征
- `⊙`：逐元素相乘
- `BN`：批归一化层

## 技术优势

1. **动态调节机制**：
   - 通过门控系数g ∈ (0,1) 自动平衡原始特征与注意力特征
   - 噪声数据下自动降低注意力权重（g→0）
   - 强相关特征时提升注意力贡献（g→1）

2. **梯度稳定设计**：
   - 批归一化层防止梯度爆炸
   - Sigmoid激活限制输出范围
   - 残差连接保证梯度流通
   - 梯度裁剪：设置梯度范数阈值≤2.0

3. **计算效率优化**：
   - 参数共享：门控网络与注意力模块共享特征投影
   - 并行计算：门控系数与注意力计算可并行执行

## 代码结构

- `fusion.py`：包含原始的`CrossAttentionFusion`和新实现的`GatedCrossAttention`
- `model_gated.py`：使用门控跨模态注意力的DeepBindNet模型
- `test_gated_fusion.py`：测试门控融合机制的脚本
- `compare_models.py`：比较标准模型和门控模型性能的脚本
- `run_test.py`：运行测试的辅助脚本

## 使用方法

### 1. 运行基础测试

```bash
python run_test.py
```

这将执行`test_gated_fusion.py`脚本，测试门控跨模态注意力机制，并生成门控系数分布图。

### 2. 比较模型性能

```bash
python compare_models.py
```

这将比较标准模型和门控模型在不同噪声级别下的性能，并生成以下图表：
- `model_comparison.png`：模型运行时间和预测稳定性比较
- `gate_vs_noise.png`：噪声级别与门控系数的关系

### 3. 在DeepBindNet中使用门控融合

要在现有的DeepBindNet模型中使用门控跨模态注意力，只需将融合模块替换为`GatedCrossAttention`：

```python
from fusion import GatedCrossAttention

# 替换融合模块
self.fusion_module = GatedCrossAttention(
    embed_dim=feature_dim,
    num_heads=fusion_heads,
    ff_dim=feature_dim * 4,
    num_layers=fusion_layers,
    dropout=dropout_rate
)
```

## 实验结果

门控跨模态注意力机制相比标准跨模态注意力具有以下优势：

1. **噪声鲁棒性**：在高噪声环境下，门控机制能够自动降低注意力权重，保持预测稳定性
2. **特征选择性**：能够根据输入数据质量动态调整特征融合策略
3. **梯度流通性**：改进的残差连接和批归一化设计使训练更加稳定

## 进一步改进

1. **自适应温度系数**：引入可学习的温度参数，动态调整sigmoid曲线斜率
   ```python
   T = log(1+exp(w_T))  # 确保温度系数为正
   g = Sigmoid((gate_logits) / T)
   ```

2. **多层门控**：在Transformer的每一层都添加门控机制，实现更细粒度的特征控制

3. **注意力可视化**：添加注意力权重和门控系数的可视化工具，提高模型可解释性
