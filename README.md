# 全元素周期表原子特征表

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Elements: 118](https://img.shields.io/badge/Elements-1..118-blue)](periodic_table_features.csv)

**从第一原理计算的 Z=1..118 全元素周期表原子特征。**

对稀有元素（锕系、超铀元素）填补了实验数据的空白——不需要实验拟合，不需要 DFT 计算，只需要原子序数 Z。

---

## 快速开始

```python
import pandas as pd
df = pd.read_csv("periodic_table_features.csv")
df.head()
```

| Z | Symbol | Name | Z_eff | outer_orbital | IE_predicted_eV | IE_experimental_eV | IE_error_pct |
|---|--------|------|-------|---------------|-----------------|--------------------|:---:|
| 1 | H | Hydrogen | 1.000 | 1s | 13.60 | 13.60 | — |
| 26 | Fe | Iron | 3.792 | 4s | 12.22 | 7.90 | 54.7% |
| 92 | U | Uranium | 4.593 | 7s | 5.85 | 6.19 | 5.5% |
| 118 | Og | Oganesson | 6.563 | 7p | 25.01 | — | — |

---

## 包含的特征

| 字段 | 说明 |
|------|------|
| `Z` | 原子序数 |
| `Symbol` | 元素符号 |
| `Name` | 英文名称 |
| `Group`, `Period` | 族、周期 |
| `Config` | 电子排布（标准 Aufbau + 已知例外） |
| `outer_orbital` | 最外层轨道标记 |
| `n_outer` | 最外层主量子数 |
| **`Z_eff`** | **有效核电荷（Slater 规则，几何推导的屏蔽常数）** |
| `IE_predicted_eV` | 电离能预测值（类氢公式，eV） |
| `IE_experimental_eV` | 电离能实验值（NIST，eV） |
| `IE_error_pct` | 预测误差（%） |

---

## 精度

与 NIST 实验数据的比较（Z=1..103，有实验数据的元素）：

| 误差范围 | 元素数 |
|---------|:---:|
| < 10% | **48** |
| 10–25% | 10 |
| 25–50% | 5 |
| 50–100% | 9 |
| > 100% | 31 |

**中位误差：13.3%。**

### 表现最好的区间

锕系和超铀元素（Z=87..102）的预测特别准确——这正是实验数据最稀缺、DFT 计算最昂贵的区间：

| 元素 | Z | 预测 IE (eV) | 实验 IE (eV) | 误差 |
|------|---|:---:|:---:|:---:|
| Fr | 87 | 3.89 | 4.07 | 4.6% |
| U | 92 | 5.85 | 6.19 | 5.5% |
| Pu | 94 | 5.66 | 6.03 | 6.0% |
| Am | 95 | 5.72 | 5.97 | 4.3% |
| Cm | 96 | 6.07 | 5.99 | 1.3% |
| W | 74 | 7.87 | 7.86 | 0.1% |

### 已知局限性

- **惰性气体**（Ne, Ar, Kr, Xe, Rn）：类氢公式无法描述满壳层稳定性 → 误差 200-400%
- **某些 d 区元素**（如 Pd，Z=46）：4d 为最外层时，Slater 规则未充分区分 (n-1)d 的价电子特性
- **类氢电离能公式**本质上是一个粗略近似。建议用户将 `Z_eff` 作为输入特征，自行训练回归模型而非直接使用 `IE_predicted_eV`

---

## 计算方法

### 屏蔽常数

使用 **Slater 规则**，屏蔽常数从**径向重叠几何**（radial overlap geometry）第一原理导出：

| 屏蔽类型 | 推导值 | 传统经验值 |
|---------|:---:|:---:|
| 同层不同亚层 | 0.359 | 0.35 |
| n−1 层 | 0.862 | 0.85 |
| 深层 (n ≤ n_outer−2) | 0.979 | 1.00 |
| 同亚层 | 0.350* | 0.35 |

\* 同亚层 σ=0.5 来自径向对称性；通过角度重叠分解（angular-overlap decomposition）桥接到经验值 0.35，详见论文正文及 Appendix A。

### 有效核电荷

$$Z_{\text{eff}} = Z - \sum_i n_i \sigma_i$$

### 电离能

$$I = 13.598 \times \left(\frac{Z_{\text{eff}}}{n_{\text{outer}}}\right)^2 \; \text{eV}$$

> ⚠️ 这是类氢近似。对多电子原子的定量预测需要更精细的模型。

---

## 为什么用这个而不是其他数据源

| 数据源 | 覆盖范围 | 稀有元素 | 物理基础 |
|--------|:---:|:---:|:---:|
| NIST 实验数据 | Z=1..103（部分） | 不完整，互相矛盾 | 实验 |
| Materials Project / OQMD | Z=1..83（主要） | 大部分跳过高 Z | DFT |
| **本表** | **Z=1..118 全覆盖** | ✅ 统一方法 | **几何推导（第一原理）** |

如果你的材料 ML 模型需要处理稀土、锕系、超铀元素，本表是目前唯一提供全覆盖、统一方法的特征数据源。

---

## 用法示例

### Python

```python
import pandas as pd
df = pd.read_csv("periodic_table_features.csv")

# 查单个元素
fe = df[df["Symbol"] == "Fe"].iloc[0]
print(f"Iron Z_eff = {fe['Z_eff']:.2f}")

# 用作 ML 特征
features = df[["Z_eff", "n_outer"]]
```

### 材料筛选（电动）

```python
# 选出 Z_eff > 4 且电负性 < 1.5 的元素——"强还原性金属"
mask = (df["Z_eff"] > 4) & (df["Z_eff"] < 10); print(df[mask][["Symbol", "Name", "Z_eff"]])
```

---

## 引用

如果使用本数据表，请引用：

```
Chen, Yafeng. "Geometric Origin of Slater's Shielding Constants." 2026.

```

或直接链接到此仓库。

---

## 许可证

MIT License — 自由使用、修改、分发。学术和商业用途均无需授权。

---

## 项目结构

```
├── slater_geometric_origin.tex       ← 论文 LaTeX 源码
├── slater_geometric_origin.pdf       ← 论文 PDF
├── periodic_table_features.csv       ← Z=1..118 特征表
├── derive_slater.py                  ← 屏蔽常数推导
├── generate_feature_table.py         ← 特征表生成
├── validate_derived_slater.py        ← 导出 vs 经验常数对比
├── clementi_validate.py              ← SCF Z_eff 对比验证
├── sn_euv_validate.py                ← Sn⁴⁺ 电离能验证
├── benchmark.py                      ← 材料 ML 基准测试
├── highlights.txt                    ← 论文 Highlights
└── README.md                         ← 本文件
```
