# C2G 参数高尔夫 —— ZhouYan 提交仓库

挑战：**C2G 参数高尔夫 —— 极限约束下的语言模型训练**（`ch-20260717031359-b8wyg0`）
提交人：`2024102310451`　｜　目标等级：**Level 2**（方案阶段，争取 Level 3）

本仓库是我对 C2G 的**单点改进方案（Muon 优化器）**及其**本地 μ-scale 可复现实验沙盘**。

## 交付物

| 文件 | 说明 |
|---|---|
| [`ZhouYan_C2G_方案草案.md`](./ZhouYan_C2G_方案草案.md) | Level 1 算力申请（≥500 字，回答门槛 4 问） |
| [`ZhouYan_C2G_方案设计.md`](./ZhouYan_C2G_方案设计.md) | Level 2 单点改进 RFC（≥1000 字，假设/对照/风险） |
| [`ZhouYan_C2G_AI日志.md`](./ZhouYan_C2G_AI日志.md) | AI 使用日志（分工、迭代、纠错、证据） |
| [`ZhouYan_C2G_AAR.md`](./ZhouYan_C2G_AAR.md) | 复盘（预期 vs 实际、卡点、下一步） |

## 代码（μ-scale 参数高尔夫沙盘）

`harness/` 下为 CPU 可跑的全链路沙盘（3.74M 参数：dim 256 / 4 层 / 4 head / seq 256 / vocab 1024）：

- `c2g_data.py`　数据/tokenizer（Windows 非 ASCII 路径自动转 ASCII 临时路径）
- `c2g_model.py`　μ-scale Transformer
- `c2g_optim.py`　AdamW / **Muon（Newton–Schulz 正交化）**
- `c2g_quant.py`　int8 + zlib 打包与 16MB 检查
- `c2g_train.py`　训练 + BPB 评测主程序
- `run_matrix.py`　多配置 × 多种子矩阵执行

## 复现

```bash
# 单次训练（默认参数见 c2g_train.py）
python harness/c2g_train.py --opt muon --seed 1337

# 运行矩阵（base / muon / recur / tie × seeds）
python harness/run_matrix.py
```

## 关键结果（本机 CPU，仅供参考的方向性判据，非榜单成绩）

| 组 | 3-seed val_bpb 均值 | 极差 |
|---|---|---|
| AdamW（base） | 2.5388 | 0.1436 |
| **Muon** | **2.4895** | **0.0005** |

> 结论：在本机 μ-scale 下，**Muon 均值更低（−0.049 BPB）且显著更 seed 稳定**；代价是每 step 更贵（+18%–32% 墙钟），需在 H100 上以"等时间"重新对齐。所有 int8+zlib artifact 均 < 16MB。
