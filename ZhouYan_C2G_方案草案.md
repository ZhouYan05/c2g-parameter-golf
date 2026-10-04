# ZhouYan C2G 方案草案（Level 1 算力申请）

> 挑战：C2G 参数高尔夫 —— 极限约束下的语言模型训练（ch-20260717031359-b8wyg0）
> 提交人：2024102310451　｜　目标等级：Level 2（争取 Level 3）

## 0. 先说清楚我的处境

本机**没有可用 GPU**：RTX 4060 Laptop（8G）上装的是 CPU-only 的 torch 2.14.1+cpu，`CUDA_AVAIL=False`。因此我的方案分两段执行，而不是凭空写一份"打算怎么训"的纸面计划：

- **第 1 段（已在本地执行，成本 $0）**：在 CPU 上搭一个 **μ-scale 参数高尔夫沙盘**（3.74M 参数：dim 256 / 4 层 / 4 head / seq 256 / vocab 1024），把训→评→量化打包→16MB 检查的**全链路打通**，让每条技术路线的判据先出一个可复现的方向性结论。
- **第 2 段（申请算力后）**：把第 1 段胜出的方向搬到 8×H100，跑官方 10 分钟 / 16MB 约束下的正式提交。

这样做的依据是 CHALLENGE.md 里的省钱建议原话：**"先在单张 A100 上用 nanoGPT 小模型验证思路，最后只跑最后的 3 次提交"**。

## 1. 门槛问题一：把 baseline 从 1.2244 往哪个方向压？

**答案：优化器方向 —— Muon（Newton–Schulz 正交化更新），单点切入。**

理由：
1. Naive baseline（9 层 / 512 维 / tied embedding / 4 KV head）用的是 AdamW；**优化器是与 tokenizer、架构都正交的一层**，可以叠加到任何胜出的架构上，收益不会被别的改动抵消。
2. **16MB 是硬约束，而 Muon 几乎不增加 artifact 体积**（它只是换更新规则，不新增参数），在"参数预算被卡死"的场景下，优化器效率的边际收益最高。
3. 2026-04 榜首的 1.0810 由 SP8192 + 3-Layer Recurrence 取得，说明这两条路已被大量队伍占据；我先做正交化更新，避免在拥挤方向上做第 N 次重复。

**为什么不是"都试试"**：我明确只把 Muon 当作本轮假设。SP8192 tokenizer 作为**已知增益**在第 2 阶段直接采纳（不当作假设去验证），depth recurrence 放在方案设计的消融矩阵里作为**对照项**，而不是并列假设。

## 2. 门槛问题二：你怎么知道这个方向有效？

**证据 1（论文/权威来源）**：Keller Jordan 的 Muon 博客 <https://kellerjordan.github.io/posts/muon/> 及其代码 modded-nanoGPT <https://github.com/KellerJordan/modded-nanogpt>。我读到的关键内容：(a) Muon 用 Newton–Schulz 迭代把 2D 参数矩阵的动量更新**正交化**，替代 AdamW 的逐元素二阶矩；(b) 在该 speedrun 里把 Karpathy 的 GPT-2 复现从 **45 分钟压到 3 分钟，相比 AdamW 训练速度提升 35%**；(c) Muon **只作用于 2D 矩阵**，embedding / head / scalar 仍走 AdamW——这一点直接决定了我的实现边界。

**证据 2（赛事定位）**：CHALLENGE.md「前例一」明确写着 Parameter Golf 就是 Keller Jordan 的 nanoGPT speedrun 的**制度化版本**，即该 speedrun 上的优化经验对本赛事有直接迁移性。

**证据 3（本地可复现的可行性判据）**：我已在 CPU 沙盘跑通全链路（tokenize → train → val BPB → int8+zlib 打包 → 16MB 检查）。其中 **uniform anchor 校验**通过：`log2(1024) × tokens_per_byte = 10 × 0.407562 = 4.075622`，与实测 anchor 完全一致——说明我的 BPB 口径与官方定义一致，沙盘上的方向性结论可以往外推。

## 3. 门槛问题三：跑多少次实验？每次看什么指标？$25 怎么花？

**每次实验固定记录**（这些字段全部落进 `logs/` 与 `submission.json`）：
| 指标 | 作用 |
|---|---|
| `val_bpb` | 主指标（越低越好） |
| `int8_val_bpb` 与 `quant_bpb_delta` | 量化代价，16MB 打包后是否掉分 |
| `artifact_bytes` | 是否 ≤ 16,000,000 B |
| `wall_seconds` / `seconds_per_step` | 10 分钟预算是否踩边 |
| `tokens_seen` / `train_tail_loss` | 判断是"训练不足"还是"方向无效" |
| `anchor_bpb_uniform` | 口径自检（与 log2(vocab)×tokens_per_byte 对齐） |

**预算测算**：单次 8×H100 完整训练约 9:30 ≈ 0.16 h × $24/h ≈ **$3.8**；**$25 ≈ 6 次完整跑**。分配如下：

| 次序 | 用途 | 目的 |
|---|---|---|
| ① | 复现官方 baseline | 确认 1.2244（±0.005），排除环境噪声 |
| ②③ | Muon 单点（2 次，小网格 lr/momentum） | 验证方向，取最好一档 |
| ④ | Muon + depth recurrence | 正交叠加是否增益 |
| ⑤ | Muon + SP8192 tokenizer | 组合效应 |
| ⑥ | 预留 | 复跑 / 救火 / 关键对照 |

**先便宜后贵**：所有筛选（含 lr 网格、实现 bug 排除）都在小尺寸 + 单卡完成，H100 只跑最后确认。

## 4. 门槛问题四：失败怎么办？

分层止损，每一层都有明确判据：

- **层 1｜μ-scale 就变差**：先**怀疑实现**而不是方向——检查 Newton–Schulz 系数 `(3.4445, -4.7750, 2.0315)`、是否只作用于 2D 矩阵、momentum warmup 是否生效；**单变量逐个改，绝不一次改两处**。排除 bug 后仍变差，才判该方向在本地尺度无效。
- **层 2｜CPU 有效但 H100 无效（尺度依赖）**：说明收益来自小尺度过拟合。立即降级到已被 1.0810 验证过的路线：**SP8192 tokenizer + depth recurrence + AdamW**，Muon 从主线转入消融章节。
- **层 3｜所有单点都无效**：提交**诚实的基线 + 完整消融 + 失败归因**，保底 Level 1 交付（BPB 数字即使没有改善，实验设计与复现性仍占 35% 权重）。**不伪造数字、不粘贴榜单**。

## 5. 预期结果（自我校准）

- **R̂（预期）**：第 1 段给出方法判据（不宣称榜单数字——硬件与尺度都不同）；第 2 段目标 Level 2（BPB < 1.18），若能顺利叠加 SP8192 则冲 < 1.15（Elite 20 门槛）。
- **明确不承诺**：本方案**不承诺**接近 1.0810。没有 8×H100 与 SP8192 训练预算之前，任何更低的数字都是空话。
