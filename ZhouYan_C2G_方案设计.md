# ZhouYan C2G 方案设计（单点改进 RFC · 申请 Level 2 算力）

> 挑战：C2G 参数高尔夫 —— 极限约束下的语言模型训练（ch-20260717031359-b8wyg0）
> 提交人：2024102310451　｜　本 RFC 对应 Level 2 门槛（单点改进，≥1000 字）
> 上一阶段产出：`ZhouYan_C2G_方案草案.md`（Level 1 算力申请）

## 1. 本轮要回答的单一问题

**在 16MB artifact 硬约束下，把优化器从 AdamW 换成 Muon（Newton–Schulz 正交化更新），能否稳定地把 baseline BPB 从 1.2244 往 1.18 以下压？**

我把"单点"定义得很严：本轮**只改优化器**。tokenizer、架构、量化、训练循环全部冻结在官方 baseline 配置上。理由见方案草案：Muon 与 tokenizer / 架构正交，几乎不增加 artifact 体积，在参数预算被卡死的场景下边际收益最高。

## 2. 假设（H）与判据（R̂）

- **H1（主假设）**：Muon 的 Newton–Schulz 正交化更新在相同 step / token 预算下，收敛速度优于 AdamW，且**跨随机种子稳定**。
- **H2（稳定性假设）**：Muon 的最终 BPB 对不同 seed 的敏感度显著低于 AdamW。
- **R̂（预期结果，可证伪）**：在同一配置下，Muon 的 3-seed 平均 val_bpb 低于 AdamW 3-seed 平均；且 Muon 的 seed 间极差（max−min）小于 AdamW 的 1/5。若为假，则本方向在等预算下不成立，执行第 6 节止损。

## 3. 对照设计（对照组必须同预算）

| 组 | 优化器 | 其他 | 作用 |
|---|---|---|---|
| A（对照） | AdamW | baseline 配置 | 基准线 |
| B（处理） | Muon | 其余与 A 完全一致 | 验证 H1 / H2 |

- **控制变量**：dim/layers/heads/seq/batch/steps、数据切片、评测集、量化口径全部相同；**只有优化器不同**。
- **重复**：每组跑 3 个随机种子（1337 / 2024 / 7），报告均值与极差，而不是挑单个最好看的结果。
- **单变量原则**：Muon 内部 lr / momentum 微调只作为优化器自身的超参搜索，不引入第二处结构改动。

## 4. 指标体系（每次实验固定落盘）

主指标 `val_bpb`；辅助 `int8_val_bpb` 与 `quant_bpb_delta`（量化代价）、`artifact_bytes_zlib_int8`（≤16,000,000 B）、`wall_seconds` / `seconds_per_step`（10 分钟预算）、`tokens_seen` / `train_tail_loss`（区分"训练不足"与"方向无效"）、`anchor_bpb_uniform`（口径自检：应等于 `log2(vocab)×tokens_per_byte`）。

## 5. 已完成的 μ-scale 预实验（成本 $0，CPU 沙盘）

在本地 CPU 上以 3.74M 参数（dim 256 / 4 层 / 4 head / seq 256 / vocab 1024）跑通"训练→BPB 评测→int8+zlib 打包→16MB 检查"全链路，用 3 个种子对照 AdamW 与 Muon：

| 组 | seed | val_bpb | wall(s) | s/step |
|---|---|---|---|---|
| AdAmW | 1337 | 2.485302 | 426.4 | 1.066 |
| AdAmW | 2024 | 2.628941 | 477.1 | 1.193 |
| AdAmW | 7 | 2.502021 | 431.1 | 1.078 |
| Muon | 1337 | 2.488902 | 561.4 | 1.403 |
| Muon | 2024 | 2.489402 | 560.6 | 1.402 |
| Muon | 7 | 2.490237 | 560.5 | 1.401 |

**读数**：AdamW 3-seed 均值 2.5388、极差 0.1436；Muon 3-seed 均值 2.4895、极差仅 0.0005。**Muon 均值更低（−0.049 BPB），且 seed 稳定性提高约两个数量级**，H2 得到强支持，H1 在 μ-scale 方向性成立。
**代价（诚实披露）**：Muon 每 step 更贵（+18%–32% 墙钟），在 10 分钟硬预算下需换取更少 step，这一点必须在 H100 上用"等时间"而非"等 step"重新对齐（见风险 R3）。
**对照项 recur（depth_share=2，effective_depth=5）**：seed1337/7 得 2.7428/2.7472，明显差于 base——μ-scale 下 recurrence 属"欠训练"，说明其收益依赖满预算，故本轮不并入主线，仅留作 H100 消融项。

## 6. 失败模式与止损（对应方案草案第 4 节，逐层可判）

- **R1｜μ-scale 有增益、H100 无增益（尺度不迁移）**：判据为 H100 上 Muon 相对 baseline 的 ΔBPB ≥ 0 且超噪声（±0.005）。止损：把主线降级为已被 1.0810 验证的 **SP8192 + depth recurrence + AdamW**，Muon 转入消融章节。
- **R2｜优化器实现 bug 伪装成"方向无效"**：先核 Newton–Schulz 系数 `(3.4445, −4.7750, 2.0315)`、确认只作用于 2D 矩阵（embedding/head/scalar 仍走 AdamW）、确认 momentum warmup 生效；**单变量逐个排查，绝不一次改两处**。
- **R3｜Muon 每 step 更贵，10 分钟预算下 step 数不足**：以"等墙钟预算"重跑对照，若等时间下 Muon 不再有优势，则改用 Muon+更小模型或降低 Muon 迭代开销。
- **R4｜全方向失败**：提交诚实 baseline + 完整消融 + 失败归因，保底 Level 1 交付（BPB 无改善时，实验设计与复现性仍占 35% 权重）。**不伪造数字、不粘贴榜单。**

## 7. $100 Level 2 预算分配（单次 8×H100 约 9:30≈$3.8）

① 复现 baseline（确认 1.2244±0.005）→ ② ③ Muon 单点 2 次（lr/momentum 小网格）→ ④ 等时间对齐复跑 → ⑤ Muon + SP8192 组合 → ⑥ 预留救火。筛选全部在 CPU / 单卡完成，H100 只跑最终确认。

## 8. 成功标准（冻结）

Level 2 达成 = 在官方约束下，Muon 单点改动使 BPB < 1.18，且改动可被 `submission.json` 与日志完整复现。未达成则按第 6 节止损并如实上报。
