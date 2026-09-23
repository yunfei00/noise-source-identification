# 阶段二：离线复合干扰数据集

分支：`feature/offline-composite-dataset-v2`

阶段一（单源采集量与 S2/S3 质量验证）已经冻结并合入 `main`。本阶段只处理 AB / AC / BC / ABC 离线复合数据，不回改阶段一实验结论。

## 1. 当前状态

- 当前 800M 推荐单源采集基线：500 个/源；1000 个/源作为保守配置。
- 单源必须先固定 train / val / test，再在各 split 内部生成复合样本。
- 禁止同一个单源原始文件跨 train / val / test 参与复合，避免数据泄漏。
- 复合数据训练前离线生成，固定 seed，并写 manifest。
- 每个复合样本必须记录来源单源文件、split、混合参数和生成版本。

## 2. 第一项任务：先确认物理量，不修改混合公式

仓库当前配置将输入描述为 `db_trace`，现有旧合成逻辑包含逐源归一化、增益、时移/极性变化、直接相加和最终归一化。该旧逻辑暂不作为最终物理方案。

在确认 CSV 第二列到底代表什么之前，不进行大规模 AB/AC/BC/ABC 生成。需要确认的最小信息：

1. CSV 第二列的单位/物理量：例如 dBm、dBµV、接收机 detector level，或其他。
2. 每个点代表频谱功率/功率密度，还是时域幅度。
3. 三个单源是否在相同 RBW/VBW、衰减、参考电平、检波方式、扫频范围和点数下采集。
4. 单源文件是否都包含同一套背景噪声/系统底噪。
5. 真实多源同时开启时，仪表显示量在物理上应如何叠加。

只有上述信息确认后，再选择：
- 功率类 dB 数据：通常先转线性功率域，处理背景噪声后叠加，再转回 dB；
- 时域幅度：按幅度/相位模型处理；
- 其他 detector 输出：按仪表测量定义确定。

## 3. 公司电脑切换到阶段二分支

```powershell
cd D:\code\noise-source-identification
git fetch origin
git switch feature/offline-composite-dataset-v2
git pull --ff-only origin feature/offline-composite-dataset-v2
git log -3 --oneline
```

完成后先不要运行旧的复合数据生成脚本。

## 4. 阶段二实施顺序

1. 确认 CSV 物理含义与采集链路。
2. 冻结物理混合公式和背景噪声处理规则。
3. 重构离线 mixer，并增加单元/数值检查。
4. 先固定单源 split。
5. 分别在 train / val / test 内生成 AB / AC / BC / ABC。
6. 写 provenance manifest，并自动检查零跨 split 泄漏。
7. 小规模生成并人工/统计验证。
8. 再生成正式复合数据集并训练。
9. 与真实混合数据进行独立验证。

## 5. 当前停止点

当前只执行第 3 节切换分支。下一步先确认物理量；在此之前不修改最终混合公式、不生成正式复合训练集。
