# 单源数据量消融实验操作手册

分支：`feature/offline-synthetic-dataset`

目标：固定验证集和测试集，只改变每个干扰源训练样本数，比较 100 / 200 / 500 / 1000 / 1500（当前实际最多 1400）的模型效果。

## 1. 更新代码

```powershell
cd D:\code\noise-source-identification
git switch feature/offline-synthetic-dataset
git pull --ff-only origin feature/offline-synthetic-dataset
git log -3 --oneline
```

关键提交应包含 `768e07e`、`adb837f`、`5f39f25`。

## 2. 已生成数据划分时不要重复生成

如果 `outputs\reports\ablation_800M` 已包含 100/200/500/1000/1500 五个 split CSV 和 `ablation_manifest.json`，直接进入第3步。

只有文件不存在时才运行：

```powershell
uv run python -m src.create_sample_count_ablation `
  --raw-root "这里替换成800M三个source目录的共同上级目录" `
  --output-dir "outputs\reports\ablation_800M" `
  --counts "100,200,500,1000,1500"
```

当前已确认：每源总数2000，训练池1400，验证300，测试300。原始CSV不移动、不复制、不修改。

## 3. 首次 Dry Run

```powershell
uv run python -m src.run_sample_count_ablation `
  --split-dir "outputs\reports\ablation_800M" `
  --counts "100,200,500,1000,1500" `
  --output-dir "outputs\ablation_runs\800M" `
  --dry-run
```

正常生成 `train_n100.yaml`、`train_n200.yaml`、`train_n500.yaml`、`train_n1000.yaml`、`train_n1500.yaml` 和 `run_manifest.json`。

## 4. 第一次只验证 N=100

```powershell
uv run python -m src.train `
  --config "outputs\ablation_runs\800M\train_n100.yaml"
```

启动后重点检查：`training_data.mode=real_only`、`train_samples=300`、`val_samples=900`、`test_samples=900`。N=100即三个source各100，共300训练样本。

如果训练集变成约4200、程序重新生成split，立即 Ctrl+C。如果出现 `torch/libshm.dll` / `shm.dll` 错误，也停止，不要重新做数据划分和前面的数据分析。

## 5. N=100跑通后，一次性跑完整实验

```powershell
uv run python -m src.run_sample_count_ablation `
  --split-dir "outputs\reports\ablation_800M" `
  --counts "100,200,500,1000,1500" `
  --output-dir "outputs\ablation_runs\800M"
```

N=1500 因训练池限制实际为每源1400，不会从验证/测试集补数据。

## 6. 最终比较

比较 N、实际每源训练数、micro-F1、macro-F1、Exact Match、各source F1。重点观察500到1000到1400的边际提升。

## 7. 已完成，不要重复

已完成：800M波形差异与饱和度分析；三个source各2000份确认；固定消融划分（每源训练1400/验证300/测试300）；五个消融split生成。正常情况下更新代码后从第3步开始；Dry Run已完成则从第4步开始。


## 8. 当前进度：N=100 已训练完成，下一步只跑固定 Test

当前 N=100 已完成训练：
- 每源训练 100，总训练样本 300。
- 固定验证集 900。
- Best Val Exact Match = 0.9833。
- Best Epoch = 16。

不要重新训练 N=100。先更新代码：

```powershell
cd D:\code\noise-source-identification
git switch feature/offline-synthetic-dataset
git pull --ff-only origin feature/offline-synthetic-dataset
```

然后只对已经保存的 N=100 best.pt 跑固定测试集：

```powershell
uv run python -m src.evaluate `
  --model "outputs\ablation_runs\800M\n100\checkpoints\best.pt" `
  --real-split test `
  --report "outputs\ablation_runs\800M\n100\reports\test_eval_report.json"
```

首先确认输出中有：

```text
split=real_test samples=900
```

记录 selected threshold summary 下的 overall 指标：
- micro_f1
- macro_f1
- sample_f1
- exact_match

测试报告保存在：

```text
outputs\ablation_runs\800M\n100\reports\test_eval_report.json
```

如果测试正常，后续完整消融 runner 已经支持“训练完成后自动用 best.pt 跑固定 test”，无需再手工逐个执行 evaluate。

> 注意：不要直接重新运行包含 N=100 的完整 runner，避免重复训练已经完成的 N=100。后续实验应从 N=200 开始。


## 9. N=100 验证通过后：自动运行剩余四档

N=100 已完成并确认：
- Best Val Exact Match = 0.9833
- Best Epoch = 16
- Fixed Test samples = 900
- Test micro-F1 = 0.9833
- Test macro-F1 = 0.9833
- Test sample-F1 = 0.9833
- Test Exact Match = 0.9833

因此不要再次训练 N=100。

更新代码后，从 N=200 开始一次性运行剩余四档：

```powershell
cd D:\code\noise-source-identification
git switch feature/offline-synthetic-dataset
git pull --ff-only origin feature/offline-synthetic-dataset

uv run python -m src.run_sample_count_ablation `
  --split-dir "outputs\reports\ablation_800M" `
  --counts "200,500,1000,1500" `
  --output-dir "outputs\ablation_runs\800M"
```

该命令会依次执行：
1. N=200：训练 -> 选择 best.pt -> 固定 Test=900
2. N=500：训练 -> 选择 best.pt -> 固定 Test=900
3. N=1000：训练 -> 选择 best.pt -> 固定 Test=900
4. N=1500 档：由于固定 val/test 后每源训练池只有 1400，因此实际使用每源 1400 -> 选择 best.pt -> 固定 Test=900

每一档训练结束后 runner 会自动调用 `src.evaluate --real-split test`，无需手工再运行测试命令。

每档测试报告位置：

```text
outputs\ablation_runs\800M\n200\reports\test_eval_report.json
outputs\ablation_runs\800M\n500\reports\test_eval_report.json
outputs\ablation_runs\800M\n1000\reports\test_eval_report.json
outputs\ablation_runs\800M\n1500\reports\test_eval_report.json
```

运行过程中，每档 Test 都应看到：

```text
split=real_test samples=900
```

全部完成后，保留各档的 best.pt 和 test_eval_report.json。最终比较 100 / 200 / 500 / 1000 / 1400（1500档实际1400）的固定 Test 指标，再决定推荐采集量。


## 10. 已完成训练后的固定 Test 补测（不要重新训练）

如果 N=100/200/500/1000/1500 已经全部训练完成，只看到 Best Exact Match / Best Epoch，而没有 `split=real_test samples=900`，不要重新运行训练 runner。

先更新代码：

```powershell
cd D:\code\noise-source-identification
git switch feature/offline-synthetic-dataset
git pull --ff-only origin feature/offline-synthetic-dataset
```

然后一次性对已经存在的 5 个 best.pt 跑同一个固定 Test=900：

```powershell
$counts = 100,200,500,1000,1500
foreach ($n in $counts) {
    Write-Host "===== N=$n FIXED TEST ====="
    uv run python -m src.evaluate `
      --model "outputs\ablation_runs\800M\n$n\checkpoints\best.pt" `
      --real-split test `
      --report "outputs\ablation_runs\800M\n$n\reports\test_eval_report.json"
}
```

每一档都必须出现：

```text
split=real_test samples=900
```

并记录 `selected threshold summary` 中的：
- micro_f1
- macro_f1
- sample_f1
- exact_match

测试报告分别保存在各自 `nXXX\reports\test_eval_report.json` 中。

注意：这一步只加载现有 best.pt 做测试，不会重新训练模型。


## 11. 下一阶段：S2 / S3 数据质量与类间重叠审计

样本量实验完成后，当前主要错误全部集中在 S2 与 S3 相互误判。结合采集现场可能存在“未采上”或“一份数据带多个特征”的情况，下一步先审计数据质量，不继续盲目增加训练样本。

本工具不会删除或修改任何原始 CSV，也不会自动把模型误判等同于脏数据。high / medium 仅表示需要人工复核。

先更新代码：

```powershell
cd D:\code\noise-source-identification
git switch feature/offline-synthetic-dataset
git pull --ff-only origin feature/offline-synthetic-dataset
```

### 11.1 找到 800M 下 S2、S3 的两个目录

将下面两个路径替换成公司电脑上实际的 S2 和 S3 800M 原始 CSV 目录，然后执行：

```powershell
uv run python -m src.audit_s2_s3_quality `
  --s2-dir "这里替换为S2的800M目录" `
  --s3-dir "这里替换为S3的800M目录" `
  --output-dir "outputs\reports\s2_s3_audit_800M" `
  --top 100
```

该脚本是 NumPy 分析工具，不依赖 PyTorch。

### 11.2 需要反馈的结果

运行结束后，终端会打印：

```text
S2 files=... risk={'high': ..., 'medium': ..., 'low': ...}
S3 files=... risk={'high': ..., 'medium': ..., 'low': ...}
audit_csv=...
summary=...
```

先把这两行 risk 数量反馈回来即可，不需要提供公司的原始 CSV。

输出文件：

```text
outputs\reports\s2_s3_audit_800M\s2_s3_audit.csv
outputs\reports\s2_s3_audit_800M\summary.json
```

CSV 中重点字段：
- own_distance：样本距离自身类别中心的距离；
- other_distance：样本距离另一类别中心的距离；
- margin_other_minus_own：正数越大越像自己的类别；负数表示在当前特征空间中反而更接近另一类；
- high：重点人工复核；
- medium：边界/类内异常候选；
- low：典型样本。

### 11.3 当前阶段禁止做的事情

不要因为 risk=high 就删除文件；不要根据模型预测直接改标签；不要重新采集全部数据。

先统计异常规模，再从 high 候选中抽取少量文件回看采集波形。确认究竟属于：
1. 没有真正采到目标源；
2. S2/S3 多特征混合；
3. 正常但处于真实类间边界；
4. 当前审计特征本身不够好。

完成这一步后，再决定是清洗数据、修改采集方法、修改标签策略，还是调整模型特征。


# 12. 800M 清洗后数据：从零完整重跑（当前唯一执行入口）

> 适用场景：三个源都已重新整理为每源 2000 条有效 CSV。旧 outputs 不再使用。本节从清空实验产物开始，一直执行到五档固定 Test。以后本轮实验只按本节操作，不再拼接前面零散命令。

## 12.1 更新代码并清空所有输出

确认当前分支后执行：

```powershell
cd D:\code\noise-source-identification
git switch feature/offline-synthetic-dataset
git pull --ff-only origin feature/offline-synthetic-dataset

if (Test-Path "outputs") {
    Remove-Item "outputs" -Recurse -Force
}
New-Item -ItemType Directory -Path "outputs" | Out-Null
Write-Host "outputs 已全部清空，本轮实验从零开始。"
```

这里只删除项目的 `outputs`，绝对不要删除原始 CSV 数据目录。

## 12.2 重新生成固定划分

把下面 `--raw-root` 改成当前 800M 三个源共同的父目录。该父目录下应能发现三个源目录，每源 2000 个 CSV。

```powershell
uv run python -m src.create_sample_count_ablation `
  --raw-root "这里替换为800M三个源的共同父目录" `
  --output-dir "outputs\reports\ablation_800M" `
  --counts "100,200,500,1000,1500" `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --seed 42
```

### 必须先核对

三个源都应显示：
- total = 2000；
- available train = 1400；
- val = 300；
- test = 300。

因此固定 Test 总数必须是 900。1500 档因为固定留出验证/测试集，实际最多使用 1400/源。

如果这里不是每源 2000，先停止，不训练。

## 12.3 五档重新训练

```powershell
uv run python -m src.run_sample_count_ablation `
  --config "configs\train.yaml" `
  --split-dir "outputs\reports\ablation_800M" `
  --counts "100,200,500,1000,1500" `
  --output-dir "outputs\ablation_runs\800M"
```

该步骤时间较长。每档会生成独立的 `best.pt`。训练过程中的大量 epoch 日志不需要记录。

如果当前 runner 在训练后自动执行 Test，可以让它完成；无论是否自动 Test，最终都执行下一节的“干净 Test”，以第 12.4 节终端短摘要为最终口述结果。

## 12.4 训练完成后，逐档执行干净 Test

不要把五档 Test 一次粘在一起。一个一个执行，每次只看终端最底部的：

`========== READ THIS TEST SUMMARY ==========`

### N=100

```powershell
uv run python -m src.evaluate `
  --model "outputs\ablation_runs\800M\n100\checkpoints\best.pt" `
  --real-split test `
  --report "outputs\ablation_runs\800M\n100\reports\test_eval_report.json"
```

### N=200

```powershell
uv run python -m src.evaluate `
  --model "outputs\ablation_runs\800M\n200\checkpoints\best.pt" `
  --real-split test `
  --report "outputs\ablation_runs\800M\n200\reports\test_eval_report.json"
```

### N=500

```powershell
uv run python -m src.evaluate `
  --model "outputs\ablation_runs\800M\n500\checkpoints\best.pt" `
  --real-split test `
  --report "outputs\ablation_runs\800M\n500\reports\test_eval_report.json"
```

### N=1000

```powershell
uv run python -m src.evaluate `
  --model "outputs\ablation_runs\800M\n1000\checkpoints\best.pt" `
  --real-split test `
  --report "outputs\ablation_runs\800M\n1000\reports\test_eval_report.json"
```

### N=1500（实际最多 1400/源）

```powershell
uv run python -m src.evaluate `
  --model "outputs\ablation_runs\800M\n1500\checkpoints\best.pt" `
  --real-split test `
  --report "outputs\ablation_runs\800M\n1500\reports\test_eval_report.json"
```

## 12.5 你只需要念终端最后这一块

新版 `evaluate.py` 会额外打印一个短摘要。前面再多日志都不用看，只看最后：

```text
========== READ THIS TEST SUMMARY ==========
samples=900 exact_match=.... macro_f1=....
source_1: f1=.... acc=....
source_3: f1=.... acc=....
source_5: f1=.... acc=....
total_errors=...
confusions:
  [真实标签] -> [预测标签]: 数量
============================================
```

每个 N 跑完后，只需要把这一块从上往下念出来。不需要打开 JSON，不需要找 CSV，不需要从长日志中自己统计 S2→S3 / S3→S2。

注意：仓库当前类别名历史上可能显示为 `source_1/source_3/source_5`，而现场口头习惯可能称 S1/S2/S3。以终端打印的 class_names 为准，后续分析时再做对应关系，不要人为改结果。

## 12.6 五档完成后再做数据质量审计

先把五个短摘要口述并完成样本量结论，再进入第 11 节 S2/S3 数据质量与类间重叠审计。不要在五档结果尚未确认前删除任何新的原始数据。


# 13. S2 / S3 局部邻域审计 V2（替代旧第 11 节 Risk 结果）

> 旧版第11节出现 S2 high=822、S3 high=3、两类 low=0，说明旧的“类别中心距离”规则过于激进。旧 Risk 结果作废，不据此删除任何文件。本节 V2 使用局部 k 近邻结构，判断典型样本、真实边界、被另一类邻域主导和孤立异常。

## 13.1 更新代码

```powershell
cd D:\code\noise-source-identification
git switch feature/offline-synthetic-dataset
git pull --ff-only origin feature/offline-synthetic-dataset
```

## 13.2 运行 V2 审计

只需要把两个目录替换成当前 800M 的 S2、S3 原始 CSV 目录：

```powershell
uv run python -m src.audit_s2_s3_quality `
  --s2-dir "这里替换为S2的800M目录" `
  --s3-dir "这里替换为S3的800M目录" `
  --output-dir "outputs\reports\s2_s3_audit_v2_800M" `
  --neighbors 30
```

该工具只依赖 NumPy，不训练模型，不修改原始数据。

## 13.3 只念最后三行数字

运行完成后，无论前面输出多少，只看：

```text
========== READ THIS S2/S3 AUDIT SUMMARY ==========
S2 files=2000 typical=... boundary=... other_dominated=... isolated=...
S3 files=2000 typical=... boundary=... other_dominated=... isolated=...
k_neighbors=30 parse_errors=...
====================================================
```

把这三行数字直接口述即可，不需要打开 JSON/CSV。

含义：
- `typical`：30个局部邻居中至少80%来自自身类别；
- `boundary`：自身类别邻居占50%～80%，属于 S2/S3 局部重叠区；
- `other_dominated`：自身类别邻居不足50%，当前特征空间更被另一类别包围；
- `isolated`：局部邻域距离超过该类别的保守外围阈值（Q3 + 3×IQR），属于真正值得优先人工复核的孤立候选。

任何类别都只是诊断标签，不能自动删除或改标签。

## 13.4 下一步停止点

得到三行摘要后先停止。不要继续清洗、删除或重新训练。

下一步将根据：
1. typical / boundary / other_dominated / isolated 的实际比例；
2. 前面固定 Test 中 S2↔S3 的 6～8 个错误；

决定是否需要把模型误判文件与 V2 审计结果自动交叉匹配。只有确认交叉关系后，才决定是否人工复核少量原始 CSV。


# 14. CNN Embedding × Test 误判交叉审计（当前执行步骤）

> 第13节手工特征 V2 得到 S2 几乎全部 other_dominated，与 CNN 在固定 Test 上约 99% 的实际分类能力矛盾。因此第13节结果只作为“手工特征空间不适用”的证据，不用于删数据。本节直接使用已训练 CNN 的 `encode()` 特征空间，并自动把模型真实误判与局部邻域位置交叉。

## 14.1 更新代码

```powershell
cd D:\code\noise-source-identification
git switch feature/offline-synthetic-dataset
git pull --ff-only origin feature/offline-synthetic-dataset
```

## 14.2 先分析 N=1000（当前错误最少：6）

```powershell
uv run python -m src.audit_embedding_errors `
  --model "outputs\ablation_runs\800M\n1000\checkpoints\best.pt" `
  --split-file "outputs\reports\ablation_800M\real_dataset_split_ablation_1000.csv" `
  --output-dir "outputs\reports\embedding_audit_800M_n1000" `
  --neighbors 30
```

如果你的 split 文件实际名称略有不同，只需要在 `outputs\reports\ablation_800M` 中确认 N=1000 对应的 CSV 文件名；不要改模型或重新训练。

## 14.3 只念最后输出块

```text
========== READ THIS EMBEDDING AUDIT SUMMARY ==========
model=best.pt test_samples=900 total_errors=...
source_1: samples=300 typical=... boundary=... other_dominated=... errors=...
source_3: samples=300 typical=... boundary=... other_dominated=... errors=...
source_5: samples=300 typical=... boundary=... other_dominated=... errors=...
error_locations: typical=... boundary=... other_dominated=...
confusions: ...
k_neighbors=30
========================================================
```

把这一整块直接口述即可，不需要打开 CSV 或 JSON。

其中：
- `typical`：CNN embedding 的30个最近邻中，至少80%与自己同类；
- `boundary`：50%～80%为自己同类；
- `other_dominated`：不足50%为自己同类；
- `error_locations`：最关键，直接告诉我们模型真正错的样本落在哪种区域；
- `confusions`：自动统计真实标签到预测标签的错误方向。

## 14.4 停止点

N=1000 跑完后先停止，把上面的短摘要口述出来。暂时不要跑 N=500/1400，也不要删除任何 CSV。

如果6个错误主要落在 boundary/other_dominated，下一步再自动提取这些少量文件做人工波形复核；如果错误主要落在 typical，则继续分析模型置信度和特征，而不是清洗原始数据。


# 15. CNN Embedding 审计修正版（替代第14节，当前执行入口）

> 第14节首次运行得到 17 个错误，而正式 N=1000 Test 为 6 个错误。原因是旧审计脚本没有完全复用正式 evaluate 的预测路径。第14节旧结果作废。本节脚本复用 `src.evaluate.collect_probabilities` / threshold 逻辑，并强制读取正式 Test JSON 做自动一致性校验。

## 15.1 更新

```powershell
cd D:\code\noise-source-identification
git switch feature/offline-synthetic-dataset
git pull --ff-only origin feature/offline-synthetic-dataset
```

## 15.2 运行 N=1000 修正版

使用第12.4节已经生成的正式 N=1000 Test 报告：

```powershell
uv run python -m src.audit_embedding_errors `
  --model "outputs\ablation_runs\800M\n1000\checkpoints\best.pt" `
  --split-file "outputs\reports\ablation_800M\real_dataset_split_ablation_1000.csv" `
  --eval-report "outputs\ablation_runs\800M\n1000\reports\test_eval_report.json" `
  --output-dir "outputs\reports\embedding_audit_800M_n1000_fixed" `
  --neighbors 30
```

## 15.3 自动安全校验

程序首先核对正式 Test JSON 与当前审计：
- Test 样本数必须一致；
- 正式 evaluate 的错误数与 embedding 审计错误数必须一致。

当前 N=1000 正式结果应对应 900 个 Test、6 个错误。这里的“6”不是写死在程序里，而是从 `test_eval_report.json` 自动读取/计算。

如果不一致，程序会直接：

```text
SAFETY CHECK FAILED
```

并停止，不允许解释 embedding。

只有看到：

```text
SAFETY_CHECK=PASS evaluate_errors=6 audit_errors=6
```

才继续读取下面的结果。

## 15.4 只口述最终摘要

成功时把以下整个短块念出来：

```text
========== READ THIS EMBEDDING AUDIT SUMMARY ==========
SAFETY_CHECK=PASS evaluate_errors=6 audit_errors=6
model=best.pt test_samples=900 total_errors=6
source_1: ...
source_3: ...
source_5: ...
error_locations: typical=... boundary=... other_dominated=...
confusions: ...
k_neighbors=30
========================================================
```

跑完先停止。不要删除数据，不需要重新训练，也暂时不要跑其他 N。
