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
