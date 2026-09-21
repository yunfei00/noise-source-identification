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
