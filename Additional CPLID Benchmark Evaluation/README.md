# CPLID 公共数据集附加实验

为回应 Reviewer 3, Comment 1，我们在 CPLID 公共数据集上补充了绝缘子缺陷分类实验。本文件夹提供该实验的数据处理与测试代码、60 个微调样本的 JSON 语料、测试结果及指标统计。

CPLID 图像可从原作者公开的仓库获得：

<https://github.com/InsulatorData/InsulatorDataSet>

考虑到图像数量及仓库大小，本文件夹不重复上传 CPLID 原始图像和处理后的实验图像。下载公开数据集后，可使用所提供的脚本复现实验数据划分和全部图像处理过程。

注意：由于Github子文件数量的上限为1000，实验部分仅上传了前1000个结果的Json

## 数据划分

CPLID 包含 600 张正常绝缘子图像和 248 张缺陷绝缘子图像。我们使用 XML 文件中的 `insulator` 边界框标出待判断的绝缘子串。每个样本只保留一个蓝色框；当原始图像中包含多个绝缘子框时，分别生成多个样本。除此之外，不对原始图像进行裁剪、拼接或文字叠加。

| 数据 | Normal | Major | 合计 |
|---|---:|---:|---:|
| 参考样本 | 2 | 2 | 4 |
| 微调样本 | 30 | 30 | 60 |
| 测试样本 | 1,041 | 216 | 1,257 |
| **合计** | **1,073** | **248** | **1,321** |

数据划分使用固定随机种子 `6047`，并以原始图像为单位进行。由同一张原始图像生成的样本不会被分配到不同数据子集中。

## 文件说明

| 目录或文件 | 内容 |
|---|---|
| `reference image/` | 4 份参考样本 JSON 语料，包括 2 个 Normal 样本和 2 个 Major 样本。 |
| `train image/` | 60 份微调样本 JSON 语料，包括 30 个 Normal 样本和 30 个 Major 样本。 |
| `scripts/` | 数据处理、微调语料构建、模型测试和指标计算脚本。 |
| `results/predictions/` | 1,257 份测试样本预测结果。 |
| `results/metrics.csv` | 测试集总体指标和混淆矩阵。 |
| `results/prompt_preview.txt` | 模型测试时使用的提示词。 |
| `prompt_config.py` | 任务定义、定级标准和提示词。 |
| `split_manifest.json` | 参考样本、微调样本和测试样本的划分记录。 |
| `insulator_60.json` | 最终使用的 60 个样本微调语料。 |
| `微调语料提示词预览.txt` | 微调输入格式预览。 |
| `requirements.txt` | Python 依赖。 |

### 数据与提示词文件

| 文件 | 内容 |
|---|---|
| `prompt_config.py` | 实验使用的任务说明、Normal/Major 判定标准、输出格式和四个参考样本。 |
| `split_manifest.json` | 每个样本所属的数据子集、类别、原始图像、XML 标注和边界框。 |
| `reference image/` | 四个参考样本的 JSON 语料，用于构造多模态参考示例。 |
| `train image/` | 60 个微调样本的 JSON 语料。 |
| `insulator_60.json` | 整理后的最终微调文件。每条数据由四个参考样本和一个待判断样本构成，图像均使用相对路径。 |
| `微调语料提示词预览.txt` | `insulator_60.json` 中图像与文字输入顺序的可读版本。 |

### 实验脚本

| 脚本 | 作用 |
|---|---|
| `01_prepare_dataset.py` | 读取 CPLID 图像和 XML 标注，绘制蓝色绝缘子框，并按照 `split_manifest.json` 生成参考、微调和测试图像。 |
| `02_generate_reference_json.py` | 调用商业多模态模型生成四个参考样本的 JSON 语料。 |
| `03_generate_training_json.py` | 以四个参考样本为示例，生成 60 个微调样本的 JSON 语料。 |
| `04_build_finetuning_dataset.py` | 将参考样本和微调样本整理为 `insulator_60.json`。 |
| `05_test_finetuned_model.py` | 通过 Ollama 调用微调后的模型，并将每个测试样本的输出保存到 `results/predictions/`。 |
| `06_calculate_metrics.py` | 读取逐样本预测结果，计算分类指标并生成 `results/metrics.csv`。 |

## 运行顺序

下载 CPLID 后，首先生成实验图像：

```bash
python scripts/01_prepare_dataset.py --clean --source /path/to/InsulatorDataSet
```

该脚本将生成 `reference image/`、`train image/` 和 `test image/` 中对应的实验图像。仓库已提供参考样本和微调样本的 JSON 语料，因此可直接构建微调文件：

```bash
python scripts/04_build_finetuning_dataset.py
```

微调完成后，使用以下命令检查测试输入并运行完整测试：

```bash
python scripts/05_test_finetuned_model.py --dry-run
python scripts/05_test_finetuned_model.py
```

如需根据现有逐样本结果重新计算指标，运行：

```bash
python scripts/06_calculate_metrics.py
```

## 测试结果

作者团队基于该附加的CPLID绝缘子数据集进行了Qwen3-VL-8B的微调，微调的模型名为 `paper_gllm_insulator_60_language_e300:latest`。该模型使用 30 个 Normal 样本和 30 个 Major 样本进行微调，并在其余 1,257 个样本上进行测试。

| 指标 | 结果 |
|---|---:|
| Accuracy | 95.62% |
| Macro-Precision | 89.85% |
| Macro-Recall | 97.36% |
| Macro-F1 | 93.00% |
| Normal Precision | 100.00% |
| Normal Recall | 94.72% |
| Normal F1 | 97.29% |
| Major Precision | 79.70% |
| Major Recall | 100.00% |
| Major F1 | 88.71% |

`results/predictions/` 提供全部 1,257 个测试样本的模型输出，`results/metrics.csv` 提供由这些结果计算得到的总体指标。
