# Lightweight Multimodal LLM-Enabled Cost-Effective Defect Grading of Power Transmission Equipment in Power Grids (TII-26-6047)

Dear Editor,

We hope this letter finds you well.  
Thank you for giving the author team the opportunity to revise and resubmit this manuscript. This repository provides the complete code and a limited amount of sample data used in the study. We hope that these materials can help the reviewers further understand and assess our work. We sincerely hope that our revisions will receive your favorable consideration.

The Author Team

---

This repository covers three defect-grading tasks:

| Task | Defect | Grades |
|---|---|---|
| `Task1` | Tension-clamp crimping | Else, Kind |
| `Task2` | Guying-fitting corrosion | Else, Kind, Major |
| `Task3` | Tangent-tower bird nest | Else, Kind, Major, Urgent |

## Repository Guide

The repository follows the experimental workflow of the study:

```text
Sample data and expert standards
            ↓
1. Multi-resolution image processing
            ↓
2. Commercial-MLLM testing and automatic annotation
            ↓
3. Fine-tuning dataset construction
            ↓
4. MLLM fine-tuning and local deployment
            ↓
5. Fine-tuned-model performance verification
```

| Stage | Directory | Purpose |
|---|---|---|
| 1 | `1_Data_Processing` | Generate the image resolutions used in the experiments. |
| 2 | `2_DGPTE_Implementation_with_Commercial_MLLMs` | Test the DGPTE prompts with commercial MLLMs and annotate the training samples. |
| 3 | `3_Fine-Tuning_Dataset_Construction` | Convert the reviewed annotations and Prompt 4 components into LLaMA-Factory datasets. |
| 4 | `4_MLLM_Fine-Tuning_and_Local_Deployment` | Fine-tune, export, and deploy the multi-task MLLM. |
| 5 | `5_Performance_Verification_of_the_Fine-Tuned_MLLM` | Run the fine-tuned model and evaluate grading performance and CoT reliability. |

The numbered directories follow this order. Their README files contain the required configuration, commands, and output descriptions.

## Shared Directories

| Directory | Use |
|---|---|
| `Data` | Expert standards, original sample data, and generated fine-tuning datasets |
| `Prompt_Components` | Shared English prompt components for Task1–Task3 |
| `Metrics` | Grading-performance and CoT-reliability metric functions |

## Additional Supporting Materials

| Item | Purpose |
|---|---|
| `Partial Defect Grading Standard for Power Transmission Equipment (Operation Volume, 2022 Edition), 2022 (in Chinese).pdf` | Provides the cover and task-relevant excerpts from China Southern Power Grid's enterprise defect-grading standard. The relevant provisions for the three grading tasks are highlighted in yellow, documenting the source of the expert standards used in this study. |
| `Additional CPLID Benchmark Evaluation/` | Provides an external evaluation on the public CPLID insulator dataset, including data-processing and evaluation scripts, 60 fine-tuning annotations, the complete predictions for 1,257 test samples, and the calculated metrics. The fine-tuned MLLM achieves an ACC of 95.62%, an MF1 of 93.00%, and a Major recall of 100.00%. |

## Detailed Documentation

| Module | Documentation |
|---|---|
| 1_Data_Processing | [README](1_Data_Processing/README.md) |
| 2_DGPTE_Implementation_with_Commercial_MLLMs | [README](2_DGPTE_Implementation_with_Commercial_MLLMs/README.md) |
| 3_Fine-Tuning_Dataset_Construction | [README](3_Fine-Tuning_Dataset_Construction/README.md) |
| 4_MLLM_Fine-Tuning_and_Local_Deployment | [README](4_MLLM_Fine-Tuning_and_Local_Deployment/README.md) |
| 5_Performance_Verification_of_the_Fine-Tuned_MLLM | [README](5_Performance_Verification_of_the_Fine-Tuned_MLLM/README.md) |
| Data | [README](Data/README.md) |
| Prompt components | [README](Prompt_Components/README.md) |
| Metrics | [README](Metrics/README.md) |
| Additional CPLID benchmark evaluation | [README](<Additional CPLID Benchmark Evaluation/README.md>) |
| Partial defect-grading standard | [PDF](<Partial_Defect_Grading_Standard_for_Power_Transmission_Equipment(Operation_Volume_2022Edition)_2022(in_Chinese).pdf>) |
