---
title: "Introduction"
subject: Training
license: CC-BY-4.0
date: 2024-12-04
authors:
  - name: Ziyuan Nan
    email: nanziyuan21s@ict.ac.cn
    affiliation: ICT CAS
---

## Objective

Our primary goal is to conduct a fair and comprehensive comparison of the influence exerted by various training frameworks. To achieve this, we have devised a systematic methodology that ensures consistency and reproducibility across different frameworks.

## Methodology

Our approach can be summarized in three key steps:

1. **Data Preprocessing**: We begin by preprocessing the desired training dataset into a [standard format](../Data/format.md). This standardized format serves as a common ground, ensuring that all frameworks are evaluated on the same data.

2. **Framework-Specific Conversion**: For each framework under consideration, we convert the standardized dataset into the format required by that specific framework.

3. **Training and Evaluation**: For each framework, we proceed to train the dataset and record the loss metrics. This allows us to quantitatively compare the performance of different frameworks.

## Consistency and Variability

To ensure a fair comparison, we maintain consistency in several key aspects across all frameworks:

- **Common Data**: All frameworks are trained on the same standardized datasets, eliminating any bias that might arise from using different data sources.
- **Uniform Hyperparameters and Feature Set**: We use identical hyperparameters and feature sets (e.g., gradient checkpointing) across all frameworks. This ensures that any observed differences in performance are attributable to the frameworks themselves, rather than variations in configuration.

Despite these commonalities, there are distinct differences that we must account for:

- **Framework-Specific Characteristics**: Each framework has its own unique architecture and design principles.
- **Dataset Converters**: The process of converting the standardized dataset into a framework-specific format introduces variability.
- **Execution Scripts**: Each framework requires a specific bash script to run with the appropriate settings and hyperparameters.

## Hyperparameters

To be Done.