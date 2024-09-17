# ConDeDet: Vulnerability Detection via Contrastive Learning with Data and Control Dependency Comments

## Introduction


**ConDeDet** 是一种用于自动化软件漏洞检测的前沿方法，利用 **对比学习** 技术提升模型的检测能力。该方法的核心创新在于引入了 **数据依赖和控制依赖注释**，增强了模型捕捉代码深层次语义信息的能力。通过为代码添加依赖注释，ConDeDet 显著提升了在漏洞检测任务中的表现，在 Devign 和 Reveal 等标准数据集上表现优异。
## Key Features

- **Dependency Annotations:** Adds data and control dependency comments to code, improving vulnerability detection.
- **Contrastive Learning:** Leverages contrastive learning to better capture the relationship between annotated and non-annotated code snippets.
- **Pre-trained Model Integration:** Works with popular pre-trained models like CodeBERT and Code LLaMA, demonstrating strong generalization across different architectures.

## Project Structure

