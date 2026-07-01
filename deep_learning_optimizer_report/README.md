# 深度学习优化器对比实验

本项目用于完成《深度学习导论》课程报告题目二：优化算法在深度学习中的演进与对比实验。

## 实验内容

- 数据集：优先使用 Fashion-MNIST，下载失败时自动切换 MNIST。
- 模型：小型 CNN。
- 主实验优化器：SGD + momentum、Adam、AdamW。
- 学习率策略：三组主实验统一使用 `CosineAnnealingLR`。
- 消融实验：AdamW `weight_decay=0` 与 `weight_decay=0.01` 对比。
- 输出：训练日志、逐轮指标 CSV、最终结果 CSV、损失曲线、准确率曲线、消融曲线、Word 报告。

## 运行方式

```bash
python train_optimizers.py
python generate_report.py
```

## 输出文件

```text
results/metrics.csv
results/final_results.csv
results/loss_curve.png
results/accuracy_curve.png
results/adamw_weight_decay_ablation.png
results/run_log.txt
results/run_screenshot.png
report/深度学习导论课程报告_优化算法对比实验.docx
```

## 复现说明

训练脚本默认参数为 `epochs=10`、`batch_size=128`、`seed=42`。脚本会自动检测 CUDA；如果存在 GPU，则优先使用 GPU 训练。报告中的表格和分析文字由 `generate_report.py` 从真实 CSV 和图片中读取生成。
