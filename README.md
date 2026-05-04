# yoga-pose-classification

## Development

To get started you should use VSCode + Colab + Jupyter + Python extensions.

Each notebook in `src` is made to work on a colab environment, to start you only need to chose the colab kernel, either CPU or T4.

## Yoga-82 Action Classifier: Project Task Breakdown

### Data Exploration & Setup
- Download and extract Yoga-82 dataset (https://sites.google.com/view/yoga-82/home?pli=1)
- Analyze dataset structure (number of classes, train/val/test split, image count per class)
- Check image dimensions, formats, quality
- Identify class imbalance issues
- Visualize sample images from each class
- Create train/val/test splits (if not provided)
- Set up PyTorch Dataset and DataLoader classes
- Plan preprocessing pipeline (resizing, normalization, augmentation)

### Model Architecture & Baseline
- Research pre-trained models for action classification (ResNet, EfficientNet, ViT)
- Implement baseline model (transfer learning approach)
- Set up training loop with loss function and optimizer
- Implement validation and testing pipelines
- Train baseline model and log initial metrics
- Generate confusion matrix, per-class precision/recall/F1 for baseline
- Document baseline performance and identify weak classes

### Optimization & Iterative Improvement
- Experiment 1: Data augmentation strategies → analyze metrics and confusion matrix
- Experiment 2: Learning rate scheduling and optimizer tuning → analyze improvements
- Experiment 3: Model architecture variants → compare per-class performance
- Experiment 4: Class weighting/rebalancing if needed → analyze impact
- Implement early stopping and model checkpointing
- Analyze failure cases (which poses/classes remain problematic)
- Measure inference time and memory usage for final model
- Document computational cost vs. accuracy trade-offs

### Final Results Summary
- Generate comprehensive results (confusion matrix, metrics, visualizations)
- Document final model architecture and hyperparameters
- Create summary of what worked and what didn't
- Prepare experiment logs and analysis for technical report