# CODNet: Multimodal Object Understanding

A multimodal computer vision pipeline combining **VGG19-LSTM, YOLOv8, and BLIP**
for multi-label object classification, object detection, image captioning,
and visual question answering.

## Key Technologies

**Python · PyTorch · VGG19 · LSTM · YOLOv8 · BLIP · MS COCO 2017**

## Project Highlights

- Multi-label classification across **80 COCO object categories**
- Object detection using **pretrained YOLOv8n**
- Image captioning and visual question answering using **pretrained BLIP**
- Reproducible training and evaluation using **Kaggle**, with logs committed
- Quantitative evaluation using **mAP, micro-F1, and macro-F1**
- Qualitative failure-case analysis

## System Architecture

The pipeline combines three complementary components:

### 1. VGG19 + LSTM

- VGG19 extracts visual features from input images.
- The LSTM-based classifier uses these visual representations to predict
  the presence of COCO object categories.
- The classifier performs **multi-label classification** across 80 categories.

### 2. YOLOv8

- Uses pretrained **YOLOv8n** for object detection.
- Produces object-level bounding boxes and confidence scores.

### 3. BLIP

- Uses a pretrained BLIP model for **image captioning**.
- Uses BLIP for **visual question answering (VQA)**.

The outputs from these components provide complementary information about
the input image for multimodal visual understanding. Only the VGG19-LSTM
classifier was trained by us; YOLOv8 and BLIP use pretrained weights.

## Dataset

The classifier was trained and evaluated using the **MS COCO 2017** dataset,
which contains 80 object categories.

For the reproducible Kaggle experiment:

- **Training:** 116,287 `train2017` images
- **Validation:** 2,000 other `train2017` images, held out for epoch selection
- **Final evaluation:** 5,000 `val2017` images
- **Number of classes:** 80
- **Split:** seeded random permutation (seed 42)

## Team

- Abinaya P S
- Kavinshree M
- Ruchithra S

## My Contribution

- Worked with teammates on integrating the **VGG19 feature extractor with the LSTM-based classifier** for multi-label image classification.
- Implemented the inference pipeline using **pretrained YOLOv8 and BLIP** for object detection, image captioning, and visual question answering, with AI assistance.
- Integrated the VGG19-LSTM classifier with the YOLOv8 and BLIP components and evaluated their outputs.
- Conducted the **Kaggle retraining and `val2017` evaluation**, including experiment tracking and result analysis.
- Prepared result visualizations, cleaned the repository, and documented the experimental findings.

## Experimental Setup

The reproducible Kaggle experiment used:

- **Training images:** 116,287 `train2017` images
- **Validation images:** 2,000 `train2017` images
- **Final evaluation:** 5,000 `val2017` images
- **Training epochs:** 20
- **GPU:** Kaggle GPU
- **Training time:** approximately 4.3 hours (about 12.9 minutes per epoch); about 4.4 hours including the final evaluation
- **Random seed:** 42
- **Classification threshold:** 0.5
- **Model selection:** best epoch selected based on validation mAP (epoch 9, val mAP 0.627)

The final `val2017` evaluation set was not used for training or epoch
selection.

## Results

Trained on **116,287** `train2017` images, validated on 2,000 other
`train2017` images, and tested once on all **5,000 `val2017`** images.

| Metric | Value |
|---|---:|
| mAP (80 classes) | **0.617** |
| Micro-F1 @0.5 | **0.647** |
| Macro-F1 @0.5 | **0.561** |

For reference, per-label accuracy is 97.79% against an all-zeros baseline of
96.34%, which is why accuracy is not used as a primary metric.

The checkpoint with the best validation mAP (**epoch 9**, val mAP 0.627) was
used for the final test.

**Overfitting:** after epoch 9, training loss kept falling (0.0530 to 0.0361)
while validation mAP slipped from 0.627 to 0.601 by epoch 20. Selecting the
best-validation checkpoint avoids this, but later epochs do not help.

**Files for this run:**

- `runs/run2_116k_20ep/training_log.log` (full log, includes harmless
  DataLoader shutdown warnings)
- `runs/run2_116k_20ep/codnet_full_20ep.ipynb`
- `runs/run2_116k_20ep/per_class_AP.csv`
- `runs/run2_116k_20ep/per_class_AP.png`

The trained weights (about 83 MB) are not stored in this repository.

## Per-class Results

| Category | AP |
|---|---:|
| Person | 0.976 |
| Giraffe | 0.968 |
| Zebra | 0.963 |
| Elephant | 0.944 |
| Tennis racket | 0.931 |
| Toothbrush | 0.343 |
| Scissors | 0.296 |
| Spoon | 0.295 |
| Toaster | 0.086 |
| Hair drier | 0.038 |

Performance is strong on large, visually distinctive categories and weaker
on small objects and rare categories. For example, toaster and hair drier
have only 8 and 9 images, respectively, in the evaluation set.

## Failure Analysis

The image with the most wrong labels in `val2017` is `000000440475`:

- **True:** person, bowl, apple, chair, couch, dining table, tv, book, vase
- **Predicted:** bottle, bowl, potted plant, microwave, oven, sink, refrigerator

Only `bowl` is correct. The model reads the scene as a kitchen and misses
most of the actual objects, which shows how it leans on scene context for
cluttered indoor images.

## Evaluation Note

These runs used different settings and are kept for transparency. Their
numbers must not be compared directly with the main result above.

| Run | Train images | Epochs | mAP | Micro-F1 | Macro-F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Original cluster run | 117,266 | 25 | not measured | not measured | not measured | `logs/output_118k.log.txt` |
| Kaggle subset run | 18,000 | 10 | 0.504 | 0.560 | 0.424 | `runs/run1_18k_10ep/` |
| **Kaggle full run** | **116,287** | **20** | **0.617** | **0.647** | **0.561** | `runs/run2_116k_20ep/` |

The original cluster run reported an **AvgWA of 96.46%**. Further analysis
showed that this was per-label accuracy at a fixed threshold, measured on
training batches, making it unsuitable as the primary metric for evaluating
multi-label recognition performance. The cluster log shows the job running
from 08:48:14 to 09:16:39 IST on 28 April 2026 (about 28 minutes for
25 epochs).

The Kaggle runs therefore use **mAP, micro-F1, and macro-F1** on the held-out
`val2017` evaluation set.

The 18,000-image run used 18,000 training images and 2,000 additional
`train2017` images for validation, with approximately 29 minutes of training
time. It was an earlier, smaller reproduction.

The 116,287-image run used 116,287 training images and 2,000 additional
`train2017` images for validation, with approximately 4.3 hours of training
time. Including the final evaluation, the total runtime was about 4.4 hours.

## Loss Function

**Binary Cross-Entropy with Logits** (`torch.nn.BCEWithLogitsLoss`), averaged
over the batch and the 80 classes:

    L = -(1 / (N * 80)) * sum_i sum_c [
        y_ic * log(sigmoid(z_ic)) +
        (1 - y_ic) * log(1 - sigmoid(z_ic))
    ]

where `z` is the model output and `y` is the multi-hot ground-truth label.

Training configuration:

- **Optimizer:** Adam
- **Learning rate:** 1e-3
- **Batch size:** 64
- **VGG19:** ImageNet pretrained weights, frozen during classifier training

## Repository Structure

```text
CODNet/
├── train.py
├── pipeline.py
├── test_pipeline.py
├── requirements.txt
├── runs/
│   ├── run1_18k_10ep/        # earlier 18,000-image run
│   └── run2_116k_20ep/       # main result: full train2017, 20 epochs
├── logs/
│   └── output_118k.log.txt   # original cluster run log
├── results/                  # earlier output screenshots and curves
├── LICENSE
└── README.md
```

## Limitations

- The classifier overfits after about epoch 9; later epochs do not improve
  validation mAP.
- Rare and small categories (toaster, hair drier, scissors, toothbrush, spoon)
  perform poorly.
- The four models are not trained jointly.
- YOLOv8 detection was not evaluated (no box mAP).
- Images are resized to 224x224 and VGG19 is frozen, which limits accuracy on
  small objects.

## License

MIT, see `LICENSE`.
