# PulmoSight: Explainable Tuberculosis Screening

PulmoSight is a research project exploring how deep learning can support
tuberculosis screening from chest X-ray images. It uses a convolutional neural
network, EfficientNet-B0, to classify an image as `TB` or `Normal` and presents a
Grad-CAM attribution map showing which image regions influenced the prediction.

## Project purpose

The project is designed as a transparent screening-support prototype. It combines
image classification, probability-based decision thresholds, validation metrics,
calibration analysis, and visual explanations in a single workflow.

The current dataset contains 16,807 Indian chest X-ray images:

- 13,445 training images
- 3,362 validation images
- 6,860 normal and 6,585 TB images in training
- 1,715 normal and 1,647 TB images in validation

The verified class mapping is `normal = 0` and `tb = 1`.

## Model

The classifier uses ImageNet-initialized EfficientNet-B0 adapted for grayscale
chest X-rays. Its current best saved checkpoint is from epoch 11 and achieved a
validation AUROC of approximately 0.9385. The project also evaluates sensitivity,
specificity, precision, negative predictive value, F1-score, AUPRC, calibration,
and threshold trade-offs instead of relying on accuracy alone.

## Explainability

Grad-CAM is used to show regions that contributed to the TB-class score. The
rainbow heatmap uses blue, green, yellow, and red attribution levels, with red
representing stronger model attribution. A lung-window display aid is included
for presentation, but it is not a trained anatomical segmentation model.

Grad-CAM is not exact lesion localization. Heatmap intensity is not disease
severity, and the current model cannot predict mild, moderate, or severe TB.

## Current web experience

The local web interface accepts one chest X-ray and presents:

- Model decision and TB probability
- The selected operating threshold
- Original X-ray, heatmap, and attribution overlay
- A presentation-ready comparison report
- Screening guidance and model limitations

## Current status

PulmoSight is **not yet complete or clinically validated**. The current project
does not contain an independent patient-level test set, lesion masks, lesion
bounding boxes, or severity labels. Patient-level independence also cannot be
verified from the available local filenames and metadata.

The next research stage is to obtain an untouched patient-level test set and,
if exact lesion location is required, a dataset with radiologist-created lesion
masks or bounding boxes. A separate severity-labelled dataset would be required
for a defensible severity model.

## Responsible interpretation

PulmoSight is not a medical diagnosis, does not identify other diseases, does not
measure disease severity, and does not confirm lesion location. A model result
should be interpreted with symptoms, clinical examination, radiology review, and
appropriate confirmatory testing under local clinical protocols and current WHO
guidance.

For a detailed faculty presentation, see
[docs/PROJECT_EXPLANATION.md](docs/PROJECT_EXPLANATION.md).
