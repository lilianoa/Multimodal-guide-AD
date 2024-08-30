## Multimodal-guide-AD

## Highlights
![MFD](docs/fig2.png)

> **<p align="justify"> Abstract:** Anomaly detection aims to distinguish normal from abnormal images, with applications in industrial defect detection and medical imaging. Current methods using textual information often focus on designing effective textual prompts but overlook their full utilization. This paper proposes a multimodal fusion network that integrates image and text information to improve anomaly detection. The network comprises an image encoder, text encoder, and stacked cross-attention module. To address the absence of text during inference, an image-only branch is introduced, guided by the multimodal fusion network through knowledge distillation. Experiments on industrial anomaly detection and medical image datasets demonstrate the effectiveness of our approach, achieving AUROC and AUPR scores of 96.5% and 89.2% on VisA, respectively.

## Anomaly detection Results on VisA
Fine-grained data-subset-wise performance comparison (AUROC, AUPR) for anomaly classification on VisA. Please refer to our paper for more details.

| Objects     | CLIP         | WinCLIP          | VAND           | AnomalyCLIP      | Ours              |
|-------------|--------------|------------------|----------------|------------------|-------------------|
| candle      | (42.9, 37.9) | (95.8, **95.4**) | (86.9, 83.8)   | (81.1, 79.3)     | (**99**, 91.8)    |
| capsules    | (81, 69.7)   | **(90.9, 85)**   | (74.3, 61.2)   | (88.7, 81.5)     | (90.8, 65.8)      |
| cashew      | (83.4, 69.1) | (96.4, 92.1)     | (94.1, 87.3)   | (89.4, 76.3)     | **(98.3, 92.81)** |
| chewinggum  | (90.4, 77.5) | (98.6, 96.5)     | (98.4, 96.4)   | **(98.9, 97.4)** | (94, 91.6)        |
| fryum       | (82, 67.2)   | (90.1, 80.3)     | (97.2, 94.3)   | (96.8, 93)       | **(99.1, 96.5)**  |
| macaroni1   | (56.8, 64.4) | (75.8, 76.2)     | (70.9, 71.6)   | (86, 87.2)       | **(98.8, 93.6)**  |
| macaroni2   | (65, 65)     | (60.3, 63.7)     | (63.2, 64.6)   | (72.1, 73.4)     | **(91.7, 75.4)**  |
| pcb1        | (56.9, 54.9) | (78.4, 73.6)     | (57.2, 53.4)   | (87, 85.4)       | **(99.2, 95.6)**  |
| pcb2        | (63.2, 62.6) | (49.2, 51.2)     | (73.8, 71.8)   | (64.3, 62.2)     | **(93.1, 81.9)**  |
| pcb3        | (53, 52.2)   | (76.5, 73.4)     | (70.7, 66.8)   | (70, 62.7)       | **(99.8, 98.2)**  |
| pcb4        | (88, 87.7)   | (77.7, 79.6)     | (95.1, **95**) | (94.4, 93.9)     | (**97.6**, 93.5)  |
| pipe_fryuml | (94.6, 88.8) | (82.3, 69.7)     | (94.8, 89.9)   | (96.3, 92.4)     | **(96.8, 94)**    |
| mean        | (71.5, 66.4) | (81.2, 78.1)     | (81.4, 78.0)   | (85.4, 82.1)     | **(96.5, 89.2)**  |
------------------------------------------------------------
### Visualization of Grad-CAM on VisA
![visa](docs/fig.png)

## Environment:

Install the required packages:
```
pip install -r requirements.txt
```

## Datasets Download:
Put the datasets in `./data` folder.

### [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)

```
data
|----visa
|-----|-- split_csv
|-----|-----|--- 1cls.csv
|-----|-----|--- 2cls_fewshot.csv
|-----|-----|--- ......
|-----|-- candle
|-----|-----|--- Data
|-----|-----|-----|----- Images
|-----|-----|-----|--------|------ Anomaly 
|-----|-----|-----|--------|------ Normal 
|-----|-----|-----|----- Masks
|-----|-----|-----|--------|------ Anomaly 
|-----|-----|--- image_anno.csv
|-----|-- capsules
|-----|--- ...
```

VisA dataset need to be preprocessed to separate the train set from the test set.

```
python ./datasets/visa_json.py
```


## How to Run

The script `run.py` and 'test.py' provides a simple illustration. For example, to run the training and evaluation on VisA with image-only input with knowledge distillation, you can use the following command:

```
python run.py
python test.py
```
Follow the configuration in `./options/KDstudent.yaml`.

