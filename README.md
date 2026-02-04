# SwinCheX

This repository provides a maintained implementation aimed at reproducing the results of the paper
**["SwinCheX: Multi-label classification on chest X-ray images with transformer"](https://arxiv.org/abs/2206.04246)**.

The codebase was adapted to run efficiently on the **Heisenberg HPC cluster** at **Ilum – School of Science**, as part of the activities developed during the **Winter Internship at the [Brazilian Synchrotron Light Laboratory (LNLS)](https://lnls.cnpem.br/en/)** within the **[Data Science Group (GCD)](https://lnls.cnpem.br/grupos/gcd-en/)**.

## Main Code Adaptations

The original implementation was modified primarily to ensure compatibility with the HPC environment and modern GPU software stacks. The key changes include:

- Adoption of **native PyTorch Automatic Mixed Precision (AMP)**, replacing NVIDIA Apex, which is unavailable under **CUDA 12.1**
- Upgrade of core dependencies to the most recent versions mutually compatible with each other and with **CUDA 12.1**



## Training on the NIH ChestX-ray14 Dataset

### Initial Setup and Pretrained Models

Instructions for environment setup and pretrained model preparation are provided in
[`get_started.md`](get_started.md).

In this reproduction, we use the **ImageNet-22K pretrained Swin-T model**, configured with an input resolution of **224×224**, as the backbone for training.



### Requirements Installation

All required Python libraries are listed in [`requirements.txt`](requirements.txt).
To install them, run:

```bash
pip install -r requirements.txt
```


### Dataset Preparation

Download the **[NIH ChestX-ray14 Dataset](https://www.kaggle.com/nih-chest-xrays/data)** from Kaggle. 

After downloading, merge the images from the provided subfolders into a single directory.
Optionally, you may organize separate folders for training, validation, and testing.


### Single-GPU Training

To launch training on a single GPU, execute:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py \
  --local_rank 0 \
  --cfg configs/swin_tiny_patch4_window7_224.yaml \
  --resume swin_tiny_patch4_window7_224.pth \
  --trainset /home/mateus25032/work/GCD/SwinCheX/CheX-ray14/images \
  --validset /home/mateus25032/work/GCD/SwinCheX/CheX-ray14/images \
  --testset  /home/mateus25032/work/GCD/SwinCheX/CheX-ray14/images \
  --train_csv_path configs/NIH/train.csv \
  --valid_csv_path configs/NIH/validation.csv \
  --test_csv_path  configs/NIH/test.csv \
  --batch-size 4 \
  --accumulation-steps 4
```
## Results
The Swin-T-based model was trained for **300 epochs**, demonstrating strong generalization performance across both validation and test sets.

### Training Runtime

- Average training time per epoch: **23.66 min**

### Validation Results

The model achieved high discriminative capability during validation:

| Metric       | Value       |
| ------------ | ----------- |
| **Mean AUC** | **0.75440** |
| **Accuracy** | **95.57%**  |
| **Loss**     | **0.16972** |

### Test Results

On the held-out test set, the model maintained competitive performance:

| Metric       | Value       |
| ------------ | ----------- |
| **Mean AUC** | **0.72008** |
| **Accuracy** | **92.45%**  |
| **Loss**     | **0.23426** |


## 📌 Notes and Observations

- If all images are stored in a single directory, then the `--trainset`, `--validset`, and `--testset` arguments should all point to the same folder.

- To resume training from an intermediate checkpoint, you must comment out the line marked with **"TODO"** inside `utils.py`.
