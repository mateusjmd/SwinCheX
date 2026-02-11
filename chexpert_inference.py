# -*- coding: latin-1 -*-
"""
Inference script for SwinCheX on CheXpert
----------------------------------------
This script performs inference-only evaluation (AUC-ROC, Accuracy)
on the CheXpert dataset, reusing the SwinCheX model and checkpoint.
It is intentionally decoupled from training and DDP logic.


This script can be executed from the command line as follows:

python chexpert_inference.py \
    --cfg path/to/model/configs \
    --checkpoint path/to/model/checkpoint \
    --chexpert_csv path/to/chexpert/data/csv \
    --chexpert_root path/to/chexpert/dataset/root \
    --device cuda
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from types import SimpleNamespace
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

from config import get_config
from models import build_model


# --------------------------------------------------
# NIH classes (original training space)
# --------------------------------------------------
NIH_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

# --------------------------------------------------
# CHEXPERT CLASSES (inference space)
# --------------------------------------------------
CHEXPERT_CLASSES = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax"
    ]


# --------------------------------------------------
# Argument parser (clean, inference-only)
# --------------------------------------------------
def build_fake_args(cfg_path):
    return SimpleNamespace(
        cfg=cfg_path,
        opts=None,

        # paths / datasets
        data_path=None,
        trainset=None,
        validset=None,
        testset=None,
        train_csv_path=None,
        valid_csv_path=None,
        test_csv_path=None,

        # output / logging
        output=None,
        tag=None,

        # training flags
        batch_size=1,
        accumulation_steps=1,
        use_checkpoint=False,
        amp_opt_level="O0",
        resume=None,
        eval=True,
        throughput=False,

        # data handling
        zip=False,
        cache_mode="no",

        # NIH / SwinCheX-specific
        #num_mlp_heads=0,
        num_mlp_heads=3,

        # distributed (dummy)
        local_rank=0,
    )


# --------------------------------------------------
# Argument parser (CheXpert inference)
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Inference on CheXpert with SwinCheX")

    parser.add_argument("--cfg", required=True,
                        help="Path to SwinCheX config YAML")

    parser.add_argument("--checkpoint", required=True,
                        help="Best checkpoint (.pth, selected by AUC-ROC)")

    parser.add_argument("--chexpert_csv", required=True,
                        help="CheXpert CSV file (labels)")

    parser.add_argument("--chexpert_root", required=True,
                        help="Root directory of CheXpert images")

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

# --------------------------------------------------
# Image utilities
# --------------------------------------------------
def preprocess_image(img, cfg):
    img = img.resize((cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0

    # HWC -> CHW
    img = torch.from_numpy(img).permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = (img - mean) / std
    return img


# --------------------------------------------------
# CheXpert Dataset (for inference only)
# --------------------------------------------------
class CheXpertDataset(Dataset):
    """
    CheXpert Dataset for inference with SwinCheX.

    - Evaluates only the intersection of NIH and CheXpert classes
    - Maps CheXpert uncertainty (-1) to negative (0)
    - Uses SwinCheX preprocess_image
    """

    CHEXPERT_CLASSES = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax"
    ]

    NIH_EQUIVALENT = {
        "Atelectasis": "Atelectasis",
        "Cardiomegaly": "Cardiomegaly",
        "Consolidation": "Consolidation",
        "Edema": "Edema",
        "Pleural Effusion": "Effusion",
        "Pneumonia": "Pneumonia",
        "Pneumothorax": "Pneumothorax"
    }

    def __init__(self, csv_file, img_root, cfg):
        self.df = pd.read_csv(csv_file)
        self.img_root = Path(img_root)
        self.cfg = cfg

        # Column indices for CheXpert classes
        self.class_cols = [
            self.df.columns.get_loc(cls)
            for cls in self.CHEXPERT_CLASSES
        ]

        # Keep NIH class order for correct logit alignment
        self.nih_indices = [
            NIH_CLASSES.index(self.NIH_EQUIVALENT[cls])
            for cls in self.CHEXPERT_CLASSES
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Image
        img_rel_path = self.df.iloc[idx, 0]
        img_path = self.img_root / img_rel_path

        img = Image.open(img_path).convert("RGB")
        img = preprocess_image(img, self.cfg)

        # Labels
        labels = self.df.iloc[idx, self.class_cols].values.astype(np.float32)

        # Uncertainty handling: -1 -> 0
        labels[labels == -1] = 0.0

        labels = torch.from_numpy(labels)

        return img, labels


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()

    # (1) Load config
    cfg_args = build_fake_args(args.cfg)
    cfg = get_config(cfg_args)

    # (2) Build model
    model = build_model(cfg)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=args.device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=False)

    model.to(args.device)
    model.eval()

    # --------------------------------------------------
    # (3) Dataset + DataLoader
    # --------------------------------------------------
    dataset = CheXpertDataset(
        csv_file=args.chexpert_csv,
        img_root=args.chexpert_root,
        cfg=cfg
    )

    # NIH-14 indices corresponding to the 7 CheXpert classes
    chexpert_nih_indices = [0, 1, 2, 3, 4, 12, 13]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --------------------------------------------------
    # (4) Inference
    # --------------------------------------------------
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            scores = model(images)  # list with 14 tensors [B,2]

            # Extracts only relevant heads
            selected_logits = []

            for idx in chexpert_nih_indices:
                head_logits = scores[idx]        # [B,2]
                positive_logit = head_logits[:, 1]  # positive class
                selected_logits.append(positive_logit)

            logits = torch.stack(selected_logits, dim=1)  # [B,7]
            probs = torch.sigmoid(logits)                 # [B,7]

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_probs).numpy()   # [N,7]
    y_true = torch.cat(all_labels).numpy()  # [N,7]

    # --------------------------------------------------
    # (5) Metrics
    # --------------------------------------------------
    auc_per_class = []
    acc_per_class = []

    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        auc_per_class.append(auc)

        y_bin = (y_pred[:, i] >= 0.5).astype(int)
        acc = accuracy_score(y_true[:, i], y_bin)
        acc_per_class.append(acc)

    print("\n=== CheXpert-7 Inference Results ===")
    for class_name, auc, acc in zip(CHEXPERT_CLASSES, auc_per_class, acc_per_class):
        print(f"{class_name:20s} | AUC = {auc:.4f} | ACC = {acc:.4f}")


    print("---------------------------------")
    print(f"Mean AUC : {np.mean(auc_per_class):.4f}")
    print(f"Mean ACC : {np.mean(acc_per_class):.4f}")


if __name__ == "__main__":
    main()
