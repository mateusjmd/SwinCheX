# -*- coding: latin-1 -*-
# gradcam_inference.py
"""
Grad-CAM inference script for SwinCheX
------------------------------------
This script performs single-image Grad-CAM without DDP, datasets, or training logic.
It is intentionally decoupled from the training pipeline.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace
from pathlib import Path


from config import get_config
from models import build_model

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
TARGET_CLASS_NAME = 'CLASS'
TARGET_CLASS_IDX = NIH_CLASSES.index(TARGET_CLASS_NAME)



# --------------------------------------------------
# Argument parser (clean, inference-only)
# --------------------------------------------------
def build_fake_args(cfg_path):
    return SimpleNamespace(
        # obrigat�rio
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
        num_mlp_heads=0,

        # distributed (dummy)
        local_rank=0,
    )
def parse_args():
    parser = argparse.ArgumentParser("Grad-CAM inference for SwinCheX")
    parser.add_argument("--cfg", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pth)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--target_layer", required=False,
                        help="Layer name for Grad-CAM (e.g. layers.3.blocks.1.norm2)")
    parser.add_argument("--class_idx", type=int, default=None,
                        help="Target class index (default: argmax)")
    parser.add_argument("--output_dir", default="./gradcam_outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# --------------------------------------------------
# Utility: find module by dotted name
# --------------------------------------------------
def get_module_by_name(model, name):
    module = model
    for attr in name.split('.'):
        if attr.isdigit():
            module = module[int(attr)]
        else:
            module = getattr(module, attr)
    return module


# --------------------------------------------------
# Grad-CAM core
# --------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def get_cam(self):
        # gradients, activations: (B, N, C)
        grads = self.gradients[0]        # (N, C)
        acts = self.activations[0]       # (N, C)

        # Weigths per token
        weights = grads.mean(dim=1)      # (N,)

        cam = (weights[:, None] * acts).sum(dim=1)  # (N,)
        cam = torch.relu(cam)

        # Infer spacial grid (ex: 7x7, 14x14)
        num_tokens = cam.shape[0]
        size = int(num_tokens ** 0.5)
        cam = cam.reshape(size, size)

        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam



    def _forward_hook(self, module, inp, out):
        self.activations = out  # [B, C, H, W]

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]  # [B, C, H, W]

    def __call__(self, logits, class_idx):
        score = logits[:, class_idx].sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam[0]
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.cpu().numpy()

# --------------------------------------------------
# Image utilities
# --------------------------------------------------
def preprocess_image(img, cfg):
    img = img.resize((cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # HWC ? CHW
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img - mean) / std
    #img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img.unsqueeze(0)
    return img


def export_overlay(cam, original_image, out_path, alpha=0.6):
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    H, W, _ = original_image.shape
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, alpha, heatmap, 1 - alpha, 0)

    cv2.imwrite(out_path, overlay)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # (1) Load config
    cfg_args = build_fake_args(args.cfg)
    cfg = get_config(cfg_args)

    print("DEBUG cfg.NIH.num_mlp_heads =", cfg.NIH.num_mlp_heads)

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

    # (3) Grad-CAM target layer
    target_layer = model.layers[2].blocks[-1].norm2
    gradcam = GradCAM(model, target_layer)

    # (4) Load images
    image_path = Path(args.image)

    if image_path.is_dir():
        image_list = sorted([
            p for p in image_path.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])
    else:
        image_list = [image_path]

    for img_path in image_list:
        print(f"Processing: {img_path.name}")

        pil_img = Image.open(img_path).convert("RGB")
        original_image = np.array(pil_img)

        input_tensor = preprocess_image(pil_img, cfg).to(args.device)

        # (5) Forward + backward
        with torch.enable_grad():
            scores = model(input_tensor)
            logits = torch.stack(scores, dim=0).mean(dim=0)

            class_idx = (
                logits.argmax(dim=1).item()
                if args.class_idx is None
                else args.class_idx
            )

            score = logits[:, class_idx].sum()
            model.zero_grad()
            score.backward()

        # (6) Grad-CAM
        cam = gradcam.get_cam()

        cam = cv2.resize(
            cam,
            (original_image.shape[1], original_image.shape[0])
        )

        out_path = os.path.join(
            args.output_dir,
            f"{img_path.stem}_gradcam_class_{class_idx}.png"
        )

        export_overlay(cam, original_image, out_path)
        print(f"Grad-CAM saved to {out_path}")

if __name__ == "__main__":
    main()
