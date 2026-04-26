import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raf_path", type=str, required=True)
    parser.add_argument("--fer_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drop_rate", type=float, default=0.3)

    return parser.parse_args()


class EmotionFolderDataset(data.Dataset):
    def __init__(self, root_dir, phase="test", transform=None):
        self.phase = phase
        self.transform = transform

        self.root_dir = Path(root_dir) / phase
        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} not found.")

        self.class_to_idx = {
            "surprise": 0,
            "fear": 1,
            "disgust": 2,
            "happy": 3,
            "sad": 4,
            "angry": 5,
            "neutral": 6,
        }

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.samples = []

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append((str(img_path), class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")

        image = image[:, :, ::-1]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class Res18SCN(nn.Module):
    def __init__(self, imagenet_pretrained=False, num_classes=7, drop_rate=0.3):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        self.backbone = models.resnet18(weights=weights)

        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(feat_dim, num_classes)
        self.alpha = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        attention_weights = self.alpha(feat)
        logits = self.fc(feat)
        out = attention_weights * logits
        return attention_weights, out


def load_model(model, model_path, device):
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    return model


def evaluate_and_report(model, loader, device, class_names):
    model.eval()
    total_correct = 0
    total_samples = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            _, outputs = model(imgs)

            preds = outputs.argmax(dim=1)

            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    accuracy = total_correct / total_samples
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=2,
        zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds)

    return accuracy, report, cm


def print_results(dataset_name, accuracy, report, cm):
    print(f"\n{'=' * 60}")
    print(f"{dataset_name} RESULTS")
    print(f"{'=' * 60}")

    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    raf_test_dataset = EmotionFolderDataset(
        root_dir=args.raf_path,
        phase="test",
        transform=eval_transform
    )
    fer_test_dataset = EmotionFolderDataset(
        root_dir=args.fer_path,
        phase="test",
        transform=eval_transform
    )

    raf_test_loader = data.DataLoader(
        raf_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    fer_test_loader = data.DataLoader(
        fer_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print("RAF-DB test size:", len(raf_test_dataset))
    print("FER2013 test size:", len(fer_test_dataset))

    model = Res18SCN(imagenet_pretrained=False, drop_rate=args.drop_rate)
    model = load_model(model, args.model_path, device)
    model = model.to(device)

    class_names = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

    raf_acc, raf_report, raf_cm = evaluate_and_report(
        model, raf_test_loader, device, class_names
    )
    fer_acc, fer_report, fer_cm = evaluate_and_report(
        model, fer_test_loader, device, class_names
    )

    print_results("RAF-DB TEST", raf_acc, raf_report, raf_cm)
    print_results("FER2013 TEST", fer_acc, fer_report, fer_cm)


if __name__ == "__main__":
    main()