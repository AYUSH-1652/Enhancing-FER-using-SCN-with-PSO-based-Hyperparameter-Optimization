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


class RafFolderDataset(data.Dataset):
    def __init__(self, root_dir, phase="test", transform=None):
        self.transform = transform
        self.root_dir = Path(root_dir) / phase

        self.class_to_idx = {
            "surprise": 0,
            "fear": 1,
            "disgust": 2,
            "happy": 3,
            "sad": 4,
            "angry": 5,
            "neutral": 6,
        }

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.samples = []

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append((str(img_path), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image = image[:, :, ::-1]

        if self.transform:
            image = self.transform(image)

        return image, label


class Res18SCN(nn.Module):
    def __init__(self, num_classes=7, drop_rate=0.3):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        fc_in = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(fc_in, num_classes)
        self.alpha = nn.Sequential(
            nn.Linear(fc_in, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        att = self.alpha(feat)
        logits = self.fc(feat)
        out = att * logits
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raf_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = RafFolderDataset(args.raf_path, phase="test", transform=transform)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = Res18SCN(drop_rate=0.3).to(device)

    print("Loading model:", args.model_path)
    ckpt = torch.load(args.model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: {acc:.4f}")

    class_names = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()