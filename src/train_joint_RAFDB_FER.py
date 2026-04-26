import os
import copy
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
from torchvision import transforms

import image_utils

SEARCH_SPACE = {
    "lr":      (1e-5, 5e-4),
    "beta":    (0.5,  0.9),
    "margin":  (0.01, 0.1),
    "mixup":   (0.1,  0.5),
    "gamma":   (1.0,  2.5),
    "dropout": (0.2,  0.5),
}

NUM_PARTICLES = 6
NUM_ITER      = 4
PSO_EPOCHS    = 5
FINAL_EPOCHS  = 80

W  = 0.5
C1 = 1.5
C2 = 1.5


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raf_path",   type=str, required=True)
    parser.add_argument("--fer_path",   type=str, required=True)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--save_dir",   type=str, default="models")

    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--workers",      type=int,   default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio",    type=float, default=0.2)
    parser.add_argument("--patience",     type=int,   default=12)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--use_tta_test", action="store_true")
    parser.add_argument("--run_pso",      action="store_true")

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_prefix_if_present(state_dict, prefixes=("module.",)):
    new_state = {}
    for k, v in state_dict.items():
        new_key = k
        for p in prefixes:
            if new_key.startswith(p):
                new_key = new_key[len(p):]
        new_state[new_key] = v
    return new_state


class EmotionFolderDataset(data.Dataset):
    def __init__(self, root_dir, phase="train", transform=None,
                 basic_aug=False, samples_override=None):
        self.phase     = phase
        self.transform = transform
        self.basic_aug = basic_aug
        self.root_dir  = Path(root_dir) / phase

        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} not found.")

        self.class_to_idx = {
            "surprise": 0,
            "fear":     1,
            "disgust":  2,
            "happy":    3,
            "sad":      4,
            "angry":    5,
            "neutral":  6,
        }

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if samples_override is not None:
            self.samples = samples_override
        else:
            self.samples = []
            for class_name, class_idx in self.class_to_idx.items():
                class_dir = self.root_dir / class_name
                if not class_dir.exists():
                    raise FileNotFoundError(f"Missing class folder: {class_dir}")
                for img_path in sorted(class_dir.iterdir()):
                    if img_path.suffix.lower() in valid_exts:
                        self.samples.append((str(img_path), class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for phase={phase} in {root_dir}")

        self.aug_func = [
            image_utils.flip_image,
            image_utils.add_gaussian_noise,
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image = image[:, :, ::-1]

        if self.phase == "train" and self.basic_aug and random.uniform(0, 1) > 0.5:
            aug_idx = random.randint(0, len(self.aug_func) - 1)
            image   = self.aug_func[aug_idx](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def stratified_split(samples, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    by_class = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    train_samples, val_samples = [], []
    for _, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def create_joint_dataloaders(args):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((236, 236)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.25, contrast=0.25,
                               saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.20)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    raf_train_full = EmotionFolderDataset(
        root_dir=args.raf_path, phase="train",
        transform=None, basic_aug=False
    )
    fer_train_full = EmotionFolderDataset(
        root_dir=args.fer_path, phase="train",
        transform=None, basic_aug=False
    )

    combined_train_samples = raf_train_full.samples + fer_train_full.samples
    train_samples, val_samples = stratified_split(
        combined_train_samples,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    train_dataset = EmotionFolderDataset(
        root_dir=args.raf_path, phase="train",
        transform=train_transform, basic_aug=True,
        samples_override=train_samples
    )
    val_dataset = EmotionFolderDataset(
        root_dir=args.raf_path, phase="train",
        transform=eval_transform, basic_aug=False,
        samples_override=val_samples
    )

    raf_test_dataset = EmotionFolderDataset(
        root_dir=args.raf_path, phase="test",
        transform=eval_transform, basic_aug=False
    )
    fer_test_dataset = EmotionFolderDataset(
        root_dir=args.fer_path, phase="test",
        transform=eval_transform, basic_aug=False
    )

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=False
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False
    )
    raf_test_loader = data.DataLoader(
        raf_test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False
    )
    fer_test_loader = data.DataLoader(
        fer_test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False
    )

    return (train_loader, val_loader, raf_test_loader, fer_test_loader,
            train_dataset, val_dataset, raf_test_dataset, fer_test_dataset)


class Res18SCN(nn.Module):
    def __init__(self, imagenet_pretrained=True, num_classes=7, drop_rate=0.3):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        self.backbone = models.resnet18(weights=weights)

        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(drop_rate)
        self.fc      = nn.Linear(feat_dim, num_classes)

        self.alpha = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)

        attention_weights = self.alpha(feat)
        logits            = self.fc(feat)
        out               = attention_weights * logits
        return attention_weights, out


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma) * ce_loss


def rank_regularization_loss(attention_weights, beta, margin):
    batch_sz = attention_weights.size(0)
    aw       = attention_weights.squeeze(1)

    tops = max(1, int(batch_sz * beta))
    if tops >= batch_sz and batch_sz > 1:
        tops = batch_sz - 1

    _, top_idx = torch.topk(aw, tops)
    _, low_idx = torch.topk(aw, batch_sz - tops, largest=False)

    mu_h = aw[top_idx].mean()
    mu_l = aw[low_idx].mean()

    return torch.clamp(mu_l - mu_h + margin, min=0.0)


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0

    lam        = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index      = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_pretrained_backbone(model, weight_path, device):
    print(f"Loading pretrained weights from: {weight_path}")
    ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, dict):
        state_dict = (ckpt.get("state_dict")
                      or ckpt.get("model_state_dict")
                      or ckpt)
    else:
        state_dict = ckpt

    state_dict    = strip_prefix_if_present(state_dict)
    backbone_dict = model.backbone.state_dict()

    loaded, skipped = 0, 0
    filtered_state  = {}
    for k, v in state_dict.items():
        if k.startswith("fc.") or k.startswith("feature."):
            skipped += 1
            continue
        if k in backbone_dict and backbone_dict[k].shape == v.shape:
            filtered_state[k] = v
            loaded += 1
        else:
            skipped += 1

    backbone_dict.update(filtered_state)
    model.backbone.load_state_dict(backbone_dict, strict=False)
    print(f"Loaded params: {loaded}, skipped: {skipped}")


def evaluate(model, loader, criterion, device, use_tta=False):
    model.eval()
    total_loss, total_correct, total_samples, iter_cnt = 0.0, 0, 0, 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            _, outputs = model(imgs)

            if use_tta:
                flipped_imgs    = torch.flip(imgs, dims=[3])
                _, outputs_flip = model(flipped_imgs)
                outputs         = (outputs + outputs_flip) / 2.0

            loss = criterion(outputs, targets)

            total_loss    += loss.item()
            iter_cnt      += 1
            preds          = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

    return total_loss / max(iter_cnt, 1), total_correct / total_samples


def train_model(base_args, params, epochs, save_path=None, final_mode=False):
    set_seed(base_args.seed)
    os.makedirs(base_args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    (train_loader, val_loader, raf_test_loader, fer_test_loader,
     train_dataset, val_dataset, raf_test_dataset, fer_test_dataset) = create_joint_dataloaders(base_args)

    print("Joint train split size :", len(train_dataset))
    print("Joint val   split size :", len(val_dataset))
    print("RAF-DB test size       :", len(raf_test_dataset))
    print("FER2013 test size      :", len(fer_test_dataset))

    model = Res18SCN(imagenet_pretrained=True, drop_rate=params["dropout"])
    if base_args.pretrained:
        load_pretrained_backbone(model, base_args.pretrained, device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=base_args.weight_decay
    )

    scheduler = None
    if final_mode:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

    criterion = FocalLoss(gamma=params["gamma"])

    best_val_acc      = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss  = 0.0
        total_correct = 0
        total_samples = 0
        iter_cnt      = 0

        for imgs, targets in train_loader:
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            mixed_imgs, targets_a, targets_b, lam = mixup_data(
                imgs, targets, alpha=params["mixup"]
            )
            attention_weights, outputs = model(mixed_imgs)

            ce_loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            rr_loss = rank_regularization_loss(
                attention_weights, beta=params["beta"], margin=params["margin"]
            )
            loss = ce_loss + rr_loss

            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            iter_cnt      += 1
            preds          = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(iter_cnt, 1)
        train_acc  = total_correct / total_samples
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_tta=False)

        print(f"[Epoch {epoch:3d}] "
              f"Train acc: {train_acc:.4f}  loss: {train_loss:.4f} | "
              f"Val acc: {val_acc:.4f}  loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0

            if save_path is not None:
                torch.save({
                    "iter":                 epoch,
                    "best_val_acc":         best_val_acc,
                    "best_params":          params,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, save_path)
                print(f"  Saved new best model -> {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= base_args.patience:
            print(f"\nEarly stopping after {base_args.patience} epochs without improvement.")
            break

    print(f"\nTraining finished.  Best val acc: {best_val_acc:.4f}")

    if not final_mode:
        return best_val_acc

    if save_path is not None and os.path.exists(save_path):
        print(f"\nLoading best model from {save_path} for final tests ...")
        best_ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=False)

        raf_test_loss, raf_test_acc = evaluate(
            model, raf_test_loader, criterion, device, use_tta=base_args.use_tta_test
        )
        fer_test_loss, fer_test_acc = evaluate(
            model, fer_test_loader, criterion, device, use_tta=base_args.use_tta_test
        )

        print(f"Final RAF-DB  TEST accuracy: {raf_test_acc:.4f}  loss: {raf_test_loss:.4f}")
        print(f"Final FER2013 TEST accuracy: {fer_test_acc:.4f}  loss: {fer_test_loss:.4f}")

    return best_val_acc


def _clip(value, key):
    lo, hi = SEARCH_SPACE[key]
    return max(lo, min(hi, value))


def init_particles():
    particles = []
    for _ in range(NUM_PARTICLES):
        position = {k: random.uniform(v[0], v[1])
                    for k, v in SEARCH_SPACE.items()}
        velocity = {k: 0.0 for k in SEARCH_SPACE}
        particles.append({"position": position, "velocity": velocity})
    return particles


def evaluate_particle(base_args, params):
    print(f"  Evaluating params: {params}")
    return train_model(
        base_args=base_args,
        params=params,
        epochs=PSO_EPOCHS,
        save_path=None,
        final_mode=False
    )


def run_pso(base_args):
    particles = init_particles()

    pbest_positions = [copy.deepcopy(p["position"]) for p in particles]
    pbest_scores    = [0.0] * NUM_PARTICLES
    gbest_position  = None
    gbest_score     = 0.0

    print("\nStarting PSO Hyperparameter Optimisation ...\n")

    for itr in range(NUM_ITER):
        print(f"\n{'='*50}")
        print(f"  PSO Iteration {itr + 1} / {NUM_ITER}")
        print(f"{'='*50}")

        for i, particle in enumerate(particles):
            print(f"\n  Particle {i + 1} / {NUM_PARTICLES}")

            score = evaluate_particle(base_args, particle["position"])
            print(f"  Val acc: {score:.4f}")

            if score > pbest_scores[i]:
                pbest_scores[i]    = score
                pbest_positions[i] = copy.deepcopy(particle["position"])
                print("  Personal best updated.")

            if score > gbest_score:
                gbest_score    = score
                gbest_position = copy.deepcopy(particle["position"])
                print("  New Global Best!")

        for i, particle in enumerate(particles):
            for k in SEARCH_SPACE:
                r1 = random.random()
                r2 = random.random()

                particle["velocity"][k] = (
                    W  * particle["velocity"][k]
                    + C1 * r1 * (pbest_positions[i][k] - particle["position"][k])
                    + C2 * r2 * (gbest_position[k]     - particle["position"][k])
                )
                particle["position"][k] = _clip(
                    particle["position"][k] + particle["velocity"][k], k
                )

    print("\nPSO Finished")
    print(f"  Best params : {gbest_position}")
    print(f"  Best val acc: {gbest_score:.4f}")
    return gbest_position


def final_training(base_args, best_params):
    print("\nStarting Final Full Training with Best Params\n")
    print("  Hyperparameters:", best_params)

    final_save_path = os.path.join(base_args.save_dir, "best_joint_RAFDB_FER.pth")

    train_model(
        base_args=base_args,
        params=best_params,
        epochs=FINAL_EPOCHS,
        save_path=final_save_path,
        final_mode=True
    )

    with open(os.path.join(base_args.save_dir, "best_params_pso_joint.json"),
              "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("\nFinal model saved as best_joint_RAFDB_FER.pth")


def run_training():
    args = parse_args()

    if args.run_pso:
        best_params = run_pso(args)
        np.save(
            os.path.join(args.save_dir, "best_params_pso_joint.npy"),
            best_params
        )
        final_training(args, best_params)

    else:
        default_params = {
            "lr":      5e-5,
            "beta":    0.60,
            "margin":  0.05,
            "mixup":   0.20,
            "gamma":   1.50,
            "dropout": 0.30,
        }
        final_training(args, default_params)


if __name__ == "__main__":
    run_training()