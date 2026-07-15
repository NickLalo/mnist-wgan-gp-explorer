"""Independent MNIST feature classifier used only for evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTClassifier(nn.Module):
    """Small high-accuracy classifier with a 128-dimensional perceptual embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.feature_layer = nn.Linear(64 * 7 * 7, 128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, images: Tensor, *, return_features: bool = False):
        encoded = self.encoder(images).flatten(1)
        features = F.relu(self.feature_layer(encoded), inplace=True)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def classifier_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


@torch.inference_mode()
def classifier_accuracy(
    classifier: MNISTClassifier, loader: DataLoader, device: torch.device
) -> float:
    classifier.eval()
    correct = total = 0
    for images, labels in loader:
        predictions = classifier(images.to(device)).argmax(dim=1).cpu()
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / total


def ensure_classifier(
    data_dir: str | Path = "data",
    checkpoint_path: str | Path = "artifacts/classifier.pt",
    *,
    device: torch.device | None = None,
    epochs: int = 5,
    batch_size: int = 512,
) -> tuple[MNISTClassifier, float]:
    """Load the evaluator, or train it once if the local artifact is absent."""
    path = Path(checkpoint_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = MNISTClassifier().to(device)
    if path.exists():
        payload = torch.load(path, map_location=device, weights_only=True)
        classifier.load_state_dict(payload["state_dict"])
        classifier.eval()
        return classifier, float(payload["test_accuracy"])

    train_set = datasets.MNIST(
        data_dir, train=True, download=True, transform=classifier_transform()
    )
    test_set = datasets.MNIST(
        data_dir, train=False, download=True, transform=classifier_transform()
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = F.cross_entropy(classifier(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach())
        print(f"classifier epoch {epoch + 1}/{epochs} loss={running_loss / len(train_loader):.4f}")

    accuracy = classifier_accuracy(classifier, test_loader, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": classifier.state_dict(), "test_accuracy": accuracy}, path)
    classifier.eval()
    print(f"classifier test accuracy={accuracy:.4%}; saved {path}")
    return classifier, accuracy
