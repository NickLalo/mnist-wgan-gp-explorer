"""MNIST data loading."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MNISTDataModule(L.LightningDataModule):
    """Download MNIST and expose reproducible train/validation/test loaders."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        batch_size: int = 256,
        num_workers: int = 8,
        validation_size: int = 5_000,
        seed: int = 112,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_size = validation_size
        self.seed = seed
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def prepare_data(self) -> None:
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            full = datasets.MNIST(
                self.data_dir, train=True, transform=self.transform, download=False
            )
            train_size = len(full) - self.validation_size
            generator = __import__("torch").Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                full, [train_size, self.validation_size], generator=generator
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = datasets.MNIST(
                self.data_dir, train=False, transform=self.transform, download=False
            )

    def _loader(self, dataset, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)
