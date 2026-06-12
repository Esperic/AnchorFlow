from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .av2_dataset_map import Av2MapDataset, collate_fn_map


class Av2MapDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        data_folder: str = "forecast-mae-map-v2",
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        test: bool = False,
        validate_on_load: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.data_folder = data_folder
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test
        self.validate_on_load = validate_on_load

    def setup(self, stage: Optional[str] = None) -> None:
        cache_root = self.data_root / self.data_folder
        if not self.test:
            self.train_dataset = Av2MapDataset(
                cache_root,
                cached_split="train",
                validate_on_load=self.validate_on_load,
            )
            self.val_dataset = Av2MapDataset(
                cache_root,
                cached_split="val",
                validate_on_load=self.validate_on_load,
            )
        else:
            self.test_dataset = Av2MapDataset(
                cache_root,
                cached_split="test",
                validate_on_load=self.validate_on_load,
            )

    def _loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_map,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._loader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
        )
