from torch.utils.data import Dataset, DataLoader


def create_dataloader(dataset: Dataset, batch_size: int,shuffle:bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
