from datetime import date
from torch.utils.data import Subset
from etc.dataset import ForecastDataset
def split_dataset_by_time(
    dataset: ForecastDataset,
    train_end: date,
    validation_end: date,
    test_end: date,
) -> tuple[Subset, Subset, Subset]:
    if not train_end < validation_end < test_end:
        raise ValueError("Split boundaries must be strictly increasing")
    train_indices=[]
    validation_indices=[]
    test_indices=[]
    for sample_index, (series_index, target_start) in enumerate(
            dataset.sample_index
    ):
        series = dataset.series[series_index]

        target_first = series["date"][target_start]
        target_last = series["date"][
            target_start + dataset.horizon - 1
            ]
        if target_last <= train_end:
            train_indices.append(sample_index)
        elif target_first > train_end and target_last <= validation_end:
            validation_indices.append(sample_index)
        elif target_first > validation_end and target_last <= test_end:
            test_indices.append(sample_index)
    return (
        Subset(dataset, train_indices),
        Subset(dataset, validation_indices),
        Subset(dataset, test_indices),
    )

