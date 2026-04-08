import os
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_ALIASES = {
    "toys": "Toys",
    "music": "Music",
    "video": "Video",
    "ml1m": "ML1M",
    "ml-1m": "ML1M",
    "ml10m": "ML10M",
    "ml-10m": "ML10M",
    "yelp": "Yelp",
}


def normalize_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset: {name}. Choose from Toys, Music, Video, ML1M, ML10M, Yelp.")
    return DATASET_ALIASES[key]


class SequenceDataset(Dataset):
    def __init__(self, data: Dict):
        self.user_ids = []
        self.movie_sequences = []
        self.time_sequences = []
        self.target_movies = []
        self.first_order_users = []
        self.first_order_movies = []
        self.second_order_movies = []
        self.third_order_movies = []

        for _, user_data in data.items():
            self.user_ids.append(user_data["user_idx"])
            self.movie_sequences.append(user_data["movie_sequence"])
            self.time_sequences.append(user_data["time_sequence"])
            self.target_movies.append(user_data["target_movie"])
            self.first_order_users.append(user_data["first_order_users"])
            self.first_order_movies.append(user_data["first_order_movies"])
            self.second_order_movies.append(user_data["second_order_movies"])
            self.third_order_movies.append(user_data["third_order_movies"])

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "user_id": self.user_ids[idx],
            "movie_sequence": torch.tensor(self.movie_sequences[idx], dtype=torch.long),
            "time_sequence": torch.tensor(self.time_sequences[idx], dtype=torch.long),
            "target_movie": self.target_movies[idx],
            "first_order_users": torch.tensor(self.first_order_users[idx], dtype=torch.long),
            "first_order_movies": torch.tensor(self.first_order_movies[idx], dtype=torch.long),
            "second_order_movies": torch.tensor(self.second_order_movies[idx], dtype=torch.long),
            "third_order_movies": torch.tensor(self.third_order_movies[idx], dtype=torch.long),
        }


def _pad_1d_tensors(tensors: Iterable[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    tensors = list(tensors)
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for tensor in tensors:
        pad_len = max_len - tensor.size(0)
        if pad_len > 0:
            tensor = torch.cat([tensor, torch.full((pad_len,), pad_value, dtype=tensor.dtype)])
        padded.append(tensor)
    return torch.stack(padded)


def custom_collate_fn(batch):
    batch_user_ids = torch.tensor([item["user_id"] for item in batch], dtype=torch.long)
    batch_target_movies = torch.tensor([item["target_movie"] for item in batch], dtype=torch.long)

    movie_sequences = [item["movie_sequence"] for item in batch]
    time_sequences = [item["time_sequence"] for item in batch]
    seq_lengths = [len(seq) for seq in movie_sequences]
    max_seq_length = max(seq_lengths)

    padded_movie_sequences = []
    padded_time_sequences = []
    for i, seq_len in enumerate(seq_lengths):
        padded_movie_seq = torch.cat([
            torch.zeros(max_seq_length - seq_len, dtype=torch.long),
            batch[i]["movie_sequence"],
        ])
        padded_movie_sequences.append(padded_movie_seq)

        padded_time_seq = torch.cat([
            torch.zeros((max_seq_length - seq_len, 6), dtype=torch.long),
            batch[i]["time_sequence"],
        ])
        padded_time_sequences.append(padded_time_seq)

    return {
        "user_ids": batch_user_ids,
        "movie_sequences": torch.stack(padded_movie_sequences),
        "time_sequences": torch.stack(padded_time_sequences),
        "target_movies": batch_target_movies,
        "first_order_users": _pad_1d_tensors([item["first_order_users"] for item in batch]),
        "first_order_movies": _pad_1d_tensors([item["first_order_movies"] for item in batch]),
        "second_order_movies": _pad_1d_tensors([item["second_order_movies"] for item in batch]),
        "third_order_movies": _pad_1d_tensors([item["third_order_movies"] for item in batch]),
    }


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_dataset_paths(data_root: str | Path, dataset: str) -> Dict[str, Path]:
    dataset = normalize_dataset_name(dataset)
    root = Path(data_root)

    candidate_dirs = [
        root / dataset,
        root / dataset.lower(),
        root / dataset.upper(),
        root,
    ]

    required_names = {
        "train": "train_data_date.pkl",
        "val": "val_data_date.pkl",
        "test": "test_data_date.pkl",
        "num_users": "user_vocab_size_date.pkl",
        "num_items": "movie_vocab_size_date.pkl",
    }

    for base in candidate_dirs:
        paths = {key: base / filename for key, filename in required_names.items()}
        if all(path.exists() for path in paths.values()):
            return paths

    searched = "\n".join(str(x) for x in candidate_dirs)
    raise FileNotFoundError(
        f"Could not find dataset files for {dataset}. Searched:\n{searched}"
    )


def prepare_log_dir(output_root: str | Path) -> Path:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def seed_everything(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def count_workers(default: int = 4) -> int:
    cpu_count = os.cpu_count() or default
    return max(0, min(default, cpu_count))
