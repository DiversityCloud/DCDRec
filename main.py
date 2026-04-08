import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DiffusionRecommender
from trainer import Trainer, format_metrics
from utils import (
    SequenceDataset,
    count_workers,
    custom_collate_fn,
    get_device,
    load_pickle,
    normalize_dataset_name,
    prepare_log_dir,
    resolve_dataset_paths,
    seed_everything,
)


SUPPORTED_DATASETS = ["Toys", "Music", "Video", "ML1M", "ML10M", "Yelp"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the diffusion recommender.")
    parser.add_argument("--dataset", type=str, default="Toys", choices=SUPPORTED_DATASETS)
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--output_dir", type=str, default="./log")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=count_workers())
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--x_dim", type=int, default=64)
    parser.add_argument("--num_diffusion_steps", type=int, default=500)
    parser.add_argument("--guidance_scale", type=float, default=0.9)
    parser.add_argument("--unconditional_prob", type=float, default=0.5)
    parser.add_argument("--num_neg", type=int, default=100)
    return parser.parse_args()



def build_loaders(paths, batch_size: int, num_workers: int):
    train_data = load_pickle(paths["train"])
    val_data = load_pickle(paths["val"])
    test_data = load_pickle(paths["test"])

    train_dataset = SequenceDataset(train_data)
    val_dataset = SequenceDataset(val_data)
    test_dataset = SequenceDataset(test_data)

    common_kwargs = {
        "batch_size": batch_size,
        "collate_fn": custom_collate_fn,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader



def main():
    args = parse_args()
    dataset = normalize_dataset_name(args.dataset)
    seed_everything(args.seed)

    device = get_device(args.device)
    paths = resolve_dataset_paths(args.data_root, dataset)
    log_dir = prepare_log_dir(args.output_dir)
    log_path = log_dir / f"{dataset}.txt"

    num_users = load_pickle(paths["num_users"])
    num_items = load_pickle(paths["num_items"])
    train_loader, val_loader, test_loader = build_loaders(paths, args.batch_size, args.num_workers)

    model = DiffusionRecommender(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        x_dim=args.x_dim,
        num_diffusion_steps=args.num_diffusion_steps,
        device=str(device),
        guidance_scale=args.guidance_scale,
        unconditional_prob=args.unconditional_prob,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model=model, optimizer=optimizer, device=device, num_neg=args.num_neg)

    best_epoch = -1
    best_metric = float("-inf")
    best_metrics = None
    best_state = None

    with open(log_path, "w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            trainer.train_one_epoch(train_loader)
            val_metrics, _, _ = trainer.evaluate(val_loader, return_timing=True)

            summary = f"Epoch {epoch:03d} | {format_metrics(val_metrics, prefix='Val')}"
            print(summary)
            log_file.write(summary + "\n")
            log_file.flush()

            if val_metrics["NDCG@10"] > best_metric:
                best_metric = val_metrics["NDCG@10"]
                best_epoch = epoch
                best_metrics = dict(val_metrics)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        best_line = f"BestEpoch={best_epoch} | {format_metrics(best_metrics, prefix='BestVal')}"
        print(best_line)
        log_file.write(best_line + "\n")

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics, _, _ = trainer.evaluate(test_loader, return_timing=True)
        test_line = f"Test | {format_metrics(test_metrics, prefix='Test')}"
        print(test_line)
        log_file.write(test_line + "\n")

        ckpt_path = log_dir / f"{dataset}_best.pt"
        if best_state is not None:
            torch.save(best_state, ckpt_path)
            log_file.write(f"Checkpoint={ckpt_path}\n")

    print(f"Saved log to: {log_path}")


if __name__ == "__main__":
    main()
