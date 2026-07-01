import argparse
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")

    def log(self, message: str = ""):
        print(message, flush=True)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    optimizer_name: str
    experiment: str
    lr: float
    momentum: float = 0.0
    weight_decay: float = 0.0


def load_datasets(data_dir: Path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    attempts = [
        ("FashionMNIST", datasets.FashionMNIST),
        ("MNIST", datasets.MNIST),
    ]
    errors = []
    for dataset_name, dataset_cls in attempts:
        try:
            train_dataset = dataset_cls(
                root=str(data_dir), train=True, download=True, transform=transform
            )
            test_dataset = dataset_cls(
                root=str(data_dir), train=False, download=True, transform=transform
            )
            return dataset_name, train_dataset, test_dataset
        except Exception as exc:  # pragma: no cover - only used when download mirrors fail.
            errors.append(f"{dataset_name}: {repr(exc)}")
    raise RuntimeError("Dataset download failed. " + " | ".join(errors))


def make_loaders(train_dataset, test_dataset, batch_size: int, num_workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common_kwargs
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)
    return train_loader, test_loader


def build_optimizer(config: ExperimentConfig, model: nn.Module):
    if config.optimizer_name == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name == "Adam":
        return optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    if config.optimizer_name == "AdamW":
        return optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    raise ValueError(f"Unknown optimizer: {config.optimizer_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
    return total_loss / total, total_correct / total * 100.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
    return total_loss / total, total_correct / total * 100.0


def convergence_description(metrics: pd.DataFrame, target_acc: float):
    reached = metrics.loc[metrics["test_acc"] >= target_acc]
    if not reached.empty:
        epoch = int(reached.iloc[0]["epoch"])
        return f"第{epoch}轮达到{target_acc:.1f}%测试准确率"
    best_row = metrics.loc[metrics["test_acc"].idxmax()]
    return (
        f"{len(metrics)}轮内未达到{target_acc:.1f}%，"
        f"最佳为第{int(best_row['epoch'])}轮{best_row['test_acc']:.2f}%"
    )


def plot_main_curves(metrics_df: pd.DataFrame):
    main_runs = ["SGD", "Adam", "AdamW"]
    colors = {"SGD": "#1f77b4", "Adam": "#d62728", "AdamW": "#2ca02c"}

    plt.figure(figsize=(10, 6), dpi=180)
    for run_name in main_runs:
        sub = metrics_df[metrics_df["run_name"] == run_name]
        if sub.empty:
            continue
        plt.plot(
            sub["epoch"],
            sub["train_loss"],
            color=colors[run_name],
            linestyle="-",
            marker="o",
            markersize=3,
            label=f"{run_name} train_loss",
        )
        plt.plot(
            sub["epoch"],
            sub["test_loss"],
            color=colors[run_name],
            linestyle="--",
            marker="s",
            markersize=3,
            label=f"{run_name} test_loss",
        )
    plt.title("Loss Curves of SGD, Adam and AdamW")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(10, 6), dpi=180)
    for run_name in main_runs:
        sub = metrics_df[metrics_df["run_name"] == run_name]
        if sub.empty:
            continue
        plt.plot(
            sub["epoch"],
            sub["train_acc"],
            color=colors[run_name],
            linestyle="-",
            marker="o",
            markersize=3,
            label=f"{run_name} train_acc",
        )
        plt.plot(
            sub["epoch"],
            sub["test_acc"],
            color=colors[run_name],
            linestyle="--",
            marker="s",
            markersize=3,
            label=f"{run_name} test_acc",
        )
    plt.title("Accuracy Curves of SGD, Adam and AdamW")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(60, 100)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_curve.png")
    plt.close()


def plot_ablation(metrics_df: pd.DataFrame):
    labels = {
        "AdamW_no_wd": "AdamW weight_decay=0",
        "AdamW": "AdamW weight_decay=0.01",
    }
    colors = {"AdamW_no_wd": "#9467bd", "AdamW": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)

    for run_name, label in labels.items():
        sub = metrics_df[metrics_df["run_name"] == run_name]
        if sub.empty:
            continue
        axes[0].plot(
            sub["epoch"],
            sub["test_acc"],
            marker="o",
            markersize=3,
            color=colors[run_name],
            label=label,
        )
        axes[1].plot(
            sub["epoch"],
            sub["test_loss"],
            marker="s",
            markersize=3,
            color=colors[run_name],
            label=label,
        )

    axes[0].set_title("AdamW Weight Decay Ablation: Test Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8)

    axes[1].set_title("AdamW Weight Decay Ablation: Test Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Loss")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "adamw_weight_decay_ablation.png")
    plt.close(fig)


def make_log_screenshot(log_path: Path, output_path: Path):
    text = log_path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    keep = []
    for line in lines:
        if (
            line.startswith("Experiment")
            or line.startswith("Dataset")
            or line.startswith("Device")
            or line.startswith("Python")
            or "Epoch" in line
            or line.startswith("Final")
            or line.startswith("Saved")
        ):
            keep.append(line)
    keep = keep[-34:]

    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ]
    font_path = next((p for p in font_candidates if Path(p).exists()), None)
    font = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
    line_height = 28
    width = 1800
    height = max(520, 70 + line_height * len(keep))
    image = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(image)
    draw.text((26, 22), "$ python train_optimizers.py", font=font, fill=(210, 245, 210))
    y = 62
    for line in keep:
        draw.text((26, y), line[:150], font=font, fill=(232, 232, 232))
        y += line_height
    image.save(output_path)


def run_experiment(
    config: ExperimentConfig,
    dataset_name: str,
    train_dataset,
    test_dataset,
    args,
    device,
    logger: TeeLogger,
):
    set_seed(args.seed)
    train_loader, test_loader = make_loaders(
        train_dataset, test_dataset, args.batch_size, args.num_workers, args.seed
    )
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )

    logger.log(
        f"Experiment {config.run_name}: optimizer={config.optimizer_name}, "
        f"lr={config.lr}, momentum={config.momentum}, "
        f"weight_decay={config.weight_decay}, scheduler=CosineAnnealingLR"
    )

    rows = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        epoch_seconds = time.time() - start
        row = {
            "dataset": dataset_name,
            "experiment": config.experiment,
            "run_name": config.run_name,
            "optimizer": config.optimizer_name,
            "epoch": epoch,
            "lr": config.lr,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_seconds": epoch_seconds,
        }
        rows.append(row)
        logger.log(
            f"{config.run_name:12s} | Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | "
            f"test_loss={test_loss:.4f} | test_acc={test_acc:.2f}% | "
            f"time={epoch_seconds:.1f}s"
        )
    logger.log("")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare SGD, Adam and AdamW on Fashion-MNIST/MNIST."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--target-acc", type=float, default=90.0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "run_log.txt"
    logger = TeeLogger(log_path)
    start_all = time.time()

    try:
        set_seed(args.seed)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.set_float32_matmul_precision("high")
        else:
            device = torch.device("cpu")

        dataset_name, train_dataset, test_dataset = load_datasets(DATA_DIR)
        logger.log("Deep Learning Optimizer Comparison")
        logger.log(f"Dataset: {dataset_name}")
        logger.log(f"Train samples: {len(train_dataset)}")
        logger.log(f"Test samples: {len(test_dataset)}")
        logger.log(f"Device: {device}")
        if device.type == "cuda":
            logger.log(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.log(f"Python: {sys.version.split()[0]}")
        logger.log(f"PyTorch: {torch.__version__}")
        logger.log(f"Torchvision: {torchvision.__version__}")
        logger.log(f"Platform: {platform.platform()}")
        logger.log(f"Epochs: {args.epochs}")
        logger.log(f"Batch size: {args.batch_size}")
        logger.log(f"Seed: {args.seed}")
        logger.log("")

        configs = [
            ExperimentConfig("SGD", "SGD", "main", lr=0.05, momentum=0.9, weight_decay=5e-4),
            ExperimentConfig("Adam", "Adam", "main", lr=0.001, weight_decay=0.0),
            ExperimentConfig("AdamW", "AdamW", "main", lr=0.001, weight_decay=0.01),
            ExperimentConfig(
                "AdamW_no_wd", "AdamW", "ablation", lr=0.001, weight_decay=0.0
            ),
        ]

        all_rows = []
        for config in configs:
            all_rows.extend(
                run_experiment(
                    config,
                    dataset_name,
                    train_dataset,
                    test_dataset,
                    args,
                    device,
                    logger,
                )
            )

        metrics_df = pd.DataFrame(all_rows)
        metrics_path = RESULTS_DIR / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        final_rows = []
        for run_name, sub in metrics_df.groupby("run_name", sort=False):
            final = sub.sort_values("epoch").iloc[-1]
            best_idx = sub["test_acc"].idxmax()
            best = sub.loc[best_idx]
            final_rows.append(
                {
                    "dataset": final["dataset"],
                    "experiment": final["experiment"],
                    "optimizer": run_name,
                    "optimizer_type": final["optimizer"],
                    "lr": final["lr"],
                    "momentum": final["momentum"],
                    "weight_decay": final["weight_decay"],
                    "best_test_acc": best["test_acc"],
                    "best_epoch": int(best["epoch"]),
                    "final_test_acc": final["test_acc"],
                    "final_train_acc": final["train_acc"],
                    "final_train_loss": final["train_loss"],
                    "final_test_loss": final["test_loss"],
                    "convergence_speed": convergence_description(
                        sub.sort_values("epoch"), args.target_acc
                    ),
                }
            )
        final_df = pd.DataFrame(final_rows)
        final_path = RESULTS_DIR / "final_results.csv"
        final_df.to_csv(final_path, index=False)

        plot_main_curves(metrics_df)
        plot_ablation(metrics_df)

        meta = {
            "dataset": dataset_name,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "platform": platform.platform(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "scheduler": "CosineAnnealingLR",
            "eta_min": args.eta_min,
            "target_acc": args.target_acc,
            "total_seconds": time.time() - start_all,
        }
        (RESULTS_DIR / "experiment_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.log("Final results:")
        logger.log(final_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        logger.log("")
        logger.log(f"Saved metrics: {metrics_path}")
        logger.log(f"Saved final results: {final_path}")
        logger.log(f"Saved loss curve: {RESULTS_DIR / 'loss_curve.png'}")
        logger.log(f"Saved accuracy curve: {RESULTS_DIR / 'accuracy_curve.png'}")
        logger.log(
            f"Saved AdamW ablation curve: {RESULTS_DIR / 'adamw_weight_decay_ablation.png'}"
        )
        logger.log(f"Total elapsed seconds: {time.time() - start_all:.1f}")
    finally:
        logger.close()

    make_log_screenshot(log_path, RESULTS_DIR / "run_screenshot.png")


if __name__ == "__main__":
    main()
