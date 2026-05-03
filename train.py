import argparse
import csv
import os
import socket
import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model_utils import SUPPORTED_MODELS, get_model


def dataset_exists(data_dir: str) -> bool:
    return os.path.exists(os.path.join(data_dir, "cifar-10-batches-py"))


def get_existing_best_accuracy(checkpoint_path: str) -> float:
    if not os.path.exists(checkpoint_path):
        return 0.0

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return float(checkpoint.get("best_val_acc", checkpoint.get("val_acc", 0.0)))
    except Exception as exc:
        print(f"Warning: could not read existing checkpoint {checkpoint_path}: {exc}")
        return 0.0


def save_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    epoch: int,
    model_name: str,
    val_acc: float,
    val_loss: float,
    train_acc: float,
    train_loss: float,
    image_size: int,
    experiment_tag: str,
    sm_allocation: str,
):
    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": float(val_acc),
            "val_acc": float(val_acc),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "train_loss": float(train_loss),
            "image_size": int(image_size),
            "num_classes": 10,
            "experiment_tag": experiment_tag,
            "sm_allocation": sm_allocation,
        },
        checkpoint_path,
    )


def export_cifar10_test_folder(data_dir: str) -> str:
    export_dir = os.path.join(data_dir, "cifar10_test_images")
    marker_path = os.path.join(export_dir, ".complete")

    if os.path.exists(marker_path):
        print(f"CIFAR-10 test image folder already exists: {export_dir}")
        return export_dir

    print(f"Creating CIFAR-10 test image folder: {export_dir}")

    raw_test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=not dataset_exists(data_dir),
    )

    for idx, (image_array, label) in enumerate(
        zip(raw_test_dataset.data, raw_test_dataset.targets)
    ):
        class_name = raw_test_dataset.classes[label]
        class_dir = os.path.join(export_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        image_path = os.path.join(class_dir, f"{idx:05d}.png")
        Image.fromarray(image_array).save(image_path)

    with open(marker_path, "w", encoding="utf-8") as marker_file:
        marker_file.write("complete\n")

    return export_dir


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total, total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Training with checkpointing and timing instrumentation"
    )

    parser.add_argument("--model", type=str, default="resnet18", choices=SUPPORTED_MODELS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    parser.add_argument("--image_size", type=int, default=64)

    parser.add_argument("--use_subset", action="store_true")
    parser.add_argument("--train_subset_size", type=int, default=5000)
    parser.add_argument("--test_subset_size", type=int, default=1000)

    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument(
        "--create_test_folder",
        action="store_true",
        help="Export CIFAR-10 test images once to data_dir/cifar10_test_images.",
    )

    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Initialize model without ImageNet pretrained weights.",
    )

    parser.add_argument("--experiment_tag", type=str, default="baseline")
    parser.add_argument("--sm_allocation", type=str, default="native")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hostname = socket.gethostname()

    print(f"Host: {hostname}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Experiment: {args.experiment_tag}")
    print(f"SM allocation: {args.sm_allocation}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(
            f"GPU: {props.name}, "
            f"{props.total_memory // (1024 ** 2)} MB, "
            f"{props.multi_processor_count} SMs"
        )

    download_flag = not dataset_exists(args.data_dir)
    print("\nDownloading CIFAR-10..." if download_flag else "\nCIFAR-10 already cached.")

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=download_flag,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=download_flag,
        transform=test_transform,
    )

    if args.create_test_folder:
        export_cifar10_test_folder(args.data_dir)

    if args.use_subset:
        train_dataset = Subset(
            train_dataset,
            range(min(args.train_subset_size, len(train_dataset))),
        )
        test_dataset = Subset(
            test_dataset,
            range(min(args.test_subset_size, len(test_dataset))),
        )
        print(f"Using subset: {len(train_dataset)} train, {len(test_dataset)} test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    num_train_images = len(train_dataset)

    print(f"\nBuilding {args.model}...")

    model = get_model(
        args.model,
        num_classes=10,
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pth")
    best_acc = get_existing_best_accuracy(checkpoint_path)

    print(f"Best checkpoint so far for {args.model}: {best_acc:.4f}")

    csv_path = os.path.join(
        args.output_dir,
        f"results_{args.model}_{args.experiment_tag}.csv",
    )

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(
        [
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "epoch_time_sec",
            "throughput_img_per_sec",
            "model",
            "experiment_tag",
            "sm_allocation",
            "device",
            "hostname",
            "checkpoint_saved",
            "checkpoint_path",
        ]
    )

    total_start = time.time()

    for epoch in range(args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 60}")

        epoch_start = time.time()

        train_loss, train_acc, images_processed = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.time() - epoch_start
        throughput = images_processed / epoch_time if epoch_time > 0 else 0.0

        checkpoint_saved = False

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s | Throughput: {throughput:.1f} img/s")

        if val_acc > best_acc:
            best_acc = val_acc

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                model_name=args.model,
                val_acc=val_acc,
                val_loss=val_loss,
                train_acc=train_acc,
                train_loss=train_loss,
                image_size=args.image_size,
                experiment_tag=args.experiment_tag,
                sm_allocation=args.sm_allocation,
            )

            checkpoint_saved = True
            print(f"  >> New best checkpoint saved at {checkpoint_path}")

        csv_writer.writerow(
            [
                epoch + 1,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                f"{epoch_time:.2f}",
                f"{throughput:.1f}",
                args.model,
                args.experiment_tag,
                args.sm_allocation,
                str(device),
                hostname,
                checkpoint_saved,
                checkpoint_path if checkpoint_saved else "",
            ]
        )

        csv_file.flush()

    total_time = time.time() - total_start
    avg_throughput = (
        (num_train_images * args.epochs) / total_time if total_time > 0 else 0.0
    )

    csv_file.close()

    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model:          {args.model}")
    print(f"Experiment:     {args.experiment_tag}")
    print(f"SM Allocation:  {args.sm_allocation}")
    print(f"Epochs:         {args.epochs}")
    print(f"Total Time:     {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"Avg Throughput: {avg_throughput:.1f} img/s")
    print(f"Best Val Acc:   {best_acc:.4f}")
    print(f"Results CSV:    {csv_path}")
    print(f"Best Checkpoint:{checkpoint_path}")


if __name__ == "__main__":
    main()
