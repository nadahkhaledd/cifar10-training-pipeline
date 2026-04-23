import os
import csv
import copy
import time
import argparse
import socket

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


def get_model(model_name: str, num_classes: int):
    """Build and return a model with the final layer adjusted for num_classes."""
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}. "
                         f"Choose from: resnet18, mobilenet_v2, efficientnet_b0, vgg16")

    return model


def dataset_exists(data_dir):
    return os.path.exists(os.path.join(data_dir, "cifar-10-batches-py"))


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return loss, accuracy, and number of images processed."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
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
    """Evaluate and return loss and accuracy."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training with Timing Instrumentation")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "mobilenet_v2", "efficientnet_b0", "vgg16"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--use_subset", action="store_true")
    parser.add_argument("--train_subset_size", type=int, default=5000)
    parser.add_argument("--test_subset_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    # Experiment metadata: tag this run for later analysis
    parser.add_argument("--experiment_tag", type=str, default="baseline",
                        help="Tag for this experiment (e.g., baseline, flyt-100pct, flyt-50-50)")
    parser.add_argument("--sm_allocation", type=str, default="native",
                        help="SM allocation description (e.g., native, 142, 71, 43)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hostname = socket.gethostname()

    # Print GPU info if available
    print(f"Host: {hostname}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Experiment: {args.experiment_tag}")
    print(f"SM allocation: {args.sm_allocation}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory // (1024**2)} MB, {props.multi_processor_count} SMs")

    # Dataset
    download_flag = not dataset_exists(args.data_dir)
    if download_flag:
        print("\nDownloading CIFAR-10...")
    else:
        print("\nCIFAR-10 already cached.")

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True,
                                     download=download_flag, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                    download=download_flag, transform=test_transform)

    if args.use_subset:
        train_dataset = Subset(train_dataset, range(min(args.train_subset_size, len(train_dataset))))
        test_dataset = Subset(test_dataset, range(min(args.test_subset_size, len(test_dataset))))
        print(f"Using subset: {len(train_dataset)} train, {len(test_dataset)} test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    num_train_images = len(train_dataset)

    # Model
    print(f"\nBuilding {args.model}...")
    model = get_model(args.model, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # CSV results file for this experiment
    csv_path = os.path.join(args.output_dir,
                            f"results_{args.model}_{args.experiment_tag}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                         "epoch_time_sec", "throughput_img_per_sec",
                         "model", "experiment_tag", "sm_allocation", "device", "hostname"])

    total_start = time.time()

    # Training loop with per-epoch timing
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        epoch_start = time.time()

        train_loss, train_acc, images_processed = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        throughput = images_processed / epoch_time  # images/second

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s | Throughput: {throughput:.1f} img/s")

        # Write to CSV
        csv_writer.writerow([epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}",
                             f"{epoch_time:.2f}", f"{throughput:.1f}",
                             args.model, args.experiment_tag, args.sm_allocation,
                             str(device), hostname])
        csv_file.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  >> New best model (Acc: {best_acc:.4f})")

    total_time = time.time() - total_start
    avg_throughput = (num_train_images * args.epochs) / total_time

    csv_file.close()

    # Save best model
    model.load_state_dict(best_model_wts)
    model_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment_tag}_best.pth")
    torch.save(model.state_dict(), model_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Model:          {args.model}")
    print(f"Experiment:     {args.experiment_tag}")
    print(f"SM Allocation:  {args.sm_allocation}")
    print(f"Epochs:         {args.epochs}")
    print(f"Total Time:     {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Avg Throughput: {avg_throughput:.1f} img/s")
    print(f"Best Val Acc:   {best_acc:.4f}")
    print(f"Results CSV:    {csv_path}")
    print(f"Model saved:    {model_path}")


if __name__ == "__main__":
    main()
