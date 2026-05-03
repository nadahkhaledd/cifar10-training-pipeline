import argparse
import csv
import os
import socket
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model_utils import SUPPORTED_MODELS, get_model
from train import dataset_exists, export_cifar10_test_folder


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_checkpoint(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        return checkpoint, checkpoint["model_state_dict"]

    return {
        "image_size": None,
        "best_val_acc": None,
        "model_name": None,
        "experiment_tag": None,
        "sm_allocation": None,
    }, checkpoint


@torch.no_grad()
def evaluate_test_split(model, loader, criterion, device):
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

    return running_loss / total, running_correct / total, total


@torch.no_grad()
def predict_image(model, image_path: str, transform, device, top_k: int = 3):
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

    values, indices = torch.topk(
        probabilities,
        k=min(top_k, len(CIFAR10_CLASSES)),
    )

    return [
        (CIFAR10_CLASSES[idx.item()], float(value.item()))
        for value, idx in zip(values, indices)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Test CIFAR-10 checkpoints or run single-image inference"
    )

    parser.add_argument("--model", type=str, default="resnet18", choices=SUPPORTED_MODELS)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=64)

    parser.add_argument("--use_subset", action="store_true")
    parser.add_argument("--test_subset_size", type=int, default=1000)

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional single image path for direct inference.",
    )

    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument(
        "--create_test_folder",
        action="store_true",
        help="Create data_dir/cifar10_test_images if it does not already exist.",
    )

    parser.add_argument("--experiment_tag", type=str, default="test")
    parser.add_argument("--sm_allocation", type=str, default="unknown")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
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

    checkpoint_path = args.checkpoint or os.path.join(
        args.checkpoint_dir,
        f"{args.model}_best.pth",
    )

    checkpoint = {
        "image_size": args.image_size,
        "best_val_acc": None,
        "model_name": args.model,
        "experiment_tag": None,
        "sm_allocation": None,
    }

    state_dict = None
    using_checkpoint = False

    if os.path.exists(checkpoint_path):
        checkpoint, state_dict = load_checkpoint(checkpoint_path, device)
        using_checkpoint = True
    else:
        print(f"Warning: checkpoint not found: {checkpoint_path}")
        print("Using model initial weights instead.")

    checkpoint_model = checkpoint.get("model_name")

    if checkpoint_model and checkpoint_model != args.model:
        print(
            f"Warning: checkpoint model_name is '{checkpoint_model}' "
            f"but --model is '{args.model}'. Using --model architecture."
        )

    image_size = int(checkpoint.get("image_size") or args.image_size)
    transform = build_transform(image_size)

    model = get_model(
        args.model,
        num_classes=10,
        pretrained=False,
    ).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    print(f"Model: {args.model}")

    if using_checkpoint:
        print(f"Checkpoint: {checkpoint_path}")
    else:
        print("Checkpoint: None - initial model weights")

    if checkpoint.get("best_val_acc") is not None:
        print(f"Checkpoint best val acc: {float(checkpoint['best_val_acc']):.4f}")

    if checkpoint.get("experiment_tag") is not None:
        print(f"Checkpoint experiment: {checkpoint.get('experiment_tag')}")

    if checkpoint.get("sm_allocation") is not None:
        print(f"Checkpoint SM allocation: {checkpoint.get('sm_allocation')}")

    if args.create_test_folder:
        export_cifar10_test_folder(args.data_dir)

    if args.image:
        start = time.time()

        predictions = predict_image(
            model=model,
            image_path=args.image,
            transform=transform,
            device=device,
            top_k=args.top_k,
        )

        inference_time = time.time() - start

        print(f"\nImage: {args.image}")
        print("Top predictions:")

        for label, prob in predictions:
            print(f"  {label}: {prob:.4f}")

        print(f"Inference time: {inference_time:.4f}s")
        return

    download_flag = not dataset_exists(args.data_dir)

    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=download_flag,
        transform=transform,
    )

    if args.use_subset:
        test_dataset = Subset(
            test_dataset,
            range(min(args.test_subset_size, len(test_dataset))),
        )
        print(f"Using test subset: {len(test_dataset)} images")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    start = time.time()

    test_loss, test_acc, total = evaluate_test_split(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    eval_time = time.time() - start

    print("\nEvaluation summary")
    print(f"Test images:    {total}")
    print(f"Test loss:      {test_loss:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Eval time:      {eval_time:.2f}s")

    csv_path = Path(args.output_dir) / f"eval_{args.model}_{args.experiment_tag}.csv"
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        if write_header:
            writer.writerow(
                [
                    "model",
                    "checkpoint",
                    "test_loss",
                    "test_acc",
                    "total_images",
                    "eval_time_sec",
                    "device",
                    "hostname",
                    "experiment_tag",
                    "sm_allocation",
                    "checkpoint_experiment_tag",
                    "checkpoint_sm_allocation",
                ]
            )

        writer.writerow(
            [
                args.model,
                checkpoint_path if using_checkpoint else "initial_weights",
                f"{test_loss:.4f}",
                f"{test_acc:.4f}",
                total,
                f"{eval_time:.2f}",
                str(device),
                hostname,
                args.experiment_tag,
                args.sm_allocation,
                checkpoint.get("experiment_tag"),
                checkpoint.get("sm_allocation"),
            ]
        )

    print(f"Eval CSV:       {csv_path}")


if __name__ == "__main__":
    main()
