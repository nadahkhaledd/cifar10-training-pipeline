import io
import os

import torch
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

from model_utils import get_model


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class CIFARHandler(BaseHandler):
    def initialize(self, ctx):
        properties = ctx.system_properties

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = properties.get("model_dir")
        self.model_name = os.environ.get("MODEL_NAME", "resnet18")
        checkpoint_name = os.environ.get("CHECKPOINT_NAME", f"{self.model_name}_best.pth")
        checkpoint_path = os.path.join(model_dir, checkpoint_name)

        self.image_size = int(os.environ.get("IMAGE_SIZE", "64"))
        self.top_k = int(os.environ.get("TOP_K", "3"))

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.model = get_model(self.model_name, num_classes=10, pretrained=False)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.image_size = int(checkpoint.get("image_size", self.image_size))
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print(f"Warning: checkpoint not found: {checkpoint_path}. Using initial weights.")

        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            print(
                f"TorchServe handler initialized on cuda | "
                f"{props.name}, {props.multi_processor_count} SMs"
            )
        else:
            print("TorchServe handler initialized on CPU")

    def preprocess(self, requests):
        images = []

        for request in requests:
            data = request.get("data") or request.get("body")

            image = Image.open(io.BytesIO(data)).convert("RGB")
            tensor = self.transform(image)
            images.append(tensor)

        return torch.stack(images).to(self.device)

    def inference(self, inputs):
        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities

    def postprocess(self, outputs):
        results = []

        for probs in outputs:
            values, indices = torch.topk(
                probs,
                k=min(self.top_k, len(CIFAR10_CLASSES)),
            )

            results.append(
                [
                    {
                        "label": CIFAR10_CLASSES[idx.item()],
                        "probability": float(value.item()),
                    }
                    for value, idx in zip(values, indices)
                ]
            )

        return results
