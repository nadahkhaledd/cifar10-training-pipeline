import time
import torch
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory // (1024**2)} MB, {props.multi_processor_count} SMs")

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
model.eval()

x = torch.randn(1, 3, 224, 224, device=device)

# warmup
with torch.no_grad():
    for _ in range(5):
        _ = model(x)

if device.type == "cuda":
    torch.cuda.synchronize()

start = time.time()

with torch.no_grad():
    for _ in range(20):
        y = model(x)

if device.type == "cuda":
    torch.cuda.synchronize()

end = time.time()

print("Output shape:", tuple(y.shape))
print("Total time:", round(end - start, 4), "sec")
print("Avg inference:", round((end - start) / 20 * 1000, 3), "ms")
