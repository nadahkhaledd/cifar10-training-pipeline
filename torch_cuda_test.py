import torch
import time

device = torch.device("cuda")

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))

# simple tensor test
x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)

torch.cuda.synchronize()

start = time.time()

for _ in range(50):
    z = x @ y

torch.cuda.synchronize()

end = time.time()

print("Result shape:", z.shape)
print("Total time:", end - start)
print("Avg matmul:", (end - start)/50)
