import time
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda")

a = torch.randn(3000, 3000, device=device)
b = torch.randn(3000, 3000, device=device)

torch.cuda.synchronize()
start = time.time()

c = a @ b

torch.cuda.synchronize()
end = time.time()

print("Result shape:", c.shape)
print("Device:", c.device)
print("Time:", round(end - start, 4), "seconds")
print(
    "VRAM used:",
    round(torch.cuda.memory_allocated() / 1024**2, 1),
    "MB"
)