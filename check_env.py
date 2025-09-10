import torch, shutil, sys, os

print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", getattr(torch.version, "cuda", None))
print("Torch ver:", torch.__version__)
print("ffmpeg present:", shutil.which("ffmpeg") is not None)
print("Writeable ./outputs:", os.access("./outputs", os.W_OK) or not os.path.exists("./outputs"))
