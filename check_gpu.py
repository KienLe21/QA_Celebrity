import torch
print(torch.__version__)  # Kiểm tra phiên bản PyTorch
print(torch.cuda.is_available())  # True nếu PyTorch nhận GPU
print(torch.version.cuda)  # Kiểm tra phiên bản CUDA
print(torch.backends.cudnn.version())  # Kiểm tra cuDNN version
