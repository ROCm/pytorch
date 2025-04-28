import torch
import torch.nn as nn


# Create a random input tensor
in_tensor = torch.randn(1, 3, 448, 448).to('cuda')

# Create a simple convolutional layer
conv_2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1).to('cuda')

# Perform convolution
output = conv_2d(in_tensor)

print(f"Input shape: {in_tensor.shape}")
print(f"Output shape: {output.shape}")
print("Convolution completed successfully")
