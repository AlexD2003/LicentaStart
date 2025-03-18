import torch
import matplotlib.pyplot as plt

# Load a tensor image
tensor_img = torch.load("tensor_dataset/mdb001.pt")  # Load a saved tensor

# Convert to NumPy for visualization
plt.imshow(tensor_img.squeeze(), cmap="gray")  # Remove channel dimension
plt.axis("off")
plt.show()
