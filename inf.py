import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import CSRNet  # or wherever your CSRNet is defined

# ----- Load your trained model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet()
model = model.to(device)
model.eval()

# Optional: Load weights if you have saved them
model.load_state_dict(torch.load('weights.pth'))

# ----- Preprocess the input image -----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet mean
                         std=[0.229, 0.224, 0.225])   # standard ImageNet std
])

# Load image
img_path = r'people-counting\test\Lcd5gzEbi.jpg'
img = Image.open(img_path).convert('RGB')
img_transformed = transform(img).unsqueeze(0).to(device)  # add batch dimension

# ----- Make prediction -----
with torch.no_grad():
    output = model(img_transformed)
    output = output.squeeze().cpu().numpy()  # remove batch and channel dim

# ----- Postprocess and visualize -----
predicted_count = np.sum(output)
print(f"Predicted count: {predicted_count:.2f}")

# Optionally show the density map
plt.imshow(output, cmap='jet')
plt.title(f"Predicted Count: {predicted_count:.2f}")
plt.colorbar()
plt.show()
