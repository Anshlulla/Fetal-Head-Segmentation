import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

# ---- CBAM Modules ----
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# ---- ResNet50_CBAM ----
from torchvision import models
class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_CBAM, self).__init__()
        base_model = models.resnet50(pretrained=True)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = nn.Sequential(base_model.layer1, CBAM(256))
        self.layer2 = nn.Sequential(base_model.layer2, CBAM(512))
        self.layer3 = nn.Sequential(base_model.layer3, CBAM(1024))
        self.layer4 = nn.Sequential(base_model.layer4, CBAM(2048))

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ---- Class Names ----
class_names = [
    "anold-chiari-malformation", "arachnoid-cyst", "cerebellah-hypoplasia", "colphocephaly",
    "encephalocele", "holoprosencephaly", "hydracenphaly", "intracranial-hemorrdge",
    "intracranial-tumor", "m-magna", "mild-ventriculomegaly", "moderate-ventriculomegaly",
    "normal", "polencephaly", "severe-ventriculomegaly", "vein-of-galen"
]

# ---- Image Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- Grad-CAM Prediction ----
def gradcam_predict(img_path, model_path="models/cbam_model_Resnet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ResNet50_CBAM(num_classes=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Grad-CAM hooks
    gradients = None
    activations = None

    def save_gradient(grad):
        nonlocal gradients
        gradients = grad

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        output.register_hook(save_gradient)

    # Register forward hook
    target_layer = model.layer4[0][0]  # Access ResNet's original layer inside the Sequential
    hook_handle = target_layer.register_forward_hook(forward_hook)

    # Preprocess image
    original_image = Image.open(img_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward for Grad-CAM
    model.zero_grad()
    output[0, pred_class].backward()

    # Grad-CAM generation
    grads = gradients.detach().cpu().numpy()[0]
    acts = activations.detach().cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam -= cam.min()
    cam /= cam.max()
    cam = cv2.resize(cam, (224, 224))

    # Overlay
    original_image_cv = np.array(original_image.resize((224, 224)))
    gray = cv2.cvtColor(original_image_cv, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    overlay = cv2.addWeighted(gray_3ch.astype(np.float32) / 255, 0.6, heatmap, 0.4, 0)
    overlay_img = (overlay * 255).astype(np.uint8)

    os.makedirs("static", exist_ok=True)
    overlay_path = "static/gradcam_overlay.png"
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    hook_handle.remove()

    return overlay_path, class_names[pred_class]
