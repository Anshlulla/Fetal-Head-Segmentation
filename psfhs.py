import os
import torch
import numpy as np
import cv2
import SimpleITK as sitk
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define preprocessing
def get_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2()
    ])

# Load model
def load_model(weights_path):
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Predict mask from image path
def predict_mask(image_path, model):
    transform = get_transform()

    # Read .mha image
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    if image.ndim == 3:
        image = image[image.shape[0] // 2]  # Take middle slice if 3D
    elif image.ndim != 2:
        raise ValueError("Unsupported image shape: expected 2D or 3D image.")

    image = np.stack([image] * 3, axis=-1)  # Convert grayscale to 3-channel

    original = cv2.resize(image, (256, 256))
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        pred_mask = torch.sigmoid(model(image_tensor))
        pred_mask = (pred_mask > 0.5).float()

    return original, pred_mask[0][0].cpu().numpy()

# Segment and prepare image for visualization
weights_path = "models/psfhs.pth"

def segment_with_psfhs(image_path, weights_path=weights_path, save_path=None):
    model = load_model(weights_path)
    original, mask = predict_mask(image_path, model)

    # Post-process mask
    mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Convert to BGR if needed
    if original.ndim == 2 or original.shape[-1] != 3:
        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        overlay = original.copy()

    # Initialize outputs
    head_circumference = None
    bpd = None
    aop = None
    hpd = None
    head_position = "Unknown"
    head_rotation = "Unknown"
    delivery_type = "Unknown"

    if len(contours) >= 2:
        c1, c2 = contours
        area1 = cv2.contourArea(c1)
        area2 = cv2.contourArea(c2)

        ps_contour = c1 if area1 < area2 else c2
        fh_contour = c2 if area1 < area2 else c1

        # Draw contours and labels
        cv2.drawContours(overlay, [fh_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(overlay, [ps_contour], -1, (0, 0, 255), 2)
        cv2.putText(overlay, 'FH', tuple(fh_contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, 'PS', tuple(ps_contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # === Measurements ===
        ellipse = cv2.fitEllipse(fh_contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        bpd = minor_axis
        head_circumference = np.pi * (3 * (major_axis + minor_axis) - np.sqrt((3 * major_axis + minor_axis) * (major_axis + 3 * minor_axis))) / 2

        # Get PS axis line: line between top and bottom points of pubic symphysis contour
        ps_pts = ps_contour[:, 0, :]
        ps_top = tuple(ps_pts[ps_pts[:, 1].argmin()])
        ps_bottom = tuple(ps_pts[ps_pts[:, 1].argmax()])

        # Compute PS axis vector
        ps_axis = np.array(ps_bottom) - np.array(ps_top)
        ps_axis_unit = ps_axis / np.linalg.norm(ps_axis)

        # Get FH center from ellipse fit
        fh_center = np.array(center)

        # Vector from PS bottom (inferior point) to FH center
        vec_to_fh = fh_center - np.array(ps_bottom)
        vec_to_fh_unit = vec_to_fh / np.linalg.norm(vec_to_fh)

        # === Angle of Progression (AoP) ===
        cos_angle = np.clip(np.dot(ps_axis_unit, vec_to_fh_unit), -1.0, 1.0)
        aop = np.degrees(np.arccos(cos_angle))

        # === Head–Pelvis Distance (HPD) ===
        # Distance from FH center to the PS axis line
        ps_line_vec = np.array(ps_bottom) - np.array(ps_top)
        ps_line_unit = ps_line_vec / np.linalg.norm(ps_line_vec)
        vec_fh_top = fh_center - np.array(ps_top)
        proj_len = np.dot(vec_fh_top, ps_line_unit)
        proj_point = np.array(ps_top) + proj_len * ps_line_unit
        hpd = np.linalg.norm(fh_center - proj_point)

        # === Determine Head Position ===
        if aop < 90 and hpd > 25:
            head_position = "Floating"
        elif 90 <= aop <= 105 and 15 <= hpd <= 25:
            head_position = "Engaged"
        elif 106 <= aop <= 120 and 5 <= hpd <= 14:
            head_position = "At Ischial Spines"
        elif aop > 120 and hpd < 5:
            head_position = "Deep Engagement"

        # === Determine Head Rotation ===
        if -30 <= angle <= 30:
            head_rotation = "Occiput Anterior (OA)"
        elif angle > 30:
            head_rotation = "Occiput Posterior (OP)"
        elif angle < -30:
            head_rotation = "Occiput Transverse (OT)"

        # === Delivery Type ===
        if head_position == "Deep Engagement" and head_rotation == "Occiput Anterior (OA)":
            delivery_type = "Likely Vaginal"
        elif head_position == "Floating" or head_rotation == "Occiput Posterior (OP)":
            delivery_type = "Cesarean Recommended"
        else:
            delivery_type = "Monitor Progress / Trial of Labor"

    else:
        print("Warning: Less than 2 contours found.")

    # Save image if needed
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg']:
            save_path += '.png'
        try:
            success = cv2.imwrite(save_path, overlay)
            if not success:
                raise ValueError(f"cv2.imwrite() failed. Check file path or format: {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save image: {e}")

    return {
        "image": overlay,
        "head_circumference": head_circumference,
        "bpd": bpd,
        "aop": aop,
        "hpd": hpd,
        "head_position": head_position,
        "head_rotation": head_rotation,
        "delivery_type": delivery_type
    }
