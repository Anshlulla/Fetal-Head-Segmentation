from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
from torchvision import transforms
from attention_unet_model import AttentionUNet
from torchvision.transforms.functional import to_tensor,to_pil_image
from PIL import Image
from skimage.segmentation import mark_boundaries
from resnet_cbam_model import ResNet50_CBAM 
from gradcam import gradcam_predict
from torchvision import models, transforms
import torch.nn as nn
from psfhs import segment_with_psfhs
import SimpleITK as sitk


UPLOAD_FOLDER = 'static/uploads/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mha'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/image-processing', methods=['GET', 'POST'])
def image_processing():
    if request.method == 'POST':
        # Handle file upload logic
        if 'ultrasound-image' not in request.files:
            return "No file part"

        file = request.files['ultrasound-image']
        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_path)

            result = segment_fetal_head(original_path)
            gestational_age = calculate_gestational_age(result["bpd"], result["hc"])

            segmented_filename = f"segmented_{filename}"
            segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_filename)
            cv2.imwrite(segmented_path, result["image"])

            relative_original = os.path.relpath(original_path, 'static').replace("\\", "/")
            relative_segmented = os.path.relpath(segmented_path, 'static').replace("\\", "/")

            return render_template('image_processing.html',
                                   selected_tab="image-processing",
                                   original_image=relative_original,
                                   segmented_image=relative_segmented,
                                   bpd=result["bpd"],
                                   ofd=result["ofd"],
                                   hc=result["hc"],
                                   ha=result["ha"],
                                   gestational_age=gestational_age)
        return "Invalid file type"
    
    # If it's GET request, just render the form
    return render_template('image_processing.html', selected_tab="image-processing")

@app.route('/deep-learning', methods=['GET', 'POST'])
def deep_learning():
    if request.method == 'POST':
        # Handle file upload logic
        if 'ultrasound-image' not in request.files:
            return "No file part"

        file = request.files['ultrasound-image']
        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_path)

            result = segment_with_attention_unet(original_path,model,device)
            gestational_age = calculate_gestational_age(result["bpd"], result["hc"])

            segmented_filename = f"segmented_{filename}"
            segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_filename)
            cv2.imwrite(segmented_path, result["image"])

            relative_original = os.path.relpath(original_path, 'static').replace("\\", "/")
            relative_segmented = os.path.relpath(segmented_path, 'static').replace("\\", "/")

            return render_template('deep_learning.html',
                                   selected_tab="image-processing",
                                   original_image=relative_original,
                                   segmented_image=relative_segmented,
                                   bpd=result["bpd"],
                                   ofd=result["ofd"],
                                   hc=result["hc"],
                                   ha=result["ha"],
                                   gestational_age=gestational_age)
        return "Invalid file type"
    
    # If it's GET request, just render the form
    return render_template('deep_learning.html', selected_tab="image-processing")
            


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def segment_fetal_head(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read the uploaded image.")

    denoised_img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(denoised_img)

    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    height, width = img.shape
    max_dynamic_radius = min(height, width) // 2

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=40, minRadius=25, maxRadius=max_dynamic_radius)

    output_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    bpd = ofd = hc = ha = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]

        bpd = round(2 * r * 0.8, 2)
        ofd = round(2 * r * 1.0, 2)
        ha = int(cv2.countNonZero(clahe_img))

        contours, _ = cv2.findContours(clahe_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hc = round(cv2.arcLength(contours[0], True), 2)

        cv2.circle(output_img, (x, y), r, (0, 255, 255), 2)
        cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)
    else:
        cv2.putText(output_img, "No Circle Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return {
        "image": output_img,
        "bpd": bpd,
        "ofd": ofd,
        "hc": hc,
        "ha": ha
    }

def calculate_gestational_age(bpd, hc, pixel_to_mm=0.264):
    bpd_mm = bpd * pixel_to_mm
    hc_mm = hc * pixel_to_mm

    ga_bpd = round((bpd_mm - 47.5) / 3.75, 1)
    ga_hc = round((hc_mm - 300) / 10, 1)

    gestational_age = round((ga_bpd + ga_hc) / 2, 1)
    return gestational_age

# Load model globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionUNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load("models/attention_unet_model.pt", map_location=device))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
from PIL import Image
import torch
from torchvision.transforms import ToTensor

def show_img_mask(img, mask):
    if torch.is_tensor(img):
        img = to_pil_image(img)
        mask = to_pil_image(mask)
        
    # Mark boundaries on the image with the mask
    img_mask = mark_boundaries(np.array(img), np.array(mask), outline_color=(0, 1, 0), color=(0, 1, 0))
    
    # Return the masked image (can be a NumPy array)
    return img_mask  # This is the image with the boundary mask

def segment_with_attention_unet(image_path, model, device, w=128, h=192):
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    original_size = img.size  # (width, height)
    img_resized = img.resize((w, h))
    img_t = ToTensor()(img_resized).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        pred = model(img_t).cpu()
    pred = torch.sigmoid(pred)[0]
    mask_pred = (pred[0] >= 0.5)

    # Convert to numpy and resize to original image size
    mask_pred_np = mask_pred.detach().cpu().numpy()
    mask_bin = (mask_pred_np * 255).astype(np.uint8)
    mask_bin_resized = cv2.resize(mask_bin, original_size, interpolation=cv2.INTER_NEAREST)

    # Load original image for overlay
    img_original = Image.open(image_path).convert('L')
    img_np = np.array(img_original)

    # Generate boundary overlay image on original size
    out_img = mark_boundaries(img_np, mask_bin_resized, color=(0, 1, 0))

    # Convert to uint8 and BGR for OpenCV drawing
    out_img_uint8 = (out_img * 255).astype(np.uint8)
    out_img_bgr = cv2.cvtColor(out_img_uint8, cv2.COLOR_RGB2BGR)

    # Show the image with mask using show_img_mask function
    show_img_mask(img_original, mask_bin_resized)

    # Biometric measurements
    bpd = ofd = hc = ha = 0
    contours, _ = cv2.findContours(mask_bin_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            bpd = round(min(MA, ma), 2)
            ofd = round(max(MA, ma), 2)
            hc = round(cv2.arcLength(cnt, True), 2)
            ha = round(cv2.contourArea(cnt), 2)

            # Optionally, draw the ellipse
            cv2.ellipse(out_img_bgr, ellipse, (0, 255, 255), 2)
    else:
        # Optionally, add text if no head is detected
        cv2.putText(out_img_bgr, "No head detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return {
        "image": out_img_bgr,
        "bpd": bpd,
        "ofd": ofd,
        "hc": hc,
        "ha": ha
    }

# Load classification model
classification_model = ResNet50_CBAM(num_classes=16).to(device) 
classification_model.load_state_dict(torch.load("models/cbam_model_Resnet.pth", map_location=device))
classification_model.eval()

def classify_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Ensure image is in RGB format
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)  # shape: (1, num_classes)
        probs = torch.nn.functional.softmax(outputs, dim=1)  # softmax to get confidence scores
        confidence, predicted = torch.max(probs, 1)

    return predicted.item(), confidence.item()  # Return both class index and confidence


@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        if 'ultrasound-image' not in request.files:
            return "No file part"

        file = request.files['ultrasound-image']
        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predicted_class_index, confidence_score = classify_image(filepath, classification_model, device)

            class_names = [
                "arnold-chiari-malformation",
                "arachnoid-cyst",
                "cerebellah-hypoplasia",
                "colphocephaly",
                "encephalocele",
                "holoprosencephaly",
                "hydracenphaly",
                "intracranial-hemorrdge",
                "intracranial-tumor",
                "m-magna",
                "mild-ventriculomegaly",
                "moderate-ventriculomegaly",
                "normal",
                "polencephaly",
                "severe-ventriculomegaly",
                "vein-of-galen"
            ]

            class_name = class_names[predicted_class_index]
            relative_image = os.path.relpath(filepath, 'static').replace("\\", "/")

            prediction = {
                'class_name': class_name,
                'confidence': round(confidence_score * 100, 2),  # confidence in %
                'image_path': f"{relative_image}"
            }

            return render_template('Abnormaility_Classification.html',
                                   selected_tab="classification",
                                   prediction=prediction)

        return "Invalid file type"
    
    return render_template('Abnormaility_Classification.html', selected_tab="classification")

@app.route('/gradcam', methods=['GET', 'POST'])
def gradcam():
    if request.method == 'POST':
        file = request.files.get('ultrasound-image')

        if not file or file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            overlay_path, predicted_class = gradcam_predict(filepath)

            # Relative paths from the 'static' folder
            relative_overlay_path = os.path.relpath(overlay_path, 'static').replace("\\", "/")
            relative_input_path = os.path.relpath(filepath, 'static').replace("\\", "/")

            return render_template('GRAD_CAM.html',
                                   selected_tab="gradcam",
                                   original_image=relative_input_path,
                                   gradcam_image=relative_overlay_path,
                                   prediction_label=predicted_class,
                                   prediction_score="N/A")  # Add score later if needed

        return "Invalid file type"

    return render_template('GRAD_CAM.html', selected_tab="gradcam")


@app.route('/psfhs', methods=['GET', 'POST'])
def psfhs():
    if request.method == 'POST':
        file = request.files.get('ultrasound-image')

        if not file or file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # === Convert .mha to displayable image ===
            try:
                image = sitk.GetArrayFromImage(sitk.ReadImage(filepath))
                if image.ndim == 3:
                    image = image[image.shape[0] // 2]  # Middle slice
                image = np.stack([image] * 3, axis=-1)  # Grayscale to 3-channel
                image = cv2.resize(image, (256, 256))  # Resize for consistency

                # Save original image as PNG
                base_name, _ = os.path.splitext(filename)
                original_display_filename = f"original_{base_name}.png"
                original_display_path = os.path.join(app.config['UPLOAD_FOLDER'], original_display_filename)
                cv2.imwrite(original_display_path, image)
            except Exception as e:
                return f"[ERROR] Failed to convert original image for display: {e}"

            # === Run PSFHS segmentation ===
            try:
                result = segment_with_psfhs(filepath)
                segmented_image = result["image"]

                segmented_filename = f"psfhs_segmented_{base_name}.png"
                segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_filename)
                success = cv2.imwrite(segmented_path, segmented_image)
                if not success:
                    raise ValueError("cv2.imwrite() failed. Check file path or format.")
            except Exception as e:
                return f"[ERROR] Could not save segmented image: {e}"

            # === Relative paths for HTML ===
            relative_original = os.path.relpath(original_display_path, 'static').replace("\\", "/")
            relative_segmented = os.path.relpath(segmented_path, 'static').replace("\\", "/")

            return render_template('PSFHS.html',
                                   selected_tab="psfhs",
                                   original_image=relative_original,
                                   segmented_image=relative_segmented,
                                   head_circumference=round(result["head_circumference"], 2) if result["head_circumference"] else None,
                                   bpd=round(result["bpd"], 2) if result["bpd"] else None,
                                   aop=round(result["aop"], 2) if result["aop"] else None,
                                   hpd=round(result["hpd"], 2) if result["hpd"] else None,
                                   head_position=result["head_position"],
                                   head_rotation=result["head_rotation"],
                                   delivery_type=result["delivery_type"]
                                   )

        return "Invalid file type"

    return render_template('PSFHS.html', selected_tab="psfhs")




if __name__ == '__main__':
    app.run(debug=True)

