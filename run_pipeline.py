import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

# ------------------- CLAHE ------------------- not accurate
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_enhanced = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ------------------- Normalize -------------------
def normalize(image):
    return image / 255.0

# ------------------- Pupil Center Detection -------------------
def detect_pupil_center(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_blur = cv2.medianBlur(gray, 5)
    # DEBUGGING: 🟡 Apply CLAHE BEFORE detection for better contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 🔍 Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

   
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # 📏 Automatically scale min/max radius based on image size
    h, w = gray.shape
    minRadius = int(min(h, w) * 0.05)   # ~3% of image size
    maxRadius = int(min(h, w) * 0.3)   # ~15% of image size

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=50,
        param2=30,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    #if circles is not None:
        #circles = np.uint16(np.around(circles))
        #return circles[0][0]  # (x, y, r)
    
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        # Choose the circle closest to image center
        img_center = np.array([gray.shape[1]//2, gray.shape[0]//2])
        distances = [np.linalg.norm(np.array((x, y)) - img_center) for (x, y, r) in circles]
        best_circle = circles[np.argmin(distances)]
        return best_circle

    
    return None
    #circles = cv2.HoughCircles(
        #gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
       # param1=50, param2=30, minRadius=15, maxRadius=60
    #)
   # if circles is not None:
       # circles = np.uint16(np.around(circles))
        #return circles[0][0]  # x, y, r
    #return None

def get_pupil_center(image):
    result = detect_pupil_center(image)
    if result is None:
        raise ValueError("❌ Pupil not detected.")
    #x, y, r = result
    #return int(x), int(y)

    #debugging
    x, y, r = result
    # DEBUGGING: return int(x), int(y), int(r)

    # DEBUGGING
    # 🔄 Offset based on radius
    offset_x = int(0.0 * r)   # 50% right
    offset_y = int(0.0 * r)  # 0% down
    # Adjust center coordinates 
    # to account for the offset


    adjusted_x = int(x + offset_x)
    adjusted_y = int(y + offset_y)
    #debugging
    print(f"Original center: ({x}, {y}) → Adjusted: ({adjusted_x}, {adjusted_y}), Radius: {r}")


    return adjusted_x, adjusted_y, int(r)
    





# ------------------- Crop Around Pupil -------------------
def crop_to_pupil(image, center_x, center_y, r, scale=2.0):
    h, w = image.shape[:2]
    crop_size = int(r * scale)
    margin = crop_size // 2
    x1 = max(0, center_x - margin)
    y1 = max(0, center_y - margin)
    x2 = min(w, center_x + margin)
    y2 = min(h, center_y + margin)

    cropped = image[y1:y2, x1:x2]

    # Resize to 128x128 regardless of original crop size
    cropped_resized = cv2.resize(cropped, (128, 128))
    return cropped_resized


# ------------------- Full Pipeline -------------------
def full_pipeline(image):
    #center_x, center_y = get_pupil_center(image)
    #pupil_crop = crop_to_pupil(image, center_x, center_y)
    #enhanced = apply_clahe(pupil_crop)
    #normalized = normalize(enhanced)
    #return normalized  # Final preprocessed image (numpy array)
    
    # ------------------- DEBUGGING ---------------------
    center_x, center_y, r = get_pupil_center(image)
    # DEBUG: Visualize detected circle
    debug_image = image.copy()
    cv2.circle(debug_image, (center_x, center_y), 3, (0, 255, 0), -1)  # center dot
    cv2.circle(debug_image, (center_x, center_y), r, (255, 0, 0), 2)   # detected circle
    cv2.imwrite("debug_detected_pupil.jpg", debug_image)  # Save for inspection
    print("📸 Saved debug image as 'debug_detected_pupil.jpg'")

    pupil_crop = crop_to_pupil(image, center_x, center_y, r)
    enhanced = apply_clahe(pupil_crop)
    normalized = normalize(enhanced)

    return normalized

# ------------------- Main Run -------------------
if __name__ == "__main__":
    image = cv2.imread("test5_eye.jpg")
    if image is None:
        raise FileNotFoundError("❌ Image not found. Make sure 'test_eye.jpg' is in this folder.")

    result = full_pipeline(image)

    # Save result
    cv2.imwrite("output_result.jpg", (result * 255).astype("uint8"))
    print("✅ Saved result to output_result.jpg")
    
