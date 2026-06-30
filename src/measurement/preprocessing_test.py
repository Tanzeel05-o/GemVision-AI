import cv2
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_PATH = "data/raw/test5.jpg"
MIN_CONTOUR_AREA = 1000

# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ==========================================
# METHOD 1
# ADAPTIVE THRESHOLD
# ==========================================

adaptive = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# ==========================================
# METHOD 2
# OTSU THRESHOLD
# ==========================================

_, otsu = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# ==========================================
# MORPHOLOGY KERNEL
# ==========================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

# ==========================================
# METHOD 3
# ADAPTIVE + MORPHOLOGY
# ==========================================

adaptive_clean = cv2.morphologyEx(
    adaptive,
    cv2.MORPH_OPEN,
    kernel
)

adaptive_clean = cv2.morphologyEx(
    adaptive_clean,
    cv2.MORPH_CLOSE,
    kernel
)

# ==========================================
# METHOD 4
# OTSU + MORPHOLOGY
# ==========================================

otsu_clean = cv2.morphologyEx(
    otsu,
    cv2.MORPH_OPEN,
    kernel
)

otsu_clean = cv2.morphologyEx(
    otsu_clean,
    cv2.MORPH_CLOSE,
    kernel
)

# ==========================================
# METHOD 5
# CANNY EDGE
# ==========================================

canny = cv2.Canny(
    blur,
    50,
    150
)

# ==========================================
# FUNCTION TO COUNT OBJECTS
# ==========================================

def count_objects(binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    count = 0

    output = image.copy()

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > MIN_CONTOUR_AREA:

            count += 1

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                str(count),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    return count, output

# ==========================================
# COUNT OBJECTS
# ==========================================

adaptive_count, adaptive_box = count_objects(adaptive)

otsu_count, otsu_box = count_objects(otsu)

adaptive_clean_count, adaptive_clean_box = count_objects(adaptive_clean)

otsu_clean_count, otsu_clean_box = count_objects(otsu_clean)

canny_count, canny_box = count_objects(canny)

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n========== PREPROCESSING COMPARISON ==========\n")

print(f"Adaptive Threshold          : {adaptive_count} Objects")

print(f"Otsu Threshold              : {otsu_count} Objects")

print(f"Adaptive + Morphology       : {adaptive_clean_count} Objects")

print(f"Otsu + Morphology           : {otsu_clean_count} Objects")

print(f"Canny Edge                 : {canny_count} Objects")

# ==========================================
# DISPLAY RESULTS
# ==========================================

cv2.imshow("Original Image", image)

cv2.imshow("Adaptive Threshold", adaptive_box)

cv2.imshow("Otsu Threshold", otsu_box)

cv2.imshow("Adaptive + Morphology", adaptive_clean_box)

cv2.imshow("Otsu + Morphology", otsu_clean_box)

cv2.imshow("Canny Edge", canny_box)

cv2.waitKey(0)

cv2.destroyAllWindows()