import cv2

# ==================================
# CONFIGURATION
# ==================================

IMAGE_PATH = "data/raw/test4.jpg"
REFERENCE_DIAMETER_MM = 23  # ₹5 coin

# ==================================
# LOAD IMAGE
# ==================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    exit()

output = image.copy()

# ==================================
# PREPROCESSING
# ==================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# ==================================
# FIND CONTOURS
# ==================================

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

objects = []

for contour in contours:

    area = cv2.contourArea(contour)

    # Ignore small noise
    if area > 1000:

        x, y, w, h = cv2.boundingRect(contour)

        objects.append({
            "contour": contour,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })
print("Total Objects:", len(objects))

# ==================================
# DEBUG OUTPUT
# ==================================

print("\n===== DETECTED OBJECTS =====")

for i, obj in enumerate(objects):

    print(f"\nObject {i+1}")
    print(f"Area   : {obj['area']:.2f}")
    print(f"Width  : {obj['w']}")
    print(f"Height : {obj['h']}")

# ==================================
# CHECK DETECTIONS
# ==================================

if len(objects) < 2:
    print("\nCould not detect both coin and object!")
    cv2.imshow("Threshold", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# ==================================
# FIND COIN
# Closest to square shape
# ==================================

coin = None
best_score = 999999

for obj in objects:

    ratio = obj["w"] / obj["h"]

    score = abs(1 - ratio)

    if score < best_score:

        best_score = score
        coin = obj

# ==================================
# FIND MAIN OBJECT
# Largest non-coin contour
# ==================================

remaining_objects = []

for obj in objects:

    if id(obj) != id(coin):
        remaining_objects.append(obj)

target_object = max(
    remaining_objects,
    key=lambda x: x["area"]
)

# ==================================
# SCALE CALCULATION
# ==================================

coin_pixels = max(
    coin["w"],
    coin["h"]
)

mm_per_pixel = (
    REFERENCE_DIAMETER_MM /
    coin_pixels
)

# ==================================
# OBJECT MEASUREMENTS
# ==================================

object_length_mm = (
    target_object["w"] *
    mm_per_pixel
)

object_width_mm = (
    target_object["h"] *
    mm_per_pixel
)

# ==================================
# DRAW COIN
# ==================================

cv2.rectangle(
    output,
    (coin["x"], coin["y"]),
    (coin["x"] + coin["w"],
     coin["y"] + coin["h"]),
    (0, 255, 0),
    2
)

cv2.putText(
    output,
    "COIN",
    (coin["x"], coin["y"] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

# ==================================
# DRAW OBJECT
# ==================================

cv2.rectangle(
    output,
    (target_object["x"], target_object["y"]),
    (target_object["x"] + target_object["w"],
     target_object["y"] + target_object["h"]),
    (255, 0, 0),
    2
)

cv2.putText(
    output,
    "OBJECT",
    (target_object["x"], target_object["y"] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 0),
    2
)

# ==================================
# RESULTS
# ==================================

print("\n===== SCALE INFORMATION =====")

print(
    f"Coin Width (pixels): "
    f"{coin_pixels}"
)

print(
    f"Scale: "
    f"{mm_per_pixel:.4f} mm/pixel"
)

print("\n===== OBJECT MEASUREMENTS =====")

print(
    f"Length : "
    f"{object_length_mm:.2f} mm"
)

print(
    f"Width  : "
    f"{object_width_mm:.2f} mm"
)

# ==================================
# DISPLAY
# ==================================

cv2.imshow(
    "Threshold",
    thresh
)

cv2.imshow(
    "Measurement Result",
    output
)

cv2.waitKey(0)
cv2.destroyAllWindows()