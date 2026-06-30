import cv2
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_PATH = "data/raw/test4.jpg"

REFERENCE_DIAMETER_MM = 23

MIN_CONTOUR_AREA = 1000

# ==========================================
# LOAD IMAGE
# ==========================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    return image

# ==========================================
# PREPROCESSING
# ==========================================

def preprocess(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    return gray, blur

# ==========================================
# THRESHOLDING
# ==========================================

def create_threshold(blur):

    thresh = cv2.adaptiveThreshold(

        blur,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY_INV,

        11,

        2

    )

    return thresh

# ==========================================
# MORPHOLOGY
# ==========================================

def clean_threshold(thresh):

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (5,5)

    )

    opened = cv2.morphologyEx(

        thresh,

        cv2.MORPH_OPEN,

        kernel

    )

    cleaned = cv2.morphologyEx(

        opened,

        cv2.MORPH_CLOSE,

        kernel

    )

    return cleaned

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(image, cleaned):

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []

    image_height, image_width = image.shape[:2]

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore small contours
        if area < MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore contours touching image border
        if (
            x <= 2 or
            y <= 2 or
            x + w >= image_width - 2 or
            y + h >= image_height - 2
        ):
            continue

        aspect_ratio = w / h

        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        M = cv2.moments(contour)

        if M["m00"] != 0:

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        else:

            cx = 0
            cy = 0

        objects.append({

            "contour": contour,

            "area": area,

            "perimeter": perimeter,

            "aspect_ratio": aspect_ratio,

            "circularity": circularity,

            "center": (cx, cy),

            "x": x,

            "y": y,

            "w": w,

            "h": h

        })

    return objects

# ==========================================
# DEBUG INFORMATION
# ==========================================

def display_features(output, objects):

    print("\n========== DETECTED OBJECTS ==========")

    for i, obj in enumerate(objects):

        print(f"\nObject {i+1}")

        print(f"Area          : {obj['area']:.2f}")

        print(f"Perimeter     : {obj['perimeter']:.2f}")

        print(f"Aspect Ratio  : {obj['aspect_ratio']:.2f}")

        print(f"Circularity   : {obj['circularity']:.3f}")

        print(f"Center        : {obj['center']}")

        print(f"Width         : {obj['w']}")

        print(f"Height        : {obj['h']}")

        cv2.rectangle(

            output,

            (obj["x"], obj["y"]),

            (obj["x"] + obj["w"],
             obj["y"] + obj["h"]),

            (0,255,255),

            2

        )

        cv2.circle(

            output,

            obj["center"],

            4,

            (0,0,255),

            -1

        )

        cv2.putText(

            output,

            str(i+1),

            (obj["x"], obj["y"]-10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255,0,0),

            2

        )

    return output

# ==========================================
# MAIN
# ==========================================

# ==========================================
# DETECT COIN
# ==========================================

def detect_coin(objects):

    if len(objects) == 0:
        return None

    best_score = -9999
    coin = None

    for obj in objects:

        # Circularity score (higher is better)
        circularity_score = obj["circularity"]

        # Aspect ratio score (closer to 1 is better)
        aspect_score = 1 - abs(1 - obj["aspect_ratio"])

        # Final Score
        score = circularity_score + aspect_score

        if score > best_score:

            best_score = score
            coin = obj

    return coin

# ==========================================
# SELECT TARGET OBJECT
# ==========================================

def select_target_object(objects, coin):

    remaining = []

    for obj in objects:

        if id(obj) != id(coin):

            remaining.append(obj)

    if len(remaining) == 0:
        return None

    target = max(

        remaining,

        key=lambda x: x["area"]

    )

    return target

# ==========================================
# SCALE CALCULATION
# ==========================================

def calculate_scale(coin):

    coin_pixels = max(

        coin["w"],

        coin["h"]

    )

    mm_per_pixel = REFERENCE_DIAMETER_MM / coin_pixels

    return mm_per_pixel, coin_pixels

# ==========================================
# OBJECT MEASUREMENT
# ==========================================

def measure_object(target, mm_per_pixel):

    length_mm = target["w"] * mm_per_pixel

    width_mm = target["h"] * mm_per_pixel

    area_mm = target["area"] * (mm_per_pixel ** 2)

    return length_mm, width_mm, area_mm

# ==========================================
# DISPLAY RESULTS
# ==========================================

def display_results(

    output,

    coin,

    target,

    length,

    width,

    area

):

    # Coin

    cv2.rectangle(

        output,

        (coin["x"], coin["y"]),

        (coin["x"] + coin["w"], coin["y"] + coin["h"]),

        (0,255,0),

        2

    )

    cv2.putText(

        output,

        "COIN",

        (coin["x"], coin["y"]-10),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0,255,0),

        2

    )

    # Target Object

    cv2.rectangle(

        output,

        (target["x"], target["y"]),

        (target["x"] + target["w"], target["y"] + target["h"]),

        (255,0,0),

        2

    )

    cv2.putText(

        output,

        "OBJECT",

        (target["x"], target["y"]-10),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255,0,0),

        2

    )

    print("\n========== GEMVISION RESULTS ==========\n")

    print(f"Length        : {length:.2f} mm")

    print(f"Width         : {width:.2f} mm")

    print(f"Area          : {area:.2f} mm²")

    print(f"Aspect Ratio  : {target['aspect_ratio']:.2f}")

    print(f"Circularity   : {target['circularity']:.3f}")

    cv2.imshow(

        "Final Result",

        output

    )
    
    
def main():

    image = load_image(IMAGE_PATH)

    gray, blur = preprocess(image)

    # Use Otsu + Morphology
    _, otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    cleaned = clean_threshold(otsu)

    output = image.copy()

    objects = extract_features(image, cleaned)

    display_features(output.copy(), objects)

    coin = detect_coin(objects)

    if coin is None:

        print("Coin not found!")

        return

    target = select_target_object(objects, coin)

    if target is None:

        print("Target Object not found!")

        return

    mm_per_pixel, coin_pixels = calculate_scale(coin)

    length, width, area = measure_object(

        target,

        mm_per_pixel

    )

    print(f"\nCoin Pixels : {coin_pixels}")

    print(f"Scale       : {mm_per_pixel:.4f} mm/pixel")

    display_results(

        output,

        coin,

        target,

        length,

        width,

        area

    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()