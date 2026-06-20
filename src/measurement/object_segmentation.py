import cv2

image = cv2.imread("data/test4.jpg")

if image is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binary threshold
_, thresh = cv2.threshold(
    gray,
    180,
    255,
    cv2.THRESH_BINARY_INV
)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

output = image.copy()

for contour in contours:

    area = cv2.contourArea(contour)

    if area > 1000:

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            output,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        print(f"Width: {w} px")
        print(f"Height: {h} px")
        
cv2.imshow("Threshold", thresh)
cv2.imshow("Contours", output)

cv2.waitKey(0)
cv2.destroyAllWindows()