import cv2
import cvzone

cap = cv2.VideoCapture(0)
casCadePath = "filter/haarcascade_frontalface_default.xml"
casCade = cv2.CascadeClassifier(casCadePath)

# List of overlay image filenames
overlay_filenames = [
    "filter/hair.png",
    "filter/sunglass.png",
    "filter/ears.png",
    "filter/cool.png",
    
    "filter/native.png",
    "filter/pirate.png"
]

# Load overlay images and check if they are loaded correctly
overlays = []
for filename in overlay_filenames:
    overlay = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        print(f"Error: Could not load image '{filename}'")
    overlays.append(overlay)

# Filter out None values from overlays list
overlays = [overlay for overlay in overlays if overlay is not None]

if not overlays:
    print("Error: No valid overlays found.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Initialize overlay index
overlay_index = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = casCade.detectMultiScale(grayScale)

    for (x, y, w, h) in faces:
        # Resize and overlay the current overlay
        overLayReSize = cv2.resize(overlays[overlay_index], (int(w * 1.5), int(h * 1.5)))
        frame = cvzone.overlayPNG(frame, overLayReSize, [x - 45, y - 75])

    cv2.imshow("Filters", frame)

    key = cv2.waitKey(10)
    if key == ord("s"):
        break
    elif key == 13:  # Enter key
        overlay_index = (overlay_index + 1) % len(overlays)  # Cycle to the next overlay

cap.release()
cv2.destroyAllWindows()
