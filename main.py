import cv2
import time
import json
from ultralytics import YOLO

FRAME_SKIP = 5
frame_count = 0
last_results = []

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",

    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",

    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",

    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",

    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",

    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",

    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",

    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",

    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",

    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",

    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}
def analyze_scene(detected_objects):
    analysis = {}
    for obj in detected_objects:
        label = obj["label"]
        if label not in analysis:
            analysis[label] = 0
        analysis[label] += 1
    return analysis
def save_scene(detected_objects, analysis):
    scene_data = {
        "timestamp": int(time.time()),
        "objects": detected_objects,
        "analysis": analysis
    }

    try:
        with open("scene_log.json", "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(scene_data)

    with open("scene_log.json", "w") as f:
        json.dump(data, f, indent=4)

last_save = 0
SAVE_INTERVAL = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        last_results = model(frame, conf=0.4, device="cpu", verbose=False)

    results = last_results


    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in COCO_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detected_objects.append({
                "label": COCO_CLASSES[cls],
                "confidence": float(box.conf[0]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, COCO_CLASSES[cls], (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    analysis = analyze_scene(detected_objects)

    now = time.time()
    if now - last_save > SAVE_INTERVAL:
        save_scene(detected_objects, analysis)
        last_save = now

    cv2.imshow("AI Perception", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


