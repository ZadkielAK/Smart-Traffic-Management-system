from ultralytics import YOLO
import cv2

# Load YOLO model (fine-tuned weights)
model = YOLO("runs/detect/traffic_model_finetuned4/weights/best.pt").to('cuda')

def detect(image_path):
    # Run inference
    results = model(image_path, device='cuda', imgsz=1280, augment=True)
    result = results[0]

    # Load original image
    image = cv2.imread(image_path)

    # Draw bounding boxes for confident predictions
    for box in result.boxes:
        conf = box.conf.item()
        if conf >= 0.6:
            cls_id = int(box.cls.item())
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show image
    cv2.imshow("Detected Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = input("Enter the full path to the image: ")
    detect(image_path)
