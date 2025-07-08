from ultralytics import YOLO
from multiprocessing import freeze_support

def train():
    # Load your previously trained model for fine-tuning
    model = YOLO("runs/detect/custom_traffic_model14/weights/best.pt")

    # Fine-tune on the new + old combined dataset
    model.train(
        data="C:/Users/Anuraag/OneDrive/Documents/Desktop/ivp project/data/Combined_Traffic_Dataset/data.yaml",
        epochs=100,                      # Increase if needed
        imgsz=640,                      # Can try 512 if VRAM is tight
        batch=4,                        # Adjust based on your GPU
        name="traffic_model_finetuned",# Output folder
        resume=False                    # Start fresh from the current weights
    )

if __name__ == "__main__":
    freeze_support()
    train()
