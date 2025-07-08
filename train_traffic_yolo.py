from ultralytics import YOLO
from multiprocessing import freeze_support

def train():
    model = YOLO("yolov8m.pt")
    model.train(
        data="C:/Users/Anuraag/OneDrive/Documents/Desktop/ivp project/data/Dataset/Traffic_Dataset/Data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name="custom_traffic_model"
    )

if __name__ == "__main__":
    freeze_support()
    train()
