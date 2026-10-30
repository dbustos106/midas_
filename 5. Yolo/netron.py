# python3 -m venv yolo-env
# source yolo-env/bin/activate
# pip install ultralytics onnx

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx")

print(model.model)
model.info()