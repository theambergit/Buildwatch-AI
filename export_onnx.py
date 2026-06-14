"""One-time script to export YOLOv8n from PyTorch to ONNX format.

Usage:
    pip install ultralytics torch
    python export_onnx.py

This produces yolov8n.onnx which can be used without PyTorch installed.
"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, simplify=True)
print("Export complete: yolov8n.onnx")
