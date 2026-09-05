"""
train.py — Basketball Shot Detector Training Script
=====================================================
Trains a YOLOv8 model to detect:
  - Class 0: Basketball
  - Class 1: Basketball Hoop

Steps before running:
  1. Download a labeled dataset from Roboflow 
  2. Extract it so you have  train/, valid/, test/  folders
  3. Update config.yaml with the absolute paths to those folders
  4. Copy best.pt to the project root and run app.py or shot_detector.py
"""

from pathlib import Path
from ultralytics import YOLO
from utils import get_device

TRAIN_CONFIG = {
  
    "data": "config.yaml",          

    "model": "yolov8n.pt",
    "epochs":   100,     
    "imgsz":    640,     
    "batch":    16,      
    "lr0":      0.01,    
    "lrf":      0.01,    
    "weight_decay": 0.0005,  # L2 regularisation — prevents overfitting
    "dropout":      0.0,     
    "optimizer": "auto",

   
    "augment":  True,    # Enable built-in augmentation pipeline
    "degrees":  5.0,     
    "translate": 0.1,    
    "scale":    0.5,    
    "fliplr":   0.5,     
    "mosaic":   1.0,     
    "hsv_h":    0.015,   
    "hsv_s":    0.7,     
    "hsv_v":    0.4,     

    
    "project":  "runs/detect",  
    "name":     "train",        
    "save":     True,          
    "exist_ok": False,          

  
    "patience": 20,    
    "workers":  8,      
    "verbose":  True,   
    "plots":    True,   
}


if __name__ == "__main__":
    device = get_device()
    print(f"[train.py] Using device: {device.upper()}")

  
    cfg = Path(TRAIN_CONFIG["data"])
    if not cfg.exists():
        raise FileNotFoundError(f"Dataset config not found: {cfg.resolve()}")

    print(f"[train.py] Loading base model: {TRAIN_CONFIG['model']}")
    model = YOLO(TRAIN_CONFIG["model"])

    
    print("[train.py] Starting training…")
    results = model.train(device=device, **TRAIN_CONFIG)

    weights_dir = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"] / "weights"
    best_pt     = weights_dir / "best.pt"

    print("Training complete!")
    if best_pt.exists():
        print(f"  Best weights : {best_pt.resolve()}")
       
    else:
        print(f"  Weights saved to: {weights_dir.resolve()}")
   
