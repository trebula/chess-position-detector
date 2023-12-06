import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 13  # 12 class + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    params = [p for p in model.parameters() if p.requires_grad]

    if torch.cuda.is_available():
        model.load_state_dict(torch.load("models/best_model_ep5_s100_f11.0.pth"))
    else:
        model.load_state_dict(torch.load("models/best_model_ep5_s100_f11.0.pth", map_location=torch.device("cpu")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def get_position(boxes):
    # boxes is a 4*n 2d array that marks the 4 corners of the target box
    # return the bottom position, which is decided by the center of lower 1/3 of the box
    n = len(boxes)
    position = np.zeros((n, 2))
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        position[i][0] = (x1 + x2) / 2
        position[i][1] = y2 - (y2 - y1) / 6
    return position


def predict(model, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    image_tensor = image_tensor.to(device)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor)
        boxes = prediction[0]["boxes"].cpu().numpy()
        labels = prediction[0]["labels"].cpu().numpy()
        scores = prediction[0]["scores"].cpu().numpy()
        threshold = 0.5
        boxes = boxes[scores >= threshold]
        labels = labels[scores >= threshold]
        positions = get_position(boxes)
    return positions, labels
