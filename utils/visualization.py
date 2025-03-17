import cv2
import numpy as np

def visualize_results(image, detections, model):
    """
    Visualize object detection results.
    """
    image = image.squeeze(0).permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    for detection in detections.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Detection Results', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()