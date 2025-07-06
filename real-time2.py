import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision import models

class DrowsinessDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

def detect_drowsiness():
    # Load pre-trained model
    model = DrowsinessDetectionModel()
    # model.load_state_dict(torch.load('drowsiness_detection_model.pth'))
    model.load_state_dict(torch.load(r"C:\Users\Samarth Khandelwal\OneDrive\Documents\VIT\SEMESTER 4\PE1\GRAND FINALE DDS\model\drowsiness_detection_model.pth"))
    model.eval()

    # Transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Video capture
    cap = cv2.VideoCapture(0)
    
    # Drowsiness tracking
    drowsy_frames = 0
    max_drowsy_frames = 30  # Adjust based on desired sensitivity

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Crop face
            face_img = frame[y:y+h, x:x+w]
            
            # Prepare for model
            face_tensor = transform(face_img)
            face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
            
                  # Predict
            with torch.no_grad():
                outputs = model(face_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
                # 0 = awake, 1 = drowsy
                if predicted.item() == 1:
                    drowsy_frames += 1
                    cv2.putText(frame, 'DROWSY', (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    drowsy_frames = max(0, drowsy_frames - 1)
                    cv2.putText(frame, 'AWAKE', (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Alert for prolonged drowsiness
        if drowsy_frames > max_drowsy_frames:
            cv2.putText(frame, 'DROWSINESS ALERT!', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
        # Display
        cv2.imshow('Drowsiness Detection', frame)
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_drowsiness()