# Driver_Drowsiness_Detection

# PROBLEM DEFINITION 
 
The facial expression recognition method gives very good interpretation about human behaviour and emotion. Its full potential goes beyond emotion classification; it can be used to predict critical states of behaviour, like drowsiness, preventing mishaps and safety. One of the most relevant applications is drowsiness detection, especially in dangerous scenarios like driving or prolonged work sessions. Drowsiness is 0indeed one of the serious threats to not only personal wellbeing but also public safety and productivity of the workplace. Mistakes by an overfatigued worker in industrial environments may mean disaster, and drowsinessrelated road accidents are very common. 

# Hardware and System Configuration 
The drowsiness detection system is implemented on a Dell G15 5520 laptop, which provides robust 
computational capabilities with its 16 GB RAM, 512 GB SSD, and NVIDIA RTX 3050 GPU. These 
specifications are ideal for handling real-time video streams and performing deep learning model 
inference efficiently. To capture facial features, a Jio 1080p HD external webcam is employed, 
delivering high-definition video input. This hardware setup ensures smooth processing of video 
frames and accurate detection, even under varying lighting conditions or when users wear glasses or 
masks. 
# Software Environment 
The software stack for this project includes Python as the core programming language, supported by 
powerful libraries such as PyTorch for building and training the machine learning model, and 
OpenCV for real-time video processing. Development is carried out using VS Code and Jupyter 
Notebook, offering flexibility and interactivity during model training and testing. Annotation tools 
like LabelImg and YOLO Annotator are used to label the datasets, a critical step for training 
supervised models. GitHub is employed for version control, ensuring the project remains organized 
and collaborative. 
# Dataset Preparation 
Two datasets form the foundation for training the model. The first is sourced from Kaggle, containing 
labeled examples of drowsy and awake states, while the second is a custom dataset created using 
photos of the project team and friends. The custom dataset incorporates diverse conditions, such as 
variations in lighting, angles, and occlusions like glasses or masks. Images are annotated using 
LabelImg and YOLO Annotator to identify key facial landmarks, including eyes and mouth, enabling 
robust training and evaluation. 
# Model Development 
The model architecture is built using a Vision Transformer (ViT) fine-tuned for binary classification 
(drowsy vs. awake). The ViT leverages its powerful feature extraction capabilities, making it ideal for 
analyzing subtle changes in facial expressions. The classification head is replaced to adapt the model 
for this specific task. Training involves the use of the AdamW optimizer and a CosineAnnealingLR 
scheduler to dynamically adjust the learning rate. Loss is calculated using CrossEntropyLoss, 
ensuring accurate classification. Data augmentation techniques, such as random cropping and 
flipping, enhance the model's robustness to real-world scenarios. 
# Real-Time Processing 
For real-time implementation, the system captures video streams using OpenCV and processes frames 
sequentially. YOLOv5 is used to detect facial landmarks, focusing on regions of interest like the eyes 
and mouth. These detections are fed into the Vision Transformer for classification. The system 
continuously monitors Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to identify blinking 
patterns and yawns, providing heuristic checks for drowsiness. If a drowsy state is detected 
consistently across multiple frames, an alert mechanism triggers, including audio alarms or visual 
warnings. 

