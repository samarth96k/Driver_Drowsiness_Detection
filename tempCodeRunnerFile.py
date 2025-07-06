import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class DrowsinessDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Process images and labels
        for img_name in os.listdir(image_dir):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                # Get corresponding label file
                label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        label_line = f.readline().strip().split()
                        if label_line:
                            # First digit indicates drowsiness state
                            label = int(label_line[0])
                            self.images.append(os.path.join(image_dir, img_name))
                            self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

class DrowsinessDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Use pre-trained ResNet as feature extractor
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_train = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss, correct_val = 0.0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {100 * correct_train/len(train_loader.dataset):.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {100 * correct_val/len(val_loader.dataset):.2f}%")

def main():
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Paths for images and labels
    image_dir = r'C:\Users\Samarth Khandelwal\OneDrive\Documents\VIT\SEMESTER 4\PE1\GRAND FINALE DDS\model\test\images'
    label_dir = r'C:\Users\Samarth Khandelwal\OneDrive\Documents\VIT\SEMESTER 4\PE1\GRAND FINALE DDS\model\test\labels'
    
    # Create datasets
    full_dataset = DrowsinessDataset(image_dir, label_dir, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = DrowsinessDetectionModel()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer)
    
    # Save the model
    torch.save(model.state_dict(), 'drowsiness_detection_model.pth')

if __name__ == '__main__':
    main()