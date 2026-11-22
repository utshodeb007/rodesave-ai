
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# Paths
# ==========================================
script_folder = os.path.dirname(
    os.path.abspath(__file__)
)
project_root = os.path.dirname(script_folder)

train_path = os.path.join(
    project_root,
    "Data",
    "accident detection",
    "prepared_dataset",
    "train"
)

test_path = os.path.join(
    project_root,
    "Data",
    "accident detection",
    "prepared_dataset",
    "test"
)

model_path = os.path.join(
    project_root,
    "Data",
    "accident detection",
    "accident_model.pth"
)

# ==========================================
# Device
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS)")

elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU")

else:
    device = torch.device("cpu")
    print("Using CPU")

# ==========================================
# Image Transform
# ==========================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ==========================================
# Load Dataset
# ==========================================
train_dataset = datasets.ImageFolder(
    train_path,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    test_path,
    transform=transform
)

print("\nClasses:")
print(train_dataset.classes)

print("\nTraining images:", len(train_dataset))
print("Testing images:", len(test_dataset))

# ==========================================
# Data Loaders
# ==========================================
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("\nTraining batches:", len(train_loader))
print("Testing batches:", len(test_loader))

# ==========================================
# CNN Model
# ==========================================
class AccidentCNN(nn.Module):

    def __init__(self):
        super(AccidentCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128 * 16 * 16,
                128
            ),

            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==========================================
# Create Model
# ==========================================
model = AccidentCNN()
model = model.to(device)

# ==========================================
# Loss & Optimizer
# ==========================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==========================================
# Training
# ==========================================
epochs = 10

print("\n================================")
print("Starting Model Training")
print("================================")

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Prediction
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy
        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        100 * correct / total
    )

    average_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {train_accuracy:.2f}%"
    )

# ==========================================
# Test Model
# ==========================================
print("\n================================")
print("Testing Model")
print("================================")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

test_accuracy = (
    100 * correct / total
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)

# ==========================================
# Save Model
# ==========================================
torch.save(
    model.state_dict(),
    model_path
)

print("\n================================")
print("MODEL TRAINING COMPLETED!")
print("================================")

print("Model saved at:")
print(model_path)
