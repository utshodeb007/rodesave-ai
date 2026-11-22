
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==============================
# PATH
# ==============================

script_folder = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_folder)

model_path = os.path.join(
    project_root,
    "Data",
    "accident detection",
    "accident_model.pth"
)

test_folder = os.path.join(
    project_root,
    "Data",
    "accident detection",
    "prepared_dataset",
    "test"
)

output_path = os.path.join(
    project_root,
    "Data",
    "accident detection",
    "confusion_matrix.png"
)

# ==============================
# DEVICE
# ==============================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS)")

elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU")

else:
    device = torch.device("cpu")
    print("Using CPU")

# ==============================
# CNN MODEL
# ==============================

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
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==============================
# LOAD MODEL
# ==============================

model = AccidentCNN()

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model = model.to(device)
model.eval()

print("\nModel loaded successfully!")

# ==============================
# IMAGE TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ==============================
# TEST DATASET
# ==============================

classes = ["accident", "non_accident"]

y_true = []
y_pred = []

total_images = 0

print("\nEvaluating test dataset...")

for class_index, class_name in enumerate(classes):

    class_folder = os.path.join(
        test_folder,
        class_name
    )

    if not os.path.exists(class_folder):
        raise FileNotFoundError(
            f"Folder not found: {class_folder}"
        )

    files = sorted(os.listdir(class_folder))

    for file_name in files:

        if not file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            class_folder,
            file_name
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():

            output = model(image_tensor)

            prediction = torch.argmax(
                output,
                dim=1
            ).item()

        y_true.append(class_index)
        y_pred.append(prediction)

        total_images += 1

# ==============================
# MODEL EVALUATION
# ==============================

if total_images == 0:
    raise ValueError("No test images found!")

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

# ==============================
# RESULTS
# ==============================

print("\n================================")
print("ROADSAFE AI - MODEL EVALUATION")
print("================================")

print("Total Test Images:", total_images)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=classes,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(cm)

print("\nClass Order:")
print("0 =", classes[0])
print("1 =", classes[1])

# ==============================
# CONFUSION MATRIX VISUALIZATION
# ==============================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    linewidths=1,
    linecolor="black"
)

plt.title(
    "RoadSafe AI - Confusion Matrix",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

# Save Heatmap

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nConfusion Matrix saved successfully!")
print(output_path)

plt.show()

plt.close()

# ==============================
# COMPLETED
# ==============================

print("\n================================")
print("MODEL EVALUATION COMPLETED!")
print("================================")
