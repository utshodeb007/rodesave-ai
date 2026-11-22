import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. Project Paths
# ==========================================

script_folder = os.path.dirname(
    os.path.abspath(__file__)
)

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

# ==========================================
# 2. Device
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
# 3. CNN Model
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
# 4. Load Trained Model
# ==========================================

model = AccidentCNN()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")

# ==========================================
# 5. Find Test Images
# ==========================================

image_list = []

for class_name in ["accident", "non_accident"]:

    class_folder = os.path.join(
        test_folder,
        class_name
    )

    if not os.path.exists(class_folder):
        continue

    for file in os.listdir(class_folder):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_list.append(
                (
                    os.path.join(
                        class_folder,
                        file
                    ),
                    class_name
                )
            )

# ==========================================
# 6. Check Images
# ==========================================

if len(image_list) == 0:

    print("No test images found!")

    exit()

print(
    "\nTotal test images available:",
    len(image_list)
)

# ==========================================
# 7. Select Random Image
# ==========================================

image_path, actual_class = random.choice(
    image_list
)

print("\n================================")
print("RANDOM IMAGE SELECTED")
print("================================")

print("Image:", os.path.basename(image_path))
print("Actual class:", actual_class)

# ==========================================
# 8. Image Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

image = Image.open(
    image_path
).convert("RGB")

image_tensor = transform(
    image
).unsqueeze(0)

image_tensor = image_tensor.to(device)

# ==========================================
# 9. Prediction
# ==========================================

with torch.no_grad():

    output = model(
        image_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        1
    )

# ==========================================
# 10. Result
# ==========================================

classes = [
    "accident",
    "non_accident"
]

predicted_class = classes[
    predicted.item()
]

confidence_value = (
    confidence.item() * 100
)

print("\n================================")
print("AI PREDICTION")
print("================================")

print(
    "Prediction:",
    predicted_class
)

print(
    f"Confidence: {confidence_value:.2f}%"
)

print(
    "Actual:",
    actual_class
)

# ==========================================
# 11. Check Correct / Wrong
# ==========================================

if predicted_class == actual_class:

    print("Result: CORRECT!")

else:

    print("Result: WRONG!")

print("================================")