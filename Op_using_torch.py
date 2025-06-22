import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.svm import SVC  

print("Loading MNIST with PyTorch...")

dataset = torchvision.datasets.MNIST('.', download=True, transform=torchvision.transforms.ToTensor())






print("Success! MNIST loaded with PyTorch!")







train_images = []
train_labels = []
# Take first 10000 images for training
for i in range(10000):
    image, label = dataset[i]
    flat_image = image.view(-1).numpy()  # Flatten to 784 numbers
    train_images.append(flat_image)
    train_labels.append(label)
print(f"Got {len(train_images)} training images")

# === SIMPLE KNN MODEL ===
#KNN
print("\n=== TRYING KNN ===")



# Create and train KNN
knn = KNeighborsClassifier(n_neighbors=3)  # Use 3 nearest neighbors
knn.fit(train_images, train_labels)
print("KNN trained!")
# Prepare multiple test images for accuracy calculation
test_images = []
test_labels = []
test_size = 1000

print("Preparing test images...")
for i in range(10000, 10000 + test_size):
    image, label = dataset[i]
    flat_image = image.view(-1).numpy()
    test_images.append(flat_image)
    test_labels.append(label)
print(f"Got {test_size} test images")

#LOGISTIC REGRESSION
# === SIMPLE LOGISTIC REGRESSION ===

from sklearn.linear_model import LogisticRegression

print("\n=== TRYING LOGISTIC REGRESSION ===")

# Create and train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(train_images, train_labels)





# === RANDOM FOREST ===
from sklearn.ensemble import RandomForestClassifier

print("\n=== TRYING RANDOM FOREST ===")

# Create and train Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42)  # 50 trees
rf.fit(train_images, train_labels)




# === SIMPLE NEURAL NETWORK ===
import torch.nn as nn

print("\n=== TRYING SIMPLE NEURAL NETWORK ===")

# Simple neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),  # 784 pixels -> 128 neurons
            nn.ReLU(),            # Activation function
            nn.Linear(128, 10)    # 128 neurons -> 10 digits
        )
    
    def forward(self, x):
        return self.layers(x)

# Create model
model = SimpleNN()
print("Neural Network created!")
# Train the neural network
import torch.optim as optim

# Convert training data to PyTorch tensors
train_data = torch.FloatTensor(train_images)
train_labels_tensor = torch.LongTensor(train_labels)

# Set up training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training neural network...")
model.train()
for epoch in range(200):  # Train for 200 epochs
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_labels_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:  # Print progress every 20 epochs
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Neural Network trained!")



#=== DECISION TREE ===

from sklearn.tree import DecisionTreeClassifier

print("\n=== TRYING DECISION TREE ===")

# Create and train Decision Tree
dt = DecisionTreeClassifier(max_depth=10, random_state=42)  # Limit depth so it doesn't overfit
dt.fit(train_images, train_labels)


# === SVM ===
# === IMPROVED SVM ===
from sklearn.preprocessing import StandardScaler

print("\n=== TRYING IMPROVED SVM ===")

# Normalize data (helps SVM a lot!)
scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images)
test_images_scaled = scaler.transform(test_images)
print(f"test_images_scaled shape: {test_images_scaled.shape}")  # ADD THIS LINE

# Better SVM parameters
# Use RBF kernel with stronger parameters
svm = SVC(
    kernel='rbf',           # Back to RBF
    C=100,                  # Much stronger (was 10)
    gamma=0.001,            # Fine-tuned
    random_state=42
)
svm.fit(train_images_scaled, train_labels)



# === CNN (CONVOLUTIONAL NEURAL NETWORK) ===
import torch.nn.functional as F

print("\n=== TRYING IMPROVED CNN ===")

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Same architecture as before but with dropout
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.dropout(x)  # Add dropout
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create improved CNN
cnn_model = ImprovedCNN()
print("Improved CNN created!")

# More training data
print("Preparing CNN training data...")
cnn_train_data = []
for i in range(10000):  # Increased from 3000 to 5000
    image, label = dataset[i]
    cnn_train_data.append(image)

cnn_train_tensor = torch.stack(cnn_train_data)
cnn_train_labels = torch.LongTensor(train_labels)

# Enhanced training
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
cnn_criterion = nn.CrossEntropyLoss()

print("Training Improved CNN...")
cnn_model.train()
for epoch in range(200):  # Reduced from 150 to be safe
    cnn_optimizer.zero_grad()
    outputs = cnn_model(cnn_train_tensor)
    loss = cnn_criterion(outputs, cnn_train_labels)
    loss.backward()
    cnn_optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Improved CNN trained!")

# === ACCURACY COMPARISON ===
from sklearn.metrics import accuracy_score

print("\n=== ACCURACY RESULTS ===")
print(f"Tested on {test_size} images:")

# Calculate accuracy for each model
knn_acc = accuracy_score(test_labels, knn.predict(test_images))
lr_acc = accuracy_score(test_labels, lr.predict(test_images))
rf_acc = accuracy_score(test_labels, rf.predict(test_images))
dt_acc = accuracy_score(test_labels, dt.predict(test_images))
svm_acc = accuracy_score(test_labels, svm.predict(test_images_scaled))

# ADD THIS for Neural Network:
model.eval()
test_tensor = torch.FloatTensor(test_images)
with torch.no_grad():
    nn_outputs = model(test_tensor)
    nn_predictions = torch.argmax(nn_outputs, dim=1).numpy()
nn_acc = accuracy_score(test_labels, nn_predictions)


# CNN accuracy
cnn_test_data = []
for i in range(10000, 10000 + test_size):
    image, label = dataset[i]
    cnn_test_data.append(image)
cnn_test_tensor = torch.stack(cnn_test_data)

cnn_model.eval()
with torch.no_grad():
    cnn_outputs = cnn_model(cnn_test_tensor)
    cnn_predictions = torch.argmax(cnn_outputs, dim=1).numpy()
cnn_acc = accuracy_score(test_labels, cnn_predictions)


print(f"KNN: {knn_acc:.3f} ({knn_acc*100:.1f}%)")
print(f"Logistic Regression: {lr_acc:.3f} ({lr_acc*100:.1f}%)")
print(f"Random Forest: {rf_acc:.3f} ({rf_acc*100:.1f}%)")
print(f"Decision Tree: {dt_acc:.3f} ({dt_acc*100:.1f}%)")
print(f"SVM: {svm_acc:.3f} ({svm_acc*100:.1f}%)")
print(f"Neural Network: {nn_acc:.3f} ({nn_acc*100:.1f}%)")
print(f"CNN: {cnn_acc:.3f} ({cnn_acc*100:.1f}%)") 

# === ENSEMBLE METHOD ===
print(f"\n=== TRYING ENSEMBLE (CNN + KNN + Random Forest) ===")

# Get predictions from top 3 models
cnn_probs = torch.softmax(cnn_outputs, dim=1).numpy()
knn_probs = knn.predict_proba(test_images)
rf_probs = rf.predict_proba(test_images)

# Weighted average (CNN gets more weight since it's best)
ensemble_probs = (0.7 * cnn_probs + 0.15 * knn_probs + 0.15 * rf_probs)
ensemble_predictions = np.argmax(ensemble_probs, axis=1)

ensemble_acc = accuracy_score(test_labels, ensemble_predictions)
print(f"Ensemble: {ensemble_acc:.3f} ({ensemble_acc*100:.1f}%)")

# Update winner comparison
best_acc = max(knn_acc, lr_acc, rf_acc, dt_acc, svm_acc, nn_acc, cnn_acc, ensemble_acc)









if best_acc == knn_acc:
    print(f"\n🏆 Winner: KNN with {best_acc*100:.1f}% accuracy!")
elif best_acc == lr_acc:
    print(f"\n🏆 Winner: Logistic Regression with {best_acc*100:.1f}% accuracy!")
elif best_acc == rf_acc:
    print(f"\n🏆 Winner: Random Forest with {best_acc*100:.1f}% accuracy!")
elif best_acc == dt_acc:
    print(f"\n🏆 Winner: Decision Tree with {best_acc*100:.1f}% accuracy!")
elif best_acc == nn_acc:
    print(f"\n🏆 Winner: Neural Network with {best_acc*100:.1f}% accuracy!")
elif best_acc == cnn_acc:
    print(f"\n🏆 Winner: CNN with {best_acc*100:.1f}% accuracy!")
elif best_acc == ensemble_acc:
    print(f"\n🏆 Winner: Ensemble with {best_acc*100:.1f}% accuracy!")
else:
    print(f"\n🏆 Winner: SVM with {best_acc*100:.1f}% accuracy!")

