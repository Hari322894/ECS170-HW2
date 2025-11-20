import numpy as np
import matplotlib.pyplot as plt
import gzip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

#need to open up MNIST files
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# Load data
x_train_full = load_mnist_images('train-images-idx3-ubyte.gz')
y_train_full = load_mnist_labels('train-labels-idx1-ubyte.gz')
x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

#normalization
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train_full[:-12000]
y_train = y_train_full[:-12000]
x_val = x_train_full[-12000:]
y_val = y_train_full[-12000:]

x_train = torch.FloatTensor(x_train.reshape(-1, 1, 28, 28))
y_train = torch.LongTensor(y_train)
x_val = torch.FloatTensor(x_val.reshape(-1, 1, 28, 28))
y_val = torch.LongTensor(y_val)
x_test = torch.FloatTensor(x_test.reshape(-1, 1, 28, 28))
y_test_tensor = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test_tensor), batch_size=32, shuffle=False)

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 28, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(28, 56, kernel_size=3)
        self.fc1 = nn.Linear(56 * 11 * 11, 56)
        self.fc2 = nn.Linear(56, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 56 * 11 * 11)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = FashionCNN()
trainable_params = sum(p.numel() for p in model.parameters())
print(f"Number of trainable parameters: {trainable_params:,}")

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    correct = 0
    total = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    return correct / total

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    return correct / total

train_acc_history = []
val_acc_history = []

#run 10 epochs
print("Training for 10 epochs...")
for epoch in range(10):
    train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_acc = evaluate(model, val_loader)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    print(f"Epoch {epoch+1:2d}/10 - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), train_acc_history, 'o-', label='Training Accuracy')
plt.plot(range(1, 11), val_acc_history, 's-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()

test_acc = evaluate(model, test_loader)
print(f"\nTest Accuracy: {test_acc:.4f}")

model.eval()
all_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        outputs = model(batch_x)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())

y_pred = np.array(all_preds)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Misclassified Examples')

for true_class in range(10):
    ax = axes[true_class // 5, true_class % 5]
    misclassified_idx = np.where((y_test == true_class) & (y_pred != true_class))[0]
    
    if len(misclassified_idx) > 0:
        idx = misclassified_idx[0]
        img = x_test[idx].squeeze().numpy()
        pred_class = y_pred[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {class_names[true_class]}\nPred: {class_names[pred_class]}', fontsize=9)
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, f'No errors\n{class_names[true_class]}', ha='center', va='center')
        ax.axis('off')

plt.tight_layout()
plt.savefig('misclassified_examples.png')
plt.show()