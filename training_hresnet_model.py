import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models, datasets
import numpy as np
#define transformations for the training and the training dataset
#!wget -O part-000002.zip 'https://storage.googleapis.com/kaggle-data-sets/5862440/9608141/compressed/data.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241013%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241013T043423Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4d48acfce1ef0e8ed4e7dbd865b86f3a426d7ec62157a610024081ab5cc0aaa4ec24f041f520cb7dc6f1a1386fe0eb3d0de563c827a3bccf300e0815753f20d0288bdd0abd5a28508928a968c02969219597bef048af928ea5aa727a0d4e8f671c4677427d143e22b868719dd7241996accf5a91a66366d8685da53efe18db564a657801fca29a308340ca34ea4342c901e96544691757fd09f2a7b3075890435a43908b6f6160a3f543de663247c9a7cf426a6689e7aa76d472409982c96bfbc5f03784dc3ca3e2154f49049ccaf1ca2aa17e4af513bb0d23e21a5bb7f1d2ec1f519f21baa5477328fcdb3dd45b20ad6cd9256dd08811073c64440f3e55aa4e'
#!unzip part-000002.zip
_transform = transforms.Compose([transforms.Resize((244,244)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#load the training and testing dataset
train_dataset = ImageFolder(root=r"train", transform=_transform)
val_dataset = ImageFolder(root=r"val", transform=_transform)
test_dataset = ImageFolder(root=r"test", transform=_transform)

#create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"using device: {device}")

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

train_losses=[]
val_losses=[]
train_accuracies=[]
val_accuracies=[]

#training loop
num_epochs=10
print("training starts")
for epoch in range(num_epochs):
    print(str(epoch) + '\n')
    #training phase
    running_loss=0.0
    correct_train=0
    total_train=0
    for images, labels in train_loader:
        print(images)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss/len(train_loader)
    train_accuracy = 100 * correct_train/total_train

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model (images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels. size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val/total_val

    # Store losses and accuracies
    train_losses.append (train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    #Print epoch results
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss: .4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy: .2f}%')

