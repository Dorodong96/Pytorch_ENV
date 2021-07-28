''' 1. Module Import '''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

''' 2. 딥러닝 모듈을 설계할 때 활용하는 장비 확인 '''
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print('Using PyTorch version :', torch.__version__, ' Device :', DEVICE)

BATCH_SIZE = 32
EPOCHS = 15

''' 3. MNIST 데이터 Download (Trainset, Test set 분리) '''
train_dataset = datasets.MNIST(root="../data/MNIST",
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="../data/MNIST",
                              train=False,
                              transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

''' 4. 데이터 확인하기 (1) '''
for(X_train, y_train) in train_loader:
    print('X_train :', X_train.size(), 'type :', X_train.type())
    print('y_train :', y_train.size(), 'type :', y_train.type())
    break

''' 5. 데이터 확인하기 (2) '''
pltsize = 1
plt.figure(figsize=(10*pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap="gray_r")
    plt.title('Class : ' + str(y_train[i].item()))
plt.show()

''' 6. MLP (Multi Layer Perception) 모델 설계하기 '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # fc : Fully Connected Layer
        self.fc1 = nn.Linear(28*28, 512) # MNIST Data Input (28 * 28 * 1(Channel) Image)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10) # 0~9 10개의 Class 구분

    def forward(self, x): # Forward Propagation
        x = x.view(-1, 28*28)  # Flatten (2차원 데이터를 1차원 데이터로 변환)
        x = self.fc1(x)  # flatten 데이터를 fc1에 통과
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        # (0~1로 정규화, 합이 1이 되도록, Back propagation에서 Gradient 원활히 계산)
        return x

''' 7. Optimizer, Objective Function 설정하기 '''
model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)

''' 8. MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Train Epoch : {}, [{}/{}({:.0f}%)]\tTrain Loss : {:.6f}".format(
                Epoch, batch_idx*len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

''' 9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

''' 10. MLP 학습을 실행하며 Train, Test set의 Loss 및 Test set Accuracy 확인하기 '''
for Epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH : {}], \tTest Loss : {:.4f}, \tTest Accuracy : {:.2f} % \n"
          .format(Epoch, test_loss, test_accuracy))