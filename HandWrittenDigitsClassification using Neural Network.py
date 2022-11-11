import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
import torchvision.transforms as transfroms 
import torchvision

input_size = 784 # 28 * 28 
hidden_layer = 100 
num_classes = 10 
epochs = 2
batch_size =100 
alpha = 0.1 

train_dataset = torchvision.datasets.MNIST(root = './data' , train=True , transform=transfroms.ToTensor() , download = True ) 
test_dataset = torchvision.datasets.MNIST(root = './data' , train = True , transform=transfroms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True) 
test_loader = torch.utils.data.DataLoader(dataset = test_dataset , batch_size= batch_size ,shuffle = False ) 
print(len(train_dataset))
example = iter(train_loader)
sample , labels = example.next() 

print(sample.shape, labels.shape) 

class NeuralNet(nn.Module):
    def __init__(self,input_size , hidden_layer , num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size,hidden_layer) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer, num_classes) 
    def forward(self , x):
        out = self.l1(x) 
        out = self.relu(out) 
        out = self.l2(out) 
        return out  

loss = nn.CrossEntropyLoss() 
model = NeuralNet(input_size , hidden_layer  , num_classes )
optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

n_total_steps = len(train_loader) 
print(n_total_steps) 
for epoch in range(epochs):
    for i , (images,labels)in enumerate(train_loader):
        images = images.reshape(-1 ,28*28)
        # print(labels.shape)
        predicted = model(images) 
        cost = loss(predicted , labels) 
        if i%100==0:
            print(f'epoch {epoch}/{epochs} ,steps {i}/{n_total_steps}, cost {cost} ')
        cost.backward() 
        optimizer.step() 
        optimizer.zero_grad() 

with torch.no_grad():
    n_correct = 0 
    n_samples = 0 
    for i , (images,labels) in enumerate(test_loader):
        images = images.reshape(-1 ,28*28)
        pred = model(images) 
        _,prediction = torch.max(pred , 1 ) 
        n_samples+=labels.shape[0]
        n_correct += (prediction.eq(labels)).sum().item() # (prediction==labels).sum().item()
    print(n_samples) 
    print(100 * n_correct / n_samples) 
    
    
