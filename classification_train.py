import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from os import path
import geffnet
import time
import argparse
from PIL import Image


class ML_AVEDEX_Dataset(Dataset):
    def __init__(self , ann_file, classes, transforms=None):
        self.classes = classes
        self.annotations = []
        counts = 0
        with open(ann_file, 'r', encoding="utf-8") as ann_file:
            for img_dir in ann_file.readlines():
                counts +=1
                print(f'Директория картинки: {img_dir}')
                print(f'{counts} картинка')
                img_dir = img_dir[:-1].replace('\\', '/')
                ann_dir = img_dir[:img_dir.rfind('.')] + '.txt'
                with open(ann_dir, 'r', encoding='utf-8') as f:
                    data = [0]*len(self.classes)
                    for i in f.readlines():
                        data[int(i[:-1])] = 1
                    labels = torch.tensor(data, dtype=torch.float32)
                self.annotations.append([img_dir, labels])  
        self.transforms = transforms
      
    def __getitem__(self, idx):
        img_dir = self.annotations[idx][0]
        image = Image.open(img_dir).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
          
        labels = self.annotations[idx][1]
          
        return image, labels
    
    def __len__(self):
        return len(self.annotations)
  
def train(model, optimizer, loss_function, train_loader, test_loader, epochs, device, backup_dir):
    global model_name, img_size
    
    torch.save(model, f'{backup_dir}/{model_name}{img_size}_first.pth')
    total_step = len(train_loader)
    
    for epoch in range(epochs):
        print(f'{epoch + 1} Эпоха')
        print(total_step)
        train_loss = 0
        for i, batch in enumerate(train_loader):
            #print(i, batch)
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if (i + 1) % (total_step // 1000) == 0:
                print(f'Эпоха [{epoch + 1}/{epochs}], Шаг [{i + 1}/{total_step}], Тренировочные потери: {loss.data.item():.4f}')
        
        torch.save(model, f'{backup_dir}/{model_name}{img_size}_epoch_{epoch + 1}.pth')
    
    torch.save(model, f'{backup_dir}/{model_name}{img_size}_last.pth')
    #with torch.no_grad():
        #correct = 0 
        #total = 0
        #for batch in test_loader:
            #images, labels = batch
            #images = images.to(device)
            #labels = labels.to(device)
            #outputs = model(images)
            #_, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()
            
        #print(f'Точность на тестовом наборе {100 * correct / total}%')
        
parser = argparse.ArgumentParser(description='Тренировка модели на 13 классов')
parser.add_argument('-m', '--model', type=str, help='Имя модели для тренировки')
parser.add_argument('-md', '--train_dir', type=str, default=path.join(os.getcwd(), 'classification_13classes', 'data', 'classification_13classes_train.txt'), help='Путь к тренировочному набору данных')
parser.add_argument('-td', '--test_dir', type=str, default=path.join(os.getcwd(), 'classification_13classes', 'data', 'classification_13classes_test.txt'), help='Путь к тестовому набору данных')
parser.add_argument('-i', '--image_size', type=int, default=224, help='Входное разрешение')
parser.add_argument('-b', '--backup', type=str, default=path.join(os.getcwd(), 'backup', 'mixnet_l224'), help='Папка для сохранения обученых весов')
parser.add_argument('-p', '--pretrained', type=str, default='', help='Путь к обученым весам')
args = parser.parse_args()
model_name = args.model
img_size = args.image_size
pretrained_weights = args.pretrained

classes = ['Passesenger car', 'Bike', 'Bus', 'Light Truck', 'Trailer', 
           'Heavy Machinery', 'Emergency Vehicle', 'Medium Truck', 'Heavy Truck', 'Tractor Unit', 'Minibus', 'Big Bus', 'Long Bus']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_transform = transforms.Compose([transforms.Resize((img_size, img_size)), 
                                      transforms.ToTensor()])


if __name__ == '__main__':
    backup_dir = args.backup
    train_dir = args.train_dir
    test_dir = args.test_dir
    
    
    train_dataset = ML_AVEDEX_Dataset(train_dir, classes, train_transform)
    test_dataset = ML_AVEDEX_Dataset(test_dir, classes, train_transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16)
    print(model_name)
    
    if model_name in ['mixnet_l', 'efficientnet_b1']:
        if model_name == 'mixnet_l':
            model = geffnet.mixnet_l()
            model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, len(classes), bias=True))
        else:
            model = geffnet.efficientnet_b1()
            model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, len(classes), bias=False))
    else:
        raise Exception('Use mixnet_l or efficientnet_b1')
    
    if pretrained_weights:
        pretrained_model = torch.load(pretrained_weights)
        model.load_state_dict(pretrained_model.state_dict())
        
        
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss_func = nn.MSELoss()
    epochs = 20
    start = time.time()
    
    train(model, optimizer, loss_func, train_loader, test_loader, epochs, device, backup_dir)
    
    train_time = time.time() - start
    
    print(f'Время обучения: {train_time}')




