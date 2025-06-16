import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

class TransferLearningModel:
    def __init__(self, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"\nИспользуется устройство: {device}")
        print("Загрузка предобученной модели ResNet50...")
        # Загружаем предобученную ResNet50 с современным подходом
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Замораживаем веса базовой модели
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Заменяем последний слой на новый для нашего количества классов
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.model = self.model.to(device)
        print("Модель успешно инициализирована и перемещена на", device)
        
        # Определяем функцию потерь и оптимизатор
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        
        # Определяем преобразования для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

    def train(self, train_loader, val_loader, epochs=10):
        print("\nНачало процесса обучения...")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\nЭпоха [{epoch+1}/{epochs}]")
            
            # Обучение
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            train_pbar = tqdm(train_loader, desc='Обучение')
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            
            # Валидация
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            val_pbar = tqdm(val_loader, desc='Валидация')
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * correct / total
            val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f'Результаты эпохи {epoch+1}:')
            print(f'Время выполнения: {epoch_time:.2f} сек')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'Сохранена новая лучшая модель (val_loss: {val_loss:.4f})')

    def evaluate(self, test_loader):
        print("\nНачало процесса оценки модели...")
        self.model.eval()
        all_preds = []
        all_labels = []
        
        test_pbar = tqdm(test_loader, desc='Тестирование')
        with torch.no_grad():
            for inputs, labels in test_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Выводим метрики качества
        print("\nОтчет о классификации:")
        print(classification_report(all_labels, all_preds))
        
        # Строим матрицу ошибок
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Матрица ошибок')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("\nМатрица ошибок сохранена в файл 'confusion_matrix.png'")

# Пример использования
if __name__ == "__main__":
    print("Начало работы программы...")
    num_classes = 10
    
    # Создаем преобразования для данных
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    
    print("\nЗагрузка датасета CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    print(f"Загружено {len(trainset)} изображений для обучения")
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    print(f"Загружено {len(testset)} изображений для тестирования")
    
    model = TransferLearningModel(num_classes=num_classes)
    model.train(trainloader, testloader, epochs=10)
    
    model.evaluate(testloader)
    print("\nПрограмма завершена успешно!") 