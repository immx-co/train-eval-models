import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from os import path
import os
import cv2
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import argparse


class ML_AVEDEX_Dataset(Dataset):
    def __init__(self , ann_file, classes, transforms=None):
        self.classes = classes
        self.annotations = []
        counts = 0
        with open(ann_file, 'r', encoding="utf-8") as ann_file:
            for img_dir in ann_file.readlines():
                counts +=1
                print(f'Директория кадра: {img_dir}')
                print(f'{counts} кадр')
                img_dir = img_dir[:-1].replace('\\', '/')
                ann_dir = img_dir[:img_dir.rfind('.')] + '.txt'
                with open(ann_dir, 'r', encoding='windows-1251') as f:
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
    

parser = argparse.ArgumentParser(description='Оценка модели')
parser.add_argument('-p', '--path', type=str, default=path.join(os.getcwd(), 'backup', 'effnet224', 'more_than_60_epochs', 'efficientnet_b1224_last.pth'), help='Путь к весам')
parser.add_argument('-d', '--data', type=str, default=path.join(os.getcwd(), 'classification_13classes_validation', 'valid_13.txt'), help='Путь к файлу с данными')
parser.add_argument('-i', '--image_size', type=int, default=224, help='Размерность входного разрешения')
args = parser.parse_args()
model_path: str = args.path
valid_dir: str = args.data
img_size: int = args.image_size


classes = ['Passesenger car', 'Bike', 'Bus', 'Light Truck', 'Trailer', 
           'Heavy Machinery', 'Emergency Vehicle', 'Medium Truck', 'Heavy Truck', 'Tractor Unit', 'Minibus', 'Big Bus', 'Long Bus']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_transform = transforms.Compose([transforms.Resize((img_size, img_size)), 
                                      transforms.ToTensor()])

# подготовка модели к оценке
model = torch.load(model_path)
model = model.to(device)
model.eval()

# подготовка данных
valid_dataset = ML_AVEDEX_Dataset(valid_dir, classes, train_transform)
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=1)

# инициализация необходимых переменных
inference_time = 0                    # отвечает за общее время инференса
inference_time_all = []               # массив для сбора времени инференса каждого кадра
cls_actual, cls_predicted = [], []    # массивы для реальных и предсказанных лейблов соответственно
excluding_emrgcs = []                 # массив для всех кадров, исключая emergency cars

# формирование выборки без emergency cars
for image, label in valid_loader.dataset:
    if label[6] != 1:
        excluding_emrgcs.append([image, label])

# без вычисления градиентов формируем реальные и предсказанные данные
with torch.no_grad():
    for image, label in excluding_emrgcs:
        image = image.to(device).unsqueeze(0)
        
        start = time.time()
        predict = model(image)
        predict[0][6] = -10     # исключаем emergency car
        inf_end = time.time() - start
        print(f'Predict: {classes[label.argmax()]} Actual: {classes[predict.argmax()]}')
        inference_time += inf_end
        inference_time_all.append(inf_end)
        
        predict_argmax = int(predict.argmax())
        label_argmax = int(label.argmax())
        cls_actual.append(label_argmax)
        cls_predicted.append(predict_argmax)
    
# для корректного построения confusion matrix нужно внести хотя бы 1 emergency car
cls_actual.append(6)
cls_predicted.append(6)

c_m = confusion_matrix(cls_actual, cls_predicted)
TP_C = np.diag(c_m)
TP_C.setflags(write=True)
TP_C[6] = 0                        # обнуляем добавленную ранее emergency car
FP_C = c_m.sum(axis=0) - np.diag(c_m)
FN_C = c_m.sum(axis=1) - np.diag(c_m)
TN_C = c_m.sum() - (FP_C + FN_C + TP_C)
precision_c = np.round(TP_C / (TP_C + FP_C), 3)
recall_c = np.round(TP_C / (TP_C + FN_C), 3)
f1_c = np.round(2 * (precision_c * recall_c) / (precision_c + recall_c), 3)

def exclude_nan_metrics(arr_met):
    zero_metrics = [c if i != 6 and not np.isnan(c) else 0 for i, c in enumerate(arr_met)]
    zero_metrics = np.array(zero_metrics[:6] + zero_metrics[7:])
    return np.round(zero_metrics.mean(), 3)

TP, FP, FN, TN = sum(TP_C), sum(FP_C), sum(FN_C), sum(TN_C)
precision = exclude_nan_metrics(precision_c)
recall = exclude_nan_metrics(recall_c)
f1 = exclude_nan_metrics(f1_c)

cls_actual = cls_actual[:-1]
cls_predicted = cls_predicted[:-1]
right_predictions, false_predictions = 0, 0
for i in range(len(cls_actual)):
    if cls_actual[i] == cls_predicted[i]:
        right_predictions += 1
    else:
        false_predictions += 1

acc = np.round(right_predictions * 100 / (right_predictions + false_predictions), 2)

p_r_f1 = [np.round(i, 3) for i in precision_recall_fscore_support(cls_actual, cls_predicted, average='macro')[:-1]]

for i, c in enumerate(classes):
    print(f'{c}:\t precision: {precision_c[i]};  recall: {recall_c[i]};  f1-score: {f1_c[i]}')

print(f'Общее: precision: {precision};  recall: {recall};  f1-score: {f1}', end='\n')
print(f'Общее: precision: {p_r_f1[0]};  recall: {p_r_f1[1]};  f1-score: {p_r_f1[2]}', end='\n')
print()
print(f'Верно предсказано - {right_predictions} кадров, неверно - {false_predictions} кадров')
print(f'Общая точность: {acc}%; Процент ошибок: {np.round(100 - acc, 2)}%')
print(f'Общее время инференса: {np.round(inference_time, 2)} секунд\nСреднее время инференса: {np.round(np.array(inference_time_all).mean(), 4)} секунд')


cls_actual.append(6)
cls_predicted.append(6)
ConfusionMatrixDisplay.from_predictions(cls_actual, cls_predicted)
plt.show()

#pred_dict = dict(zip(classes, classes_predict))
#fact_dict = dict(zip(classes, classes_count))

#classes_count = [0 for _ in range(len(classes))]
#except_cars = [i for i in test_loader.dataset if int(i[1].argmax()) != 0]

#image, label = except_cars[1]
#image = image.to(device).unsqueeze(0)
#label = label.to(device)

#prediction = model(image)

#cv2.imshow('car', image[0][0].cpu().numpy())
#cv2.waitKey(0)
#cv2.destroyAllWindows()




