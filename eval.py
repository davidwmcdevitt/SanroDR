import os
import argparse
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    
    parser = argparse.ArgumentParser(description='Sanro Health Evaluation Script')
    
    parser.add_argument('--eval_set', type=str, required=True, choices=['kaggle', 'messidor2'], help='Evaluation dataset')
    
    parser.add_argument('--model_dict', type=str, required=True, help='State dictionary to evaluate')
    parser.add_argument('--data_dir', type=str, required=False, default='test', help='Data Directory')
    
    return parser.parse_args()

class ScanDataset(Dataset):
    
    def __init__(self, images, ratings, transform=None):
        
        self.images = images
        self.ratings = ratings
        self.transform = transform

    def __len__(self):
        self.length = len(self.images)
        return self.length

    def __getitem__(self, idx):
      
        img_path = os.path.join(self.images[idx])
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = self.ratings[idx]
        
        return img_transformed, label


class Evaluator:
    
    def __init__(self, args):
        
        self.data_dir = args.data_dir
        self.eval_set = args.eval_set
        
        self.test_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = torch.cuda.device_count()
        self.batch_size = 16
        
        self.model = timm.create_model('vit_base_patch16_384', pretrained=False, num_classes=5)
        self.state_dict = torch.load(args.model_dict)
        
        self.model.load_state_dict(self.state_dict)
        
        self.model = self.model.to(self.device)
        
        self.model.eval()
        
        print(f'Model {args.model_dict} successfully loaded.')
        
        if self.eval_set == 'kaggle':
                
            self.images_dir = os.path.join(self.data_dir, 'images')
            self.ratings_path = os.path.join(self.data_dir, 'labels.csv')
            
            self.labels_db = pd.read_csv(self.ratings_path)
            self.labels_db = self.labels_db[self.labels_db['path'].apply(lambda x: os.path.isfile(os.path.join(self.data_dir, x)))]
                
            self.images = [os.path.join(self.images_dir, item) for item in os.listdir(self.images_dir)]
            
            self.ratings = self.labels_db[self.labels_db['path'].isin([os.path.join('images', item) for item in os.listdir(self.images_dir)])]
            
            self.ratings = self.ratings['rating'].tolist()
            
        if self.eval_set == 'messidor2':
                
            self.images_dir = os.path.join(self.data_dir, 'messidor-2/messidor-2/preprocess')
            self.ratings_path = os.path.join(self.data_dir, 'messidor_data.csv')
            
            self.labels_db = pd.read_csv(self.ratings_path)
            self.labels_db = self.labels_db[self.labels_db['id_code'].apply(lambda x: os.path.isfile(os.path.join(self.images_dir, x)))]
                
            self.images = [os.path.join(self.images_dir, item) for item in os.listdir(self.images_dir)]
            
            self.ratings = self.labels_db[self.labels_db['id_code'].isin([item for item in os.listdir(self.images_dir)])]
            
            self.ratings = self.ratings['diagnosis'].tolist()

                
        self.test_data = ScanDataset(self.images, self.ratings, transform=self.test_transforms)
            
        self.test_loader = DataLoader(dataset = self.test_data, batch_size=self.batch_size, shuffle=False)
        
        print(f'{len(self.test_data)} test images found, {len(self.test_loader)} batchs per epoch.')
        
    def evaluate(self):
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            test_true = []
            test_preds = []
            
            test_loss = 0.0
            
            for data, label in tqdm(self.test_loader):
                data, label = data.to(self.device), label.to(self.device)
            
                output = self.model(data)
                loss = criterion(output, label)
                pred = output.argmax(dim=1, keepdim=True)
            
                test_true.extend(label.cpu().numpy())
                test_preds.extend(pred.cpu().numpy())
                
                test_loss += loss.item() / len(self.test_loader)
                
            test_true = np.array(test_true)
            test_preds = np.array(test_preds)
            
            np.save(os.path.join('outputs', f'{self.eval_set}_test_true.npy'), test_true)
            np.save(os.path.join('outputs', f'{self.eval_set}_test_preds.npy'), test_preds)
            
            accuracy = accuracy_score(test_true, test_preds)
            precision = precision_score(test_true, test_preds, average='macro')  # 'macro' can be changed based on needs
            recall = recall_score(test_true, test_preds, average='macro')  # 'macro' can be changed based on needs
            f1 = f1_score(test_true, test_preds, average='macro')  # 'macro' can be changed based on needs
            conf_matrix = confusion_matrix(test_true, test_preds)
            
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix')
            cm_path = os.path.join('outputs', f'{self.eval_set}_confusion_matrix.png')
            plt.savefig(cm_path)
            print(f'Confusion Matrix saved at {cm_path}')

if __name__ == "__main__":
    
    args = parse_args()
    
    sanro = Evaluator(args)
    
    sanro.evaluate()