import argparse
import os
import pandas as pd
import numpy as np
import random
from datetime import datetime

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import timm
from imblearn.over_sampling import SMOTE


def parse_args():
    
    parser = argparse.ArgumentParser(description='Sanro Health Training Script')
    
    parser.add_argument('--data_dir', type=str, required=False, default='data', help='Data Directory')
    parser.add_argument('--num_epochs', type=int, required=False, default=25, help='Number of Epochs')
    parser.add_argument('--oversample', action='store_true', required=False, default=False, help='Oversample Minority Classes')
    parser.add_argument('--class_weights', action='store_true', required=False, default=False, help='Include class weights')
    parser.add_argument('--force_balance', action='store_true', required=False, default=False, help='Train on 50% No DR and 50% any DR')
    
    parser.add_argument('--state_dict', type=str, required=False, help='Continue training from state dictionary')
    
    return parser.parse_args()

class ScanDataset(Dataset):
    
    def __init__(self, data_dir, images, ratings, transform=None):
        
        self.data_dir = data_dir
        
        self.images = images
        self.ratings = ratings
        self.transform = transform

    def __len__(self):
        self.length = len(self.images)
        return self.length

    def __getitem__(self, idx):

        try:
          img_path = os.path.join(self.data_dir, self.images[idx])
          img = Image.open(img_path)
          img_transformed = self.transform(img)

          label = self.ratings[idx]

        except:
          img_path = self.images[0]
          img = Image.open(img_path)
          img_transformed = self.transform(img)

          label = self.ratings[0]

        return img_transformed, label


class Trainer:
    
    def __init__(self, args):
        
        self.data_dir = args.data_dir
        self.images_dir = os.path.join(self.data_dir, 'images')
        self.ratings_path = os.path.join(self.data_dir, 'labels.csv')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = torch.cuda.device_count()
        
        self.num_epochs = args.num_epochs
        
        self.oversample = args.oversample
        self.weight_classes = args.class_weights
        self.force_balance = args.force_balance
        
        if args.state_dict:
            self.continue_train = True
            self.state_dict = args.state_dict
        else:
            self.continue_train = False
        
        self.batch_size = 16
        self.lr = 1e-4
        self.gamma = 0.95
        
        self.train_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.025),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_data(self):
        
        if not os.path.exists('experiments'):
            os.makedirs('experiments')
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        
        self.labels_db = pd.read_csv(self.ratings_path)
        self.labels_db = self.labels_db[self.labels_db['path'].apply(lambda x: os.path.isfile(os.path.join(self.data_dir, x)))]

        if self.force_balance:
            
            balanced_sample = self.labels_db[self.labels_db['rating'] != 0]
            balanced_sample_paths = balanced_sample['path'].to_list()
            
            no_dbr = self.labels_db[self.labels_db['rating'] == 0]
            no_dbr_paths = no_dbr['path'].to_list()
            
            no_dbr_paths = random.sample(no_dbr_paths, len(balanced_sample_paths))
            
            self.images = balanced_sample_paths + no_dbr_paths
            
        else:

            self.images = ['images/' + item for item in os.listdir(self.images_dir)]
            
        self.ratings = self.labels_db[self.labels_db['path'].isin(self.images)]
        
        self.ratings = self.ratings['rating'].tolist()
        
        self.train_images, self.val_images, self.train_ratings, self.val_ratings = train_test_split(self.images, self.ratings, test_size=0.2, stratify=self.ratings, random_state=42)
        
        if self.oversample:
            
            smote = SMOTE(random_state=42)
            
            train_ratings_array = np.array(self.train_ratings)

            resampled_indices, self.train_ratings = smote.fit_resample(np.arange(len(self.train_images)).reshape(-1, 1), train_ratings_array)
            resampled_indices = resampled_indices.flatten()
            
            self.train_images = [self.train_images[i] for i in resampled_indices]
            
        self.train_data = ScanDataset(self.data_dir, self.train_images, self.train_ratings, transform=self.train_transforms)
        self.val_data = ScanDataset(self.data_dir, self.val_images, self.val_ratings, transform=self.val_transforms)
        
        self.train_loader = DataLoader(dataset = self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(dataset = self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        print(f'{len(self.train_data)} training images found, {len(self.train_loader)} batchs per epoch.')
        
        print(f'{len(self.val_data)} validation images found, {len(self.val_loader)} batchs per epoch.')
        
        if self.weight_classes:
            
            self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.train_ratings), y=self.train_ratings)
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)
            
            print(f'Class weights: {self.class_weights}')
            
    def train(self):
        
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=len(np.unique(self.ratings)))
        self.model = self.model.to(self.device)
        
        if self.continue_train:
            print(f'Loading state dictionary from {self.state_dict}')
            experiment_path = os.path.join('experiments', self.state_dict)
            
        else:
            
            model_path = 'Sanro_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f'Building model {model_path}')
            
            experiment_path = os.path.join('experiments', model_path)
            logs_path = os.path.join(experiment_path,'logs')
            
            for dir_ in [experiment_path, logs_path]:
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
                    
            train_loss_list = []
            train_acc_list = []
            val_loss_list = []
            val_acc_list = []
            
            best_val_accuracy = 0.0
            start_epoch = 0

        if self.weight_classes:
            
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            
        else:
            
            criterion = nn.CrossEntropyLoss()
            
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)

        print(f'Starting from epoch {start_epoch}')
        for epoch in range(self.num_epochs):
            epoch += start_epoch
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            self.model.train()  

            for data, label in tqdm(self.train_loader):
                print(label)
                print(type(label))
                data, label = data.to(self.device), label.to(self.device)
        
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
        
                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(self.train_loader)
                epoch_loss += loss.item() / len(self.train_loader)
        
            train_loss_list.append(epoch_loss)
            train_acc_list.append(epoch_accuracy.detach().cpu().numpy())
        
            self.model.eval()  
            with torch.no_grad():
                epoch_val_accuracy = 0.0
                epoch_val_loss = 0.0
                val_true = []
                val_preds = []
        
                for data, label in tqdm(self.val_loader):
                    data, label = data.to(self.device), label.to(self.device)
        
                    val_output = self.model(data)
                    val_loss = criterion(val_output, label)
        
                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(self.val_loader)
                    epoch_val_loss += val_loss.item() / len(self.val_loader)
        
                    val_preds.extend(val_output.argmax(dim=1).cpu().numpy())
                    val_true.extend(label.cpu().numpy())
        
                val_loss_list.append(epoch_val_loss)
                val_acc_list.append(epoch_val_accuracy.detach().cpu().numpy())
        
                if epoch_val_accuracy > best_val_accuracy:
                    best_val_accuracy = epoch_val_accuracy
                    torch.save(self.model.state_dict(), os.path.join(experiment_path, f'model_{epoch}.pth'))
                    print(f"Saved new best model with validation accuracy: {best_val_accuracy:.4f}")
        
                print(f"Epoch: {epoch} - loss: {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - "
                      f"val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}")
        
            np.save(os.path.join(logs_path, 'train_loss.npy'), np.array(train_loss_list))
            np.save(os.path.join(logs_path, 'train_acc.npy'), np.array(train_acc_list))
            np.save(os.path.join(logs_path, 'val_loss.npy'), np.array(val_loss_list))
            np.save(os.path.join(logs_path, 'val_acc.npy'), np.array(val_acc_list))
        
        np.save(os.path.join(logs_path, 'train_loss.npy'), np.array(train_loss_list))
        np.save(os.path.join(logs_path, 'train_acc.npy'), np.array(train_acc_list))
        np.save(os.path.join(logs_path, 'val_loss.npy'), np.array(val_loss_list))
        np.save(os.path.join(logs_path, 'val_acc.npy'), np.array(val_acc_list))
        
        

if __name__ == "__main__":
    
    args = parse_args()
    
    sanro = Trainer(args)
    
    sanro.load_data()
    
    sanro.train()