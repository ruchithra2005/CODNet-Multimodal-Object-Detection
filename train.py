import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import json
from tqdm import tqdm
import pickle

# --- CLUSTER STABILITY FIXES ---
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# --- CONFIGURATION ---
ANNOTATIONS_PATH = '/nfsshare/users/ak/codnet_project/annotations/instances_train2017.json'
IMAGES_DIR = '/nfsshare/users/ak/codnet_project/train2017'
CACHE_PATH = 'coco_samples_cache.pkl'
MODEL_SAVE_PATH = 'codnet_best.pth'

BATCH_SIZE = 128    
NUM_EPOCHS = 25
LEARNING_RATE = 6e-4
NUM_WORKERS = 0     

# --- DATASET DEFINITION ---
class COCODataset(Dataset):
    def __init__(self, annotations_path, images_dir, transform=None, cache_path=None):
        self.images_dir = images_dir
        self.transform = transform
        
        if cache_path and os.path.exists(cache_path):
            print(f"📦 Loading cached sample list from {cache_path}...")
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            print("🔍 Scanning annotations (this happens once)...")
            with open(annotations_path, 'r') as f:
                data = json.load(f)
            
            img_id_to_path = {img['id']: os.path.join(images_dir, img['file_name']) for img in data['images']}
            img_id_to_cats = {}
            for ann in data['annotations']:
                img_id = ann['image_id']
                cat_id = ann['category_id']
                if img_id not in img_id_to_cats:
                    img_id_to_cats[img_id] = set()
                img_id_to_cats[img_id].add(cat_id)
            
            self.samples = []
            for img_id, path in img_id_to_path.items():
                if img_id in img_id_to_cats:
                    self.samples.append({'path': path, 'cats': list(img_id_to_cats[img_id])})
            
            if cache_path:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.samples, f)
        
        # --- ROBUST CATEGORY MAPPING ---
        all_cats_set = set()
        for s in self.samples:
            if isinstance(s, dict) and 'cats' in s:
                for cat in s['cats']:
                    all_cats_set.add(cat)
        
        all_cats = sorted(list(all_cats_set))
        self.cat_to_idx = {cat: i for i, cat in enumerate(all_cats)}
        print(f"✅ Mapping created for {len(self.cat_to_idx)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s['path']).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception:
            # Return a blank image if network drive fails
            img = torch.zeros(3, 224, 224)
        
        target = torch.zeros(80)
        for cat_id in s['cats']:
            if cat_id in self.cat_to_idx and self.cat_to_idx[cat_id] < 80:
                target[self.cat_to_idx[cat_id]] = 1.0
        return img, target

# --- MODEL DEFINITION ---
class CodNet(nn.Module):
    def __init__(self, num_classes=80):
        super(CodNet, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# --- TRAINING LOOP ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = COCODataset(ANNOTATIONS_PATH, IMAGES_DIR, transform, CACHE_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"✅ Dataset size: {len(dataset):,}")

    model = CodNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_wa = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        correct_preds = 0
        total_elements = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_preds += (preds == targets).sum().item()
            total_elements += targets.numel()
            
            avg_wa = (correct_preds / total_elements) * 100
            pbar.set_postfix(loss=f"{loss.item():.4f}", AvgWA=f"{avg_wa:.1f}%")

        epoch_wa = (correct_preds / total_elements) * 100
        if epoch_wa > best_wa:
            best_wa = epoch_wa
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🌟 New Best Model! AvgWA: {best_wa:.2f}%")

if __name__ == "__main__":
    train()

