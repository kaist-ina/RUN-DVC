#!/usr/bin/env python

import torch
import sys
import yaml
from torchvision import transforms, datasets
import torchvision

import numpy as np
import os
import json
import datetime
# from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
sys.path.append('../')
from argparse import ArgumentParser

"""
Youngmok:
Visualize features

"""
class FocalLoss(torch.nn.Module):
    def __init__(self,  effective_label_num=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
    def forward(self, input, target):
        input =  torch.clamp(input, 1e-9, 1-1e-9)
        cross_entropy = -target * torch.log(input)
        cross_entropy = cross_entropy.sum(-1)  # [N, label_channel]
        loss = ((1-input)**self.gamma) * target
        loss = loss.sum(-1)
        loss = loss * cross_entropy 
        return loss.mean()


parser = ArgumentParser()
parser.add_argument('model')
parser.add_argument('bin_fn')
parser.add_argument('--platform', type=str, default="ilmn")
args = parser.parse_args()

batch_size = 512
data_transforms = torchvision.transforms.Compose([transforms.ToTensor()])


device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
import params.param_adamatch as param
from sequence_dataset_DA import Get_Dataloader_DA, Get_Dataloader_augmentation
np.random.seed(1)
import models as model_path

encoder = model_path.Clair3_Feature( [1,1,1], args.platform)

checkpoint = torch.load(args.model, map_location=torch.device(device))
encoder.load_state_dict(checkpoint['encoder_weights'])

encoder = encoder.to(device)


from tqdm import tqdm

TASK_TYPE = 0 # 0 for genotype label

def get_features_from_encoder( loader):
    
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    print("Obtaining features from Encoder!")
    with torch.no_grad():
        encoder.eval()
        for i, (x, y) in tqdm(enumerate(loader), desc= "Acquiring Encoder Feature" ):
            x = x.squeeze().to(device)
            feature_vector = encoder(x)
            feature_vector = F.normalize(feature_vector, dim=1).to('cpu')
            x_train.extend(feature_vector)

            # youngmok: need to split y_train label, first 21 one-hot style for genotypes
            # youngmok: 4 labels are loaded from loader, [genotype][zygosity][indel length1][indel length2]
            y = y[TASK_TYPE].squeeze().to('cpu')
            y_train.extend(y.numpy())

            if i > 1 :
                break
            # if i % 10 == 0:
            #     print("FeatureV:",feature_vector[0][:10])
    x = torch.stack(x_train)
    y = torch.tensor(y_train)
    # statistics = y.sum(dim=0)/y.sum(dim=0).sum()
    statistics = y.sum(dim=0)
    statistics = statistics.numpy()
    
    return x, y, statistics

# train_loader, test_loader = Get_Dataloader(param, args.bin_fn, 8, pin_memory=False, random_validation=True)

train_loader, test_loader = Get_Dataloader_augmentation(param, args.bin_fn, 12, pin_memory=False, 
                        platform=args.platform, random_validation=True , augmentation=0)


x_test, y_test, test_statistics = get_features_from_encoder(test_loader)

label = np.argmax(y_test,axis=1)
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

print("Training T-SNE!")
model = TSNE(n_components=2, perplexity=21, n_iter=500)

X_2d = model.fit_transform(x_test)

from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))

target_ids = range(21)

for i in range(21):
    plt.scatter(X_2d[label==i,0], X_2d[label==i,1], label=i)
plt.legend()
plt.savefig('T_SNE.png')
plt.show()