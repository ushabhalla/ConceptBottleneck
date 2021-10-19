import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from OAI.dataset import load_non_image_data, load_data_from_different_splits, PytorchImagesDataset, \
    get_image_cache_for_split

def get_data_dict_from_dataloader(self, data):
        # Retrieves the relevant data from dataloader and store into a dict
        X = data['image']  # X
        y = data['y']  # y
        C_feats = data['C_feats']
        C_feats_not_nan = data['C_feats_not_nan']

        # wrap them in Variable
        X = Variable(X.float().cuda())
        y = Variable(y.float().cuda())
        if len(self.C_cols) > 0:
            C_feats = Variable(C_feats.float().cuda())
            C_feats_not_nan = Variable(C_feats_not_nan.float().cuda())

        inputs = { 'image': X }
        labels = { 'y': y,
                   'C_feats': C_feats,
                   'C_feats_not_nan': C_feats_not_nan }

        data_dict = {
            'inputs': inputs,
            'labels': labels,
        }
        return data_dict

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)
    return all_features, all_labels

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)

# Load the dataset
# root = os.path.expanduser("~/.cache")

dataloaders, datasets, dataset_sizes = load_data_from_different_splits(batch_size=128, C_cols=, y_cols=, zscore_C=, zscore_Y=, data_proportion=,
    shuffle_Cs=, merge_klg_01=, max_horizontal_translation=, max_vertical_translation=))


for epoch in range(10):
    for phase in ['train', 'val']:
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for data in dataloaders[phase]:
                data_dict = get_data_dict_from_dataloader(data)
                inputs = data_dict['inputs']
                labels = data_dict['labels']

                features = model.encode_image(inputs.to(device))
                all_features.append(features)
                all_labels.append(labels)

            # Perform logistic regression
            if phase == 'train':
                classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
                classifier.fit(all_features, all_labels)

            # Evaluate using the logistic regression classifier
            else:
                predictions = classifier.predict(all_features)
                accuracy = np.mean((all_features == predictions).astype(np.float)) * 100.
                print("Epoch:", str(epoch), f"Accuracy = {accuracy:.3f}")
