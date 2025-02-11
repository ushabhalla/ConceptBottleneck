import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import pdb
from dataset import load_non_image_data, load_data_from_different_splits, PytorchImagesDataset, \
    get_image_cache_for_split
from torchvision import transforms

def get_data_dict_from_dataloader(data, C_cols):
        # Retrieves the relevant data from dataloader and store into a dict
        X = data['image']  # X
        y = data['y']  # y
        C_feats = data['C_feats']
        C_feats_not_nan = data['C_feats_not_nan']

        # wrap them in Variable
        X = Variable(X.float())
        y = Variable(y.float())
        if len(C_cols) > 0:
            C_feats = Variable(C_feats.float())
            C_feats_not_nan = Variable(C_feats_not_nan.float())

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
C_cols=['xrosfm', 'xrscfm', 'xrjsm', 'xrostm', 'xrsctm', 'xrosfl', 'xrscfl', 'xrjsl', 'xrostl', 'xrsctl']
y_cols=['xrkl']
dataloaders, datasets, dataset_sizes = load_data_from_different_splits(batch_size=1, C_cols=C_cols, y_cols=y_cols, zscore_C=True, zscore_Y=False, data_proportion=1.0,
    shuffle_Cs=False, merge_klg_01=True, max_horizontal_translation=0.1, max_vertical_translation=0.1, sampling_strategy='uniform', augment='random_translation')
transform = transforms.ToPILImage(mode='RGB')


# concept_class_dict = {}
# for c in C_cols:
#     concept_class_dict[c] = LinearRegression()
classifier = LogisticRegression(n_jobs=3, max_iter=1000)
classifier2 = LinearRegression()

# pdb.set_trace()
for epoch in range(100):
    for phase in ['train', 'val']:
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for data in dataloaders[phase]:
                data_dict = get_data_dict_from_dataloader(data, C_cols)

                inputs = preprocess(transform(data_dict['inputs']['image'].squeeze()))
                inputs = inputs.unsqueeze(0)
                # labels = data_dict['labels']['C_feats']
                y_labels = data_dict['labels']['y']

                features = model.encode_image(inputs.cuda())

                all_features.append(features.squeeze().cpu().numpy())
                all_labels.append(y_labels.squeeze().cpu().numpy())
            
            all_features = np.array(all_features)
            all_labels = np.array(all_labels)

            # Perform logistic regression
            if phase == 'train':
                classifier.fit(all_features, all_labels)
                classifier2.fit(all_features, all_labels)
                # for i, c in enumerate(C_cols):
                #     classifier = concept_class_dict[c]
                #     classifier.fit(all_features, all_labels[:,i])

                # classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
                # classifier.fit(all_features, all_labels)

            # Evaluate using the logistic regression classifier
            else:
                # scores = []
                acc = classifier.score(all_features, all_labels)
                score = classifier2.score(all_features, all_labels)

                if epoch % 10 == 0:
                    p = classifier.predict(all_features)
                    print(p)
                    p = classifier2.predict(all_features)
                    print(p)
                    print(all_labels)
                # for i, c in enumerate(C_cols):
                #     classifier = concept_class_dict[c]
                #     score = classifier.score(all_features, all_labels[:,i])
                #     scores.append(score)

                print("Epoch:", str(epoch)+ ",", "Score:", score, "Acc: ", acc)
