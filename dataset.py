import json
import glob
import numpy as np
import math
import torch
import cv2
import os
from PIL import Image

class_to_idx = {'bishop': 0, 'king': 1, 'knight': 2, 'pawn': 3, 'queen': 4, 'rook': 5}
idx_to_class = {v: k for k, v in class_to_idx.items()}

class mixed_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, transform=None):
        self.transform = transform
        self.data_path = data_path

        if(train):
            data_path = os.path.join(data_path, 'train.json')
        else:
            data_path = os.path.join(data_path, 'test.json')

        with open(data_path, 'r') as f:
            self.data = json.load(f)

        encoding = torch.eye(len(class_to_idx))
        for i in range(len(self.data)):
            image = Image.open(os.path.join(self.data_path, self.data[i]['Depth_image_blur']))
            image = image.resize((224, 224))
            image = image.convert("RGB")
            self.data[i]['image'] = image
            # self.data[i]['type'] = encoding[self.data[i]['type']]
        self.len = len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]['image']
        if(self.transform):
            image = self.transform(image)
        return image, self.data[index]['type']
    
    def __len__(self):
        return self.len




def main():
    train = []
    test = []

    # -----------------------------
    
    print("Loading real data...")
    # Open './data/gs_data/' and iterate over all folders
    real_data = {}
    for folder in glob.glob('./data/gs_data/*'):
        piece = folder.split('/')[-1]
        for file in glob.glob(folder+'/*.png'):
            if(piece not in real_data.keys()):
                real_data[piece] = []
            real_data[piece].append({
                'Depth_image_blur': os.path.join('gs_data/', folder.split('/')[-1], file.split('/')[-1],),
                'type': class_to_idx[piece], })

    min_len  = math.inf
    for key in real_data.keys():
        print("Real: {}, Count: {}".format(key, len(real_data[key])))
        if(len(real_data[key]) < min_len):
            min_len = len(real_data[key])

    test_size = 20
    for key in real_data.keys():
        shuffle = np.random.permutation(len(real_data[key]))
        for i in range(test_size):
            test.append(real_data[key][shuffle[i]])
        for i in range(test_size, min_len):
            train.append(real_data[key][shuffle[i]])
    
    print("Train: {}, Test: {}".format(len(train), len(test)))

    # -----------------------------

    # num_synthetic = (min_len - test_size) * 1
    # synth = {}
    # print("_"*50)
    # print("Loading synthetic data...")


    # for file in glob.glob('./data/data_mod/*.json'):
    #     with open(file, 'r') as f:
    #         synth[file.split('/')[-1]] = json.load(f)

    # min_len  = math.inf
    # for key in synth.keys():
    #     print("Synth: {}, Count: {}".format(key, len(synth[key])))
    #     if(len(synth[key]) < min_len):
    #         min_len = len(synth[key])

    # for key in synth.keys():
    #     class_num = class_to_idx[key.split('.')[0]]
    #     shuffle = np.random.permutation(len(synth[key]))
    #     for i in range(num_synthetic):
    #         train.append({
    #             'Depth_image_blur': os.path.join('data_mod/', synth[key][shuffle[i]]['Deapth_image_blur']),
    #             'type': class_num, })

    # print("Train: {}, Test: {}".format(len(train), len(test)))

    # -----------------------------
    
    with open('./data/train.json', 'w') as f:
        json.dump(train, f, indent=4)
    with open('./data/test.json', 'w') as f:
        json.dump(test, f, indent=4)


if __name__ == '__main__':
    main()