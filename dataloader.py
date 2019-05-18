import os
import yaml
import argparse
import numpy as np 
import hickle as hkl
import matplotlib.pyplot as plt

from glob import glob 
from tabulate import tabulate

class Dataloader:
    def __init__(self, config):
        self.paths = sorted(glob(os.path.join(config["path"], "*hkl")))[:10]
        self.labels = [path.split("/")[-1][:-4] for path in self.paths]
        self.num_classes = len(self.paths)
        self.imsize = config["train"]["x-dim"]
        self.imchannels = config["train"]["imchannels"]

    def sample(self, batch_size):
        images, labels = [], []
        cls_size = batch_size // self.num_classes if batch_size > self.num_classes else 1
        for i, path in enumerate(self.paths):
            data = hkl.load(path)
            rand_sample = data[np.random.choice(np.arange(len(data)), cls_size, replace=False)]
            # fixed_sample = data[np.arange(cls_size)]
            images.extend(self.preprocess(rand_sample))
            labels.extend([i] * cls_size)

        return self.__shuffle(images, labels)

    def preprocess(self, images):
        images = images.reshape(-1, self.imsize, self.imsize, self.imchannels)
        return ((255-images.astype(np.float32)) - 127.5) / 127.5
            
    def __shuffle(self, images, labels):
        combined = list(zip(images, labels))
        np.random.shuffle(combined)
        images, labels = zip(*combined)
        return np.array(images), np.array(labels)

    def stats(self):
        freq = np.array([hkl.load(path).shape[0] for path in self.paths])

        print(tabulate([
                ['TOTAL', f"{int(freq.sum()):,}"], 
                ['MAX',   f"{int(freq.max()):,}"],
                ['MIN',   f"{int(freq.min()):,}"],
                ['MEAN',  f"{freq.mean():,.2f}"],
                ['STDEV', f"{freq.std():,.2f}"], 
            ], headers=['FUNC', 'VAL'], tablefmt='orgtbl'))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataloader Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    loader = Dataloader(config)
    loader.stats()

        