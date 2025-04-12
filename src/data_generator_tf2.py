import numpy as np
import cv2

from tensorflow import keras
from src.sampler import augment_sample, labels2output_map
#
#  Creates ALPR Data Generator
#


class ALPRDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, dim =  208, stride = 16, shuffle=True, OutputScale = 1.0):
        'Initialization'
        self.dim = dim
        self.stride = stride
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.OutputScale = OutputScale
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
		
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch. Pads training data to be a multiple of batch size'
#        self.indexes = list(np.arange(0, len(self.data), 1))
        self.indexes = list(np.arange(0, len(self.data), 1)) 
        self.indexes += list(np.random.choice(self.indexes, self.batch_size - len(self.data) % self.batch_size))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    # def __data_generation(self, indexes):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

    #     X = np.empty((self.batch_size, self.dim, self.dim, 3))
    #     y = np.empty((self.batch_size, self.dim//self.stride, self.dim//self.stride, 9))
    #     # Generate data
    #     for i, idx in enumerate(indexes):
    #         # Store sample
    #         XX, llp, ptslist = augment_sample(self.data[idx][0], self.data[idx][1], self.dim)
    #         YY = labels2output_map(llp, ptslist, self.dim, self.stride, alfa = 0.5)
    #         X[i,] = XX*self.OutputScale
    #         y[i,] = YY
    #     return X, y

     # Adding CED layer
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples with Canny Edge Concatenation'

        # Change shape to 4 channels (RGB + Canny)
        X = np.empty((self.batch_size, self.dim, self.dim, 4), dtype=np.float32)
        y = np.empty((self.batch_size, self.dim // self.stride, self.dim // self.stride, 9), dtype=np.float32)

        for i, idx in enumerate(indexes):
            # Augment sample
            XX, llp, ptslist = augment_sample(self.data[idx][0], self.data[idx][1], self.dim)

            # Apply Canny Edge Detection
            gray = cv2.cvtColor((XX * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, threshold1=100, threshold2=200)
            edges = edges.astype(np.float32) / 255.0  # Normalize to 0-1
            edges = np.expand_dims(edges, axis=-1)    # Add channel dimension

            # Concatenate RGB image with edge map
            X[i] = np.concatenate([XX * self.OutputScale, edges], axis=-1)

            # Generate corresponding label
            YY = labels2output_map(llp, ptslist, self.dim, self.stride, alfa=0.5)
            y[i] = YY

        return X, y
