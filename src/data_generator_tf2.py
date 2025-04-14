import numpy as np
import cv2

from tensorflow import keras
from src.sampler import augment_sample, labels2output_map
import matplotlib.pyplot as plt

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

    # # Adding CED layer
    # # Applying Canny with augmentation, visualizing script is not here for this.
    # def __data_generation(self, indexes):
    #     'Generates data containing batch_size samples with Canny Edge Concatenation'

    #     # Change shape to 4 channels (RGB + Canny)
    #     X = np.empty((self.batch_size, self.dim, self.dim, 4), dtype=np.float32)
    #     y = np.empty((self.batch_size, self.dim // self.stride, self.dim // self.stride, 9), dtype=np.float32)

    #     for i, idx in enumerate(indexes):
    #         # Augment sample
    #         XX, llp, ptslist = augment_sample(self.data[idx][0], self.data[idx][1], self.dim)

    #         # Apply Canny Edge Detection
    #         gray = cv2.cvtColor((XX * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    #         edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    #         edges = edges.astype(np.float32) / 255.0  # Normalize to 0-1
    #         edges = np.expand_dims(edges, axis=-1)    # Add channel dimension

    #         # Concatenate RGB image with edge map
    #         X[i] = np.concatenate([XX * self.OutputScale, edges], axis=-1)

    #         # Generate corresponding label
    #         YY = labels2output_map(llp, ptslist, self.dim, self.stride, alfa=0.5)
    #         y[i] = YY   

    #     return X, y

    # Applying Canny without augmentation.
    # def __data_generation(self, indexes):
    #     'Generates data containing batch_size samples with Canny Edge Concatenation, no data augmentation'

    #     X = np.empty((self.batch_size, self.dim, self.dim, 4), dtype=np.float32)
    #     y = np.empty((self.batch_size, self.dim // self.stride, self.dim // self.stride, 9), dtype=np.float32)

    #     for i, idx in enumerate(indexes):
    #         # Get raw image and label
    #         img = self.data[idx][0]
    #         label = self.data[idx][1]

    #         # Resize image to (dim x dim)
    #         img_resized = cv2.resize(img, (self.dim, self.dim))
    #         img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0  # Normalize RGB

    #         # Apply Canny to grayscale image
    #         gray = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    #         edges = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
    #         edges = np.expand_dims(edges, axis=-1)  # Shape: (dim, dim, 1)

    #         # Concatenate RGB + edge map => shape: (dim, dim, 4)
    #         X[i] = np.concatenate([img_rgb * self.OutputScale, edges], axis=-1)


    #         # Convert edges to 3-channel for display
    #         edge_display = np.repeat(edges, 3, axis=-1)

    #         # Create a white space (dim x 10 x 3)
    #         space = np.ones((self.dim, 10, 3), dtype=np.float32)

    #         # Combine: original | white gap | edges
    #         combined = np.hstack((img_rgb, space, edge_display))

    #         plt.figure(figsize=(10, 5))
    #         plt.imshow(combined)
    #         plt.title('Original Image and Canny Edge with Space Between')
    #         plt.axis('off')
    #         plt.show()

    #         # Adjust label positions (scaled by dim if needed)
    #         # YY = labels2output_map(label, None, self.dim, self.stride, alfa=0.5)
    #         # y[i] = YY

    #     return X, y

    # # Applying Canny without augmentation, visualizing with colors.
    # def __data_generation(self, indexes):
    #     'Generates data containing batch_size samples with Canny Edge Concatenation, no data augmentation'

    #     X = np.empty((self.batch_size, self.dim, self.dim, 4), dtype=np.float32)
    #     y = np.empty((self.batch_size, self.dim // self.stride, self.dim // self.stride, 9), dtype=np.float32)

    #     for i, idx in enumerate(indexes):
    #         # Get raw image and label
    #         img = self.data[idx][0]
    #         label = self.data[idx][1]

    #         # Resize image to (dim x dim)
    #         img_resized = cv2.resize(img, (self.dim, self.dim))
    #         img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0  # Normalize RGB

    #         # Apply Canny to grayscale image
    #         gray = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    #         edges = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
    #         edges = np.expand_dims(edges, axis=-1)  # Shape: (dim, dim, 1)

    #         # Enhance RGB image by overlaying Canny edges
    #         enhanced_rgb = img_rgb + (edges * self.OutputScale)
    #         enhanced_rgb = np.clip(enhanced_rgb, 0.0, 1.0)  # Ensure valid range

    #         # Concatenate edge-enhanced RGB and edge as separate channel
    #         X[i] = np.concatenate([enhanced_rgb, edges], axis=-1)  # Shape: (dim, dim, 4)

    #         # Convert enhanced RGB to display format
    #         enhanced_display = (enhanced_rgb * 255).astype(np.uint8)

    #         # Create a white space (dim x 10 x 3)
    #         space = np.ones((self.dim, 10, 3), dtype=np.uint8) * 255

    #         # Convert edge_display to uint8 for visualization
    #         edge_display = (np.repeat(edges, 3, axis=-1) * 255).astype(np.uint8)

    #         # Combine: original | space | enhanced | space | edges
    #         combined = np.hstack((
    #             (img_rgb * 255).astype(np.uint8),
    #             space,
    #             enhanced_display,
    #             space,
    #             edge_display
    #         ))

    #         plt.figure(figsize=(12, 5))
    #         plt.imshow(combined)
    #         plt.title('Original | Edge-Enhanced | Canny Edges')
    #         plt.axis('off')
    #         plt.show()

    #         # Adjust label positions (scaled by dim if needed)
    #         # YY = labels2output_map(label, None, self.dim, self.stride, alfa=0.5)
    #         # y[i] = YY

    #     return X, y

    # Tested CED implementation, without alteration in model architecture.
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples with Canny Edge blended into RGB (keeps 3 channels)'

        # Keep shape at 3 channels (RGB with edge info blended in)
        X = np.empty((self.batch_size, self.dim, self.dim, 3), dtype=np.float32)
        y = np.empty((self.batch_size, self.dim // self.stride, self.dim // self.stride, 9), dtype=np.float32)

        for i, idx in enumerate(indexes):
            # Augment sample
            XX, llp, ptslist = augment_sample(self.data[idx][0], self.data[idx][1], self.dim)

            # Apply Canny Edge Detection
            gray = cv2.cvtColor((XX * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, threshold1=100, threshold2=200)
            edges = edges.astype(np.float32) / 255.0  # Normalize to 0-1
            edges = np.expand_dims(edges, axis=-1)    # Shape: (H, W, 1)

            # Option 1: Blend edges across all 3 RGB channels
            XX_blended = XX + 0.1 * np.repeat(edges, 3, axis=-1)  # Light blend
            XX_blended = np.clip(XX_blended, 0, 1)  # Keep values in range

            # Store input with edge-enhanced RGB (shape still (H, W, 3))
            X[i] = XX_blended * self.OutputScale

            # Generate corresponding label
            YY = labels2output_map(llp, ptslist, self.dim, self.stride, alfa=0.5)
            y[i] = YY

        return X, y