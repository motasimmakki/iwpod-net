import numpy as np

from tensorflow import keras
from src.sampler import augment_sample, labels2output_map
#
#  Creates ALPR Data Generator
#
from src.iou_utils import compute_iou, extract_box_from_output


class ALPRDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, dim =  208, stride = 16, shuffle=True, OutputScale = 1.0, return_shapes=False):
        'Initialization'
        self.dim = dim
        self.stride = stride
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.OutputScale = OutputScale
        self.on_epoch_end()
        self.return_shapes = return_shapes

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y, shapes = self.__data_generation(indexes)

        if self.return_shapes:
            return X, y, shapes
        else:
            return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch. Pads training data to be a multiple of batch size'
#        self.indexes = list(np.arange(0, len(self.data), 1))
        self.indexes = list(np.arange(0, len(self.data), 1)) 
        self.indexes += list(np.random.choice(self.indexes, self.batch_size - len(self.data) % self.batch_size))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, self.dim, self.dim, 3))
        y = np.empty((self.batch_size, self.dim//self.stride, self.dim//self.stride, 9))
        shape_list = []

        for i, idx in enumerate(indexes):
            XX, llp, ptslist = augment_sample(self.data[idx][0], self.data[idx][1], self.dim)
            YY = labels2output_map(llp, ptslist, self.dim, self.stride, alfa=0.5)
            X[i,] = XX * self.OutputScale
            y[i,] = YY
            shape_list.append(ptslist)  # Save original shapes for IoU

        return X, y, shape_list

class IoUCallback(tf.keras.callbacks.Callback):
    def __init__(self, data_gen, num_batches=1):
        self.data_gen = data_gen
        self.num_batches = num_batches

    def on_train_batch_end(self, batch, logs=None):
        # Evaluate IoU on a few samples per batch
        for _ in range(self.num_batches):
            X_batch, y_true, shapes = self.data_gen.__getitem__(batch % len(self.data_gen))
            y_pred = self.model.predict_on_batch(X_batch)

            batch_ious = []
            for i in range(len(X_batch)):
                gt_pts = shapes[i][0].pts  # first LP shape
                pred_box = extract_box_from_output(y_pred[i])  # implement this function

                # Get gt bounding box from corner points
                x_coords = gt_pts[0] * self.data_gen.dim
                y_coords = gt_pts[1] * self.data_gen.dim
                gt_box = [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]

                iou = compute_iou(gt_box, pred_box)
                batch_ious.append(iou)

            avg_iou = np.mean(batch_ious)
            print(f"[IoUCallback] Batch {batch}: Mean IoU = {avg_iou:.4f}")
