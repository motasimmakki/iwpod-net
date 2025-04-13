from src.data_generator_tf2 import ALPRDataGenerator
from src.utils import image_files_from_folder
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
from src.label import readShapes, Shape
from os.path import isfile, splitext

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr'		,'--train-dir'			,type=str   , default = 'train_dir'			,help='Input data directory for training')
    args = parser.parse_args()
     
    #
    #  Loads training data
    #
    train_dir = args.train_dir
    print ('Loading training data...')
    Files = image_files_from_folder(train_dir)

    #
    #  Defines size of "fake" tiny LP annotation, used when no LP is present
    #				
    fakepts = np.array([[0.5, 0.5001, 0.5001, 0.5], [0.5, 0.5, 0.5001, 0.5001]])
    fakeshape = Shape(fakepts)
    Data = []
    ann_files = 0
    for file in Files:
        labfile = splitext(file)[0] + '.txt'
        if isfile(labfile):
            ann_files += 1
            L = readShapes(labfile)
            I = cv2.imread(file)
            if len(L) > 0:
                Data.append([I, L])
        else:
            #
            #  Appends a "fake"  plate to images without any annotation
            #
            I = cv2.imread(file)
            Data.append(  [I, [fakeshape] ]  )

    print ('%d images with labels found' % len(Data) )
    print ('%d annotation files found' % ann_files )

    # Create generator instance
    generator = ALPRDataGenerator(Data, batch_size=4, dim=208, stride=16, shuffle=False)

    # Get the first batch
    X_batch, y_batch = generator[0]  # X_batch shape: (batch_size, 208, 208, 4)

    # Plot the batch
    for i in range(min(4, len(X_batch))):
        rgb_image = X_batch[i, :, :, :3]  # RGB
        edge_image = X_batch[i, :, :, 3]  # Canny edge

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image.astype('uint8'))
        plt.title('RGB Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(edge_image, cmap='gray')
        plt.title('Canny Edge')
        plt.axis('off')

        plt.suptitle(f"Sample {i+1}")
        plt.show()
