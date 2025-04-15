# from src.data_generator_tf2 import ALPRDataGenerator
# from src.utils import image_files_from_folder
# import matplotlib.pyplot as plt
# import argparse
# import numpy as np
# import cv2
# from src.label import readShapes, Shape
# from os.path import isfile, splitext

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-tr'		,'--train-dir'			,type=str   , default = 'train_dir'			,help='Input data directory for training')
#     args = parser.parse_args()
     
#     #
#     #  Loads training data
#     #
#     train_dir = args.train_dir
#     print ('Loading training data...')
#     Files = image_files_from_folder(train_dir)

#     #
#     #  Defines size of "fake" tiny LP annotation, used when no LP is present
#     #				
#     fakepts = np.array([[0.5, 0.5001, 0.5001, 0.5], [0.5, 0.5, 0.5001, 0.5001]])
#     fakeshape = Shape(fakepts)
#     Data = []
#     ann_files = 0
#     for file in Files:
#         labfile = splitext(file)[0] + '.txt'
#         if isfile(labfile):
#             ann_files += 1
#             L = readShapes(labfile)
#             I = cv2.imread(file)
#             if len(L) > 0:
#                 Data.append([I, L])
#         else:
#             #
#             #  Appends a "fake"  plate to images without any annotation
#             #
#             I = cv2.imread(file)
#             Data.append(  [I, [fakeshape] ]  )

#     print ('%d images with labels found' % len(Data) )
#     print ('%d annotation files found' % ann_files )

#     # Create generator instance
#     generator = ALPRDataGenerator(Data, batch_size=4, dim=208, stride=16, shuffle=False)

#     # Get the first batch
#     X_batch, y_batch = generator[0]  # X_batch shape: (batch_size, 208, 208, 4)

#     # # Plot the batch
#     # for i in range(min(4, len(X_batch))):
#     #     rgb_image = X_batch[i, :, :, :3]  # RGB
#     #     edge_image = X_batch[i, :, :, 3]  # Canny edge

#     #     plt.figure(figsize=(8, 4))

#     #     plt.subplot(1, 2, 1)
#     #     plt.imshow(rgb_image.astype('uint8'))
#     #     plt.title('RGB Image')
#     #     plt.axis('off')

#     #     plt.subplot(1, 2, 2)
#     #     plt.imshow(edge_image, cmap='gray')
#     #     plt.title('Canny Edge')
#     #     plt.axis('off')

#     #     plt.suptitle(f"Sample {i+1}")
#     #     plt.show()


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
    parser.add_argument('-tr', '--train-dir', type=str, default='train_dir', help='Input data directory for training')
    args = parser.parse_args()

    # Loads training data
    train_dir = args.train_dir
    print('Loading training data...')
    Files = image_files_from_folder(train_dir)

    # Define "fake" tiny LP annotation, used when no LP is present
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
            I = cv2.imread(file)
            Data.append([I, [fakeshape]])

    print('%d images with labels found' % len(Data))
    print('%d annotation files found' % ann_files)

    # Create generator instance (Canny edge blended into RGB channels)
    generator = ALPRDataGenerator(Data, batch_size=4, dim=208, stride=16, shuffle=False)

    # Get the first batch (shape: (batch_size, 208, 208, 3))
    X_batch, y_batch = generator[0]

    # Visualize blended images
    for i in range(min(4, len(X_batch))):
        # Blended image (from generator, values in [0, 1])
        blended_img = (X_batch[i] * 255).astype('uint8')

        # Original image (from Data — note: it may be BGR from cv2)
        original_img = cv2.cvtColor(Data[i][0], cv2.COLOR_BGR2RGB)

        # Resize original to match blended (208x208)
        original_img_resized = cv2.resize(original_img, (blended_img.shape[1], blended_img.shape[0]))

        # Plot side-by-side
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original_img_resized)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(blended_img)
        plt.title('Blended with Canny Edges')
        plt.axis('off')

        plt.suptitle(f'Sample {i + 1}', fontsize=14)
        plt.tight_layout(pad=3.0)
        plt.show()
