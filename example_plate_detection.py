# import numpy as np
# from src.keras_utils import load_model
# import cv2
# from src.keras_utils import  detect_lp_width
# from src.utils 					import  im2single
# from src.drawing_utils			import draw_losangle
# import argparse
# from src.qiou_utils import compute_qiou, load_ground_truth
# import os



# if __name__ == '__main__':

# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('-i' 		,'--image'			,type=str   , default = 'images\\example_aolp_fullimage.jpg'		,help='Input Image')
# 	parser.add_argument('-v' 		,'--vtype'			,type=str   , default = 'fullimage'		,help = 'Image type (car, truck, bus, bike or fullimage)')
# 	parser.add_argument('-t' 		,'--lp_threshold'	,type=float   , default = 0.35		,help = 'Detection Threshold')

# 	#parser.add_argument('-tr'		,'--train-dir'		,type=str   , required=True		,help='Input data directory for training')
# 	args = parser.parse_args()

# 	#
# 	#  Parameters of the method
# 	#
# 	#lp_threshold = 0.35 # detection threshold
# 	lp_threshold = args.lp_threshold
# 	ocr_input_size = [80, 240] # desired LP size (width x height)
    
# 	#
# 	#  Loads network and weights
# 	#
# 	iwpod_net = load_model('weights/iwpod_net')
# 	# iwpod_net = load_model('weights/trained_iwpodnet_aolp')
    

# 	#
# 	#  Loads image with vehicle crop or full image with vehicle(s) roughly framed.
# 	#  You can use your favorite object detector here (a fine-tuned version of Yolo-v3 was
# 	#  used in the paper)
# 	#
    
# 	#
# 	#  Also inform the vehicle type:
# 	#  'car', 'bus', 'truck' 
# 	#  'bike' 
# 	#  'fullimage' 
# 	#
# 	#

# 	Ivehicle = cv2.imread(args.image)
# 	vtype = args.vtype
# 	iwh = np.array(Ivehicle.shape[1::-1],dtype=float).reshape((2,1))

# 	if (vtype in ['car', 'bus', 'truck']):
# 		#
# 		#  Defines crops for car, bus, truck based on input aspect ratio (see paper)
# 		#
# 		ASPECTRATIO = max(1, min(2.75, 1.0*Ivehicle.shape[1]/Ivehicle.shape[0]))  # width over height
# 		WPODResolution = 256# faster execution
# 		lp_output_resolution = tuple(ocr_input_size[::-1])

# 	elif  vtype == 'fullimage':
# 		#
# 		#  Defines crop if vehicles were not cropped 
# 		#
# 		ASPECTRATIO = 1 
# 		WPODResolution = 480 # larger if full image is used directly
# 		lp_output_resolution =  tuple(ocr_input_size[::-1])
# 	else:
# 		#
# 		#  Defines crop for motorbike  
# 		#
# 		ASPECTRATIO = 1.0 # width over height
# 		WPODResolution = 208
# 		lp_output_resolution = (int(1.5*ocr_input_size[0]), ocr_input_size[0]) # for bikes, the LP aspect ratio is lower

# 	#
# 	#  Runs IWPOD-NET. Returns list of LP data and cropped LP images
# 	#
# 	Llp, LlpImgs,_ = detect_lp_width(iwpod_net, im2single(Ivehicle), WPODResolution*ASPECTRATIO, 2**4, lp_output_resolution, lp_threshold)
# 	for i, img in enumerate(LlpImgs):
# 		#
# 		#  Draws LP quadrilateral in input image
# 		#
# 		pts = Llp[i].pts * iwh

# 		# ********Computing qIOU here************
# 		# Extract GT path (e.g., from ./AOLP_Dataset_splited/test/3_RP.jpg → ./AOLP_Dataset_splited/test/3_RP.txt)
# 		txt_path = os.path.splitext(args.image)[0] + '.txt'

# 		# Load GT
# 		gt_pts = load_ground_truth(txt_path, Ivehicle.shape)
        
# 		# Compute Q-IoU
# 		pts = pts.T
# 		q_iou = compute_qiou(pts, gt_pts)
# 		print(f'Q-IoU for {args.image}: {q_iou:.4f}')

# 		# print("Predicted Points:\n", pts)
# 		# print("Ground Truth Points:\n", gt_pts)

# 		# # Optionally draw GT box
# 		# for i in range(4):
# 		# 	pt1 = tuple(gt_pts[i % 4].astype(int))
# 		# 	pt2 = tuple(gt_pts[(i + 1) % 4].astype(int))
# 		# 	cv2.line(Ivehicle, pt1, pt2, (0, 255, 0), 2)  # green for GT
# 		# ------------------------------------------

# 		pts = pts.T
# 		draw_losangle(Ivehicle, pts, color = (0,0,255.), thickness = 2)
# 		#
# 		#  Shows each detected LP
# 		#
# 		cv2.imshow('Rectified plate %d'%i, img )
# 	#
# 	#  Shows original image with deteced plates (quadrilateral)
# 	#
# 	cv2.imshow('Image and LPs', Ivehicle )
# 	cv2.waitKey()
# 	cv2.destroyAllWindows()
    

 

import numpy as np
from src.keras_utils import load_model
import cv2
from src.keras_utils import detect_lp_width
from src.utils import im2single
from src.drawing_utils import draw_losangle
import argparse
from src.qiou_utils import compute_qiou, load_ground_truth
import os
import glob

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default=None, help='Input Image')
    parser.add_argument('-v', '--vtype', type=str, default='fullimage', help='Image type (car, truck, bus, bike or fullimage)')
    parser.add_argument('-t', '--lp_threshold', type=float, default=0.35, help='Detection Threshold')
    parser.add_argument('--images', type=str, default=None, help='Directory to test all images and get mean Q-IoU')
    args = parser.parse_args()

    lp_threshold = args.lp_threshold
    ocr_input_size = [80, 240]  # width x height

    # iwpod_net = load_model('weights/iwpod_net')
    # iwpod_net = load_model('weights/iwpodnet_aolp_200')
    iwpod_net = load_model('weights/iwpodnet_canny_200')
    # iwpod_net = load_model('weights/trained_iwpodnet_200')
    # iwpod_net = load_model('weights/trained_iwpodnet_aolp')
    # iwpod_net = load_model('weights/canny_iwpodnet_10')
    # iwpod_net = load_model('weights/trained_iwpodnet_10_Canny')
    # iwpod_net = load_model('weights/trained_iwpodnet_canny')
    # iwpod_net = load_model('weights/trained_iwpodnet_canny-02')

    image_paths = []
    if args.images:
        image_paths = glob.glob(os.path.join(args.images, '*.jpg'))
    elif args.image:
        image_paths = [args.image]
    else:
        raise ValueError("You must specify either --image or --images")

    qiou_list = []

    for img_path in image_paths:
        Ivehicle = cv2.imread(img_path)
        if Ivehicle is None:
            print(f"Failed to read image: {img_path}")
            continue

        vtype = args.vtype
        iwh = np.array(Ivehicle.shape[1::-1], dtype=float).reshape((2, 1))

        if vtype in ['car', 'bus', 'truck']:
            ASPECTRATIO = max(1, min(2.75, 1.0 * Ivehicle.shape[1] / Ivehicle.shape[0]))
            WPODResolution = 256
            lp_output_resolution = tuple(ocr_input_size[::-1])
        elif vtype == 'fullimage':
            ASPECTRATIO = 1
            WPODResolution = 480
            lp_output_resolution = tuple(ocr_input_size[::-1])
        else:
            ASPECTRATIO = 1.0
            WPODResolution = 208
            lp_output_resolution = (int(1.5 * ocr_input_size[0]), ocr_input_size[0])

        Llp, LlpImgs, _ = detect_lp_width(iwpod_net, im2single(Ivehicle), WPODResolution * ASPECTRATIO, 2 ** 4, lp_output_resolution, lp_threshold)

        for i, img in enumerate(LlpImgs):
            pts = Llp[i].pts * iwh
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if not os.path.exists(txt_path):
                print(f"GT file not found for: {img_path}")
                continue

            gt_pts = load_ground_truth(txt_path, Ivehicle.shape)
            pts = pts.T
            q_iou = compute_qiou(pts, gt_pts)
            print(f"Q-IoU for {img_path}: {q_iou:.4f}")
            qiou_list.append(q_iou)
            
            
            # Optionally draw GT box
            for i in range(4):
                pt1 = tuple(gt_pts[i % 4].astype(int))
                pt2 = tuple(gt_pts[(i + 1) % 4].astype(int))
                cv2.line(Ivehicle, pt1, pt2, (0, 255, 0), 2)  # green for GT
            
            pts = pts.T
            draw_losangle(Ivehicle, pts, color=(0, 0, 255), thickness=2)
            cv2.imshow(f'Rectified plate {i}', img)

        cv2.imshow('Image and LPs', Ivehicle)
        # cv2.waitKey(1)  # show quickly, or adjust as needed
        if len(qiou_list) > 1:
            cv2.waitKey(1)  # show quickly, or adjust as needed
        else:
            cv2.waitKey()  # show quickly, or adjust as needed 

    cv2.destroyAllWindows()

    if qiou_list:
        mean_qiou = sum(qiou_list) / len(qiou_list)
        print(f"\nMean Q-IoU over {len(qiou_list)} samples: {mean_qiou:.4f}")
    else:
        print("No valid Q-IoUs computed.")
