# Load the model
import numpy as np
import cv2
from keras.models import load_model
from skimage.transform import resize
from PIL import Image
from skimage.io import imread
from model import metric

model_phase2 = load_model('/home/marcassoli/Bureau/Stage 2A/phase-01-training/model_checkpoints/model_phase_2.hdf5',custom_objects={'metric': metric})
#file_model = h5py.File('/home/marcassoli/Bureau/Stage 2A/phase-01-training/model_checkpoints/model_phase_2.hdf5','r+')
type(model_phase2)

    # call the model
q = [4.0, 12.0, 2.0]
filter1 = [[0, 0, 0, 0, 0],
           [0, -1, 2, -1, 0],
           [0, 2, -4, 2, 0],
           [0, -1, 2, -1, 0],
           [0, 0, 0, 0, 0]]
filter2 = [[-1, 2, -2, 2, -1],
           [2, -6, 8, -6, 2],
           [-2, 8, -12, 8, -2],
           [2, -6, 8, -6, 2],
           [-1, 2, -2, 2, -1]]
filter3 = [[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 1, -2, 1, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]]


filter1 = np.asarray(filter1, dtype=float) / q[0]
filter2 = np.asarray(filter2, dtype=float) / q[1]
filter3 = np.asarray(filter3, dtype=float) / q[2]
    
filters = filter1+filter2+filter3



def pre_process_input(img):    
    processed_image = cv2.filter2D(img,-1,filters)
    return processed_image


def load_input(input_img_path):
  
  # load the image 
  org_img = imread(input_img_path)  

  # filter the image 
  filtered_img  = pre_process_input(org_img.copy())

  # resize input
  org_img_res = np.array([resize(org_img, (512, 512, 3))] )
  filtered_img_res = np.array([resize(filtered_img, (512, 512, 3))])

  return org_img_res, filtered_img_res


def predict_GT(image_path,model):
    org, filt = load_input(input_img_path = image_path)
    # Image.open(image_path)
    prediction = model.predict([org,filt])
    thresh = 0.9
    bin_map = prediction[0] > thresh
    Image.fromarray(np.squeeze(bin_map))
    # predict
    # show the output image

predict_GT('/home/marcassoli/Bureau/Stage 2A/MVSS-Net/MVSS-Net-master/data/fake.png',model_phase2)
