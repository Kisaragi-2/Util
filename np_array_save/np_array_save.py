import numpy as np
import glob
from keras.preprocessing.image import load_img,img_to_array
from tqdm import tqdm

IMG_SIZE=32
#True=Grayscale, False=RGB
COLOR=True
#Name to load images Folder
DIR_NAME='./Images'
#Name to save
SAVE_FILE_NAME='SaveImages'
#sahpe File Name
if COLOR:
    SAVE_FILE_NAME=SAVE_FILE_NAME+'_'+str(IMG_SIZE)+'Gray'
else:
    SAVE_FILE_NAME=SAVE_FILE_NAME+'_'+str(IMG_SIZE)+'RGB'

#load madomagi images and reshape
img_list=glob.glob(DIR_NAME+'/*.jpg')
temp_img_array_list=[]
for img in tqdm(img_list):
    temp_img=load_img(img,grayscale=COLOR,target_size=(IMG_SIZE,IMG_SIZE))
    temp_img_array=img_to_array(temp_img)
    temp_img_array_list.append(temp_img_array)


temp_img_array_list=np.array(temp_img_array_list)

#save np.array
np.save(SAVE_FILE_NAME+'.npy',temp_img_array_list)

#confirmation
print(temp_img_array_list)
print(temp_img_array_list.shape)