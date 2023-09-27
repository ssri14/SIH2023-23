import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np
import pickle

import timm
import torch
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from albumentations.pytorch import ToTensorV2
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# import firebase_admin
# from firebase_admin import auth, credentials 


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4655),(0.2023,0.1994,0.2010))])
pickle_in =open("classifier.pkl","rb")
model=pickle.load(pickle_in)

classes = ["","Grassy","Marshy","Rocky","Sandy"]


def predict_terrain(image):
   img=transform(np.array(Image.open(image)))
   img=torch.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
   prediction=model.forward(img)
   print(prediction)
   idx=torch.argmax(prediction,dim=1)
#    print(type(idx))
   return idx.item()
    
def main():
    st.title("upload image")
    html_temp = """
    <div>classification</div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    image=st.file_uploader("Please upload an image")
    result=0
    if image is not None:
        st.image(image)
    if st.button("Predict"):
        result=predict_terrain(image)
        result=result+1
    st.success('The terrain is {}'.format(classes[result]))

if __name__=='__main__':
    main()



