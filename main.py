import streamlit as st
import streamlit.components.v1 as com
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

st.set_page_config(
    page_title="Terrain Identifier",
    page_icon="⛰️",
    # layout="wide"
)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {visibility: hidden;}
.text_head {
    font-size: 52px;
    color: #7B7648;
    line-height: 55px;
    max-width: 500px;
    font-weight: 900;
    margin-top: 0px;
    margin-right: 0px;
    margin-bottom: 50px;
    margin-left: 0px;
    }
</style>

"""

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {visibility: hidden;}
</style>

"""

with open("style.css") as source:
    design = source.read()


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
    st.header("UPLOAD PHOTOS")
    image=st.file_uploader("Please upload an image")
    result=0
    if image is not None:
        st.image(image)
    if st.button("Predict"):
        result=predict_terrain(image)
        result=result+1
    st.success('The Terrain Is {}'.format(classes[result]))

if __name__=='__main__':
    main()

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
