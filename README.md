![image](https://github.com/manavjain2005/SIH2023-23/assets/112813661/f9f3a5b2-15c3-4605-b8f1-54c16a65e951)

# SIH2023-23
### Team Terrainto
# Problem Statement
Description - Vision based methods using deep learning such as CNN to perform terrain recognition (sandy/rocky/grass/marshy) enhanced with implicit quantities information such as the roughness, slipperiness, an important aspect for high-level environment perception.
- **Deployed ML model link:** https://terrain.streamlit.app/
# Python Libraries:
- Numpy
- Pandas
- Matplotlib
- Albumantations
- Torch
- Torchvision
- PIL
- Pickle
- Streamlite

# Machine Learning Model:
- ResNet18 (pre-trained) with one Fully Connected Layer (512 x 4) predicting 4 classes
### Dataset
- 31517 training images with 4 classes
- 6765 validation images and 6769 test images
- Normalized with the pretrained ResNet18 measures.

### Training
- The pre-trained layers are freezed and only the last FC layer was trainined for 1 complete pass through all the samples.
- Accuracy on validation set: 

