- This is our team mentor's github account
![image](https://github.com/manavjain2005/SIH2023-23/assets/112813661/f9f3a5b2-15c3-4605-b8f1-54c16a65e951)

# SIH2023-23
### Team Terrainto
# Problem Statement `(SIH1418)`
**Description** - Vision based methods using deep learning such as CNN to perform terrain recognition **(sandy/rocky/grass/marshy)** enhanced with implicit quantities information such as the roughness, slipperiness, an important aspect for high-level environment perception.
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
- `ResNet18 (pre-trained)` with one `Fully Connected Layer`(512 x 4) predicting `4 classes`
### Dataset
- 31517 `training` images with `4 classes` namely `Grassy`, `Marshy`, `Rocky`, `Sandy`
- 6765 `validation` images and 6769 `test` images
- Normalized with the pretrained ResNet18 measures.

### Training
- The pre-trained layers are `freezed` and only the last `FC layer` was trainined for 1 complete pass through all the samples.
- `Mini_batch size`: 32
### Validation
- Accuracy on `validation_set` 91.1 % and on `test_set` 93.8%
- `Mini_batch size`: 32
### Learning Curve
![WhatsApp Image 2023-09-27 at 18 38 18](https://github.com/manavjain2005/SIH2023-23/assets/112813661/721b1fb0-2cab-4a00-8f10-d8bd5e9cab15)

# Integrating Web-Development
- The model is deployed using `Streamlit` library
- we `Pickeled` the model for its interation with Streamlit's "app.py"
- `Image uploader` coloumn that uploads image to model and recieves predicted index as return value
- The result is displayed on the website.


# Contributors
- Manav Jain (Leader): https://github.com/manavjain2005
- Harsh Singh : https://github.com/hharshas
- Abhinav Jain : https://github.com/jainjabhi05
- Aadhya Jain : https://github.com/aadhya0002
- Dyuti Ballav Paul : https://github.com/dyuti01
- Pratham Todi : https://github.com/pra1ham28
