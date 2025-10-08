# Documentation of Comic Me V1 App

## Summary
An application that utilises a **CycleGAN** architecture to turn user portraits into comic book styled characters while preserving their identity.
Additionally AutoFace Cropper and gamma adjustment were added for increased user experience.

Deployed and ready to be tested on :

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/nathannguyendev/comic_me_v1/main.py)

## Technology Stack
- CycleGAN
- AutoCropper (OpenCV)
- Streamlit

## Local development
1. Create a virtual environment and install the dependencies listed in `requirements.txt`.
2. Download the trained weights (`model/ModelTrainOnKaggle.h5`) if they are not already present.
3. Launch the Streamlit app:
   ```bash
   streamlit run main.py
   ```
4. Open the provided URL to access the interface.

The sidebar controls let you:
- Adjust the output resolution of the generated image.
- Toggle automatic face cropping and tune the zoom level.
- Apply gamma correction before sending the image to the model.

## Data and training
Data and model training example can be found at [Kaggle](https://www.kaggle.com/nathannguyendev/face2comic)

## References
* Research on CycleGAN -- https://paperswithcode.com/paper/unpaired-image-to-image-translation-using
* CycleGAN Tutorial on TPU -- https://www.kaggle.com/amyjang/monet-cyclegan-tutorial
