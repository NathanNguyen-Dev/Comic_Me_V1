import os
from ulti import *
model_path = os.path.join('model\ComicGenTrainEx.h5')
@st.cache
def model_load():
    model = tf.keras.models.load_model(model_path)
    return model

def main():
    
    st.header('Comic Me')
    st.write('Turn myself into a comic book character')

    comic_model = model_load()


    menu = ['Image Based', 'Video Based']
    choice = st.sidebar.selectbox('How would you like to be turn ?', menu)

    # Create the Home page
    if choice == 'Image Based':
        Image = st.file_uploader('Upload your portrait here',type=['jpg'])
        gamma = st.slider('Gamma adjust', min_value=0.1, max_value=3.0,value=1.0,step=0.1)
        outputsize = st.selectbox('Output Size', [384,512,768])
        if Image is not None:
            Image = Image.read()
            Image = tf.image.decode_jpeg(Image, channels=3).numpy()
            Autocrop = st.checkbox('Auto Crop Image')  
                                        # change the value here to get different result
            Image = adjust_gamma(Image, gamma=gamma)
            st.image(Image)
            input_image = loadtest(Image,cropornot=Autocrop)
            prediction = comic_model(input_image, training=True)
            prediction = tf.squeeze(prediction,0)
            prediction = prediction* 0.5 + 0.5
            prediction = tf.image.resize(prediction, 
                           [outputsize, outputsize],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            prediction=  prediction.numpy()
            st.image(prediction)
    

    elif choice == 'Video Based':
        run = st.checkbox('Run')
        FRAMEWINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        gamma = st.slider('Gamma adjust', min_value=0.1, max_value=3.0,value=1.0,step=0.1)
        while run:
            _ , frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame  = cv2.flip(frame, 1)
            frame = adjust_gamma(frame, gamma=gamma)
            # Framecrop = st.checkbox('Auto Crop Frame')
            frame = loadframe(frame)
            frame = comic_model(frame, training=True)
            frame = tf.squeeze(frame,0)
            frame = frame* 0.5 + 0.5
            frame = tf.image.resize(frame, 
                           [384, 384],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            frame = frame.numpy()
            # frame =  cv2.resize(frame, (384,384), interpolation = cv2.INTER_AREA)

            FRAMEWINDOW.image(frame)

if __name__ == '__main__':
    main()