# import necessary libraries
import streamlit as st
import cv2
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Loading pre-trained parameters for the cascade classifier
try:
    model = load_model('assets/model/model.h5')
    face_cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

except Exception:
    st.write("Error Loading Model")



def detect(image):
    '''
    Author : Niladri Ghosh
    Email ID : niladri1406@gmail.com

    A function which takes in a single argument; the image. The Haar Cascade classifier extracts the face of an
    individual person then saves it. Next the trained cnn model is presented with the image to classify if the
    person is wearing mask. In addition, a bounded box is layed on the region of interest; face.
    '''

    #parameters for text
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (1, 1)
    class_lable=' '      
    # fontScale 
    fontScale = 1 #0.5
    # Blue color in BGR 
    color = (255, 0, 0) 
    # Line thickness of 2 px 
    thickness = 2


    # image dimention
    img_width, img_height = 250, 250

    # converting image from bytes to numpy array
    color_img = np.array(image)
    # bgr format
    color_img = color_img[:, :, ::-1]


    #resize image
    scale = 50 
    width = int(color_img.shape[1] * scale / 100)  
    height = int(color_img.shape[0] * scale / 100)  
    dim = (width, height)  
    
    # resize image  
    color_img = cv2.resize(color_img, dim, interpolation=cv2.INTER_AREA) 

    # converting to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # detecting faces
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 6) 

    # defaut class label
    class_lable = ""

    # default prediction accuracy
    pred_acc = -1
    
    # feed faces from haar cascade classifier to our custom neural network
    img_count = 0
    for (x, y, w, h) in faces:
        org = (x-10,y-10)
        img_count +=1 
        color_face = color_img[y:y+h,x:x+w] # color face
        # save image
        cv2.imwrite('faces/%dface.jpg'%(img_count),color_face)
        
        # load image for prediction
        img = load_img('faces/%dface.jpg'%(img_count), target_size=(img_width,img_height))
        
        # normalizing image
        img = img_to_array(img)/255
        img = np.expand_dims(img,axis=0)
        
        # predict using our model
        pred_prob = model.predict(img)
        
        # max probablity -- 0 or 1 (0 for Mask and 1 for Not wearning Mask)
        pred=np.argmax(pred_prob)

        if pred==1:
            pred_acc = pred_prob[0][1]
            class_lable = "Mask"
            color = (255, 0, 0)
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            # add text to image
            cv2.putText(color_img, class_lable, org, font, fontScale, color, thickness, cv2.LINE_AA)

        else:
            pred_acc = pred_prob[0][0]
            class_lable = "No Mask"
            color = (0, 255, 0)

        cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        # add text to image
        cv2.putText(color_img, class_lable, org, font, fontScale, color, thickness, cv2.LINE_AA) 

    color_img = color_img[:, :, ::-1]
    if class_lable == "" and pred_acc == -1:
        class_lable = "Unable to predict. Try any other picture!"

    # Returning the image with bounding boxes drawn on it (in case of detected objects), and faces array
    return color_img, class_lable, pred_acc


def about():
	st.write(
		'''
		**Binary classifier** is a model which detects two things. I have 
        created a custom convolutional neural network model to detect 
        weather a person is wearing mask or not. **Haar Cascade** is an object 
        detection algorithm. It can be used to detect objects in images or videos.
        In our case we have used the frontal face haar cascade to detect faces in an 
        image then classifying that image to detect a person is wearing mask. 
		
        The algorithm has four stages:
            1. Haar Feature Selection
            2. Creating  Integral Images
            3. Adaboost Training
            4. Cascading Classifiers
            5. Convolved Features
            6. Max Pooling
            7. Flatten Image
            8. Multi Neural Architecture
            9. Binary Classification
		''')


def main():
    st.title("Face Mask Detection")
    st.write("**Binary Classification using Convolutional Neural Networks**")

    choices = ["Home", "About"]
    choice = st.sidebar.selectbox("Select a section to continue.", choices)

    if choice == "Home":

        st.write("**Select about from the sidebar to know more.**")

        # types of image type accepted in.
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

        if image_file is not None:

            image = Image.open(image_file)
            #st.image(image, caption='Uploaded Image', use_column_width=True)
            #st.write("")

            if st.button("Process"):

                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
                result_img, class_lable, pred_prob = detect(image=image)
                st.image(result_img, use_column_width = True)
                st.success("**Prediction**  :  {} ".format(class_lable))
                st.success("**Accuracy **  : {0:.2f} ".format(pred_prob*100))

    elif choice == "About":
        about()


if __name__ == "__main__":
    main()
