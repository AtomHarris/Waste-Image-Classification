import os
import time
import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import l2
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import uuid

# Function to load images from a pickled file
@st.cache_data
def load_pickled_images(file_path):
    with open(file_path, 'rb') as f:
        images = pickle.load(f)
    return images

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=2)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_names[predicted_class[0]]
    return predicted_label, predictions[0][predicted_class[0]]

# Function to move misclassified image to the correct folder
def move_image_to_correct_folder(image, correct_class):
    # Create a unique filename using UUID
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    # Create the full path where the image will be saved
    image_path = os.path.join('RealWaste2/train', correct_class, unique_filename)
    # Save the image to the specified path
    image.save(image_path)
    print(st.success(f"Image moved to {correct_class} folder"))

    return correct_class

# Function to retrain the model
def retrain_model():
    train_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    )

    train_data = train_generator.flow_from_directory(
        'RealWaste2/train',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
    )

    val_data = val_generator.flow_from_directory(
        'RealWaste2/val',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
    )

    y_train = train_data.classes

    cls_wt = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {i: cls_wt[i] for i in range(len(cls_wt))}

    pretrained_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    pretrained_model.trainable = False
    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(180, activation='relu', kernel_regularizer=l2(0.001))(pretrained_model.output)
    x = tf.keras.layers.Dense(360, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = tf.keras.layers.Dense(8, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint("resnet50.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, min_delta=0.0001, verbose=1)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=12,
        class_weight=class_weights,
        callbacks=[checkpoint, reduce_lr]
    )

    with open('Pickle_files/models/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    st.success("Model retrained successfully")

# Load the pre-trained model from the .pkl file
@st.cache_resource
def load_model():
    with open('Pickle_files/models/resnet50.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Class names
class_names = ['Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

# Company Logo
st.sidebar.image('Images/image4.png', use_column_width=True)

# Sidebar Selectbox initialization
selected = st.sidebar.selectbox("Go to", ["Home", "About", "Contact"])

# Path to your pickled images folder
pickled_images_folder = 'Pickle_files/images'

# Initialize session state for material handling instructions
if 'instructions' not in st.session_state:
    st.session_state.instructions = {
        'compost': "Instructions to be given by customer",
        'recycle': "Instructions to be given by customer",
        'trash': "Instructions to be given by customer"
    }

# Main Home Classification Page
if selected == "Home":

    # Function to randomly select one image from a list
    def select_one_image(images):
        return random.choice(images)

    # Home Page Option Menu Initialization
    selected1 = option_menu(
        menu_title=None,
        options=["Home", "Material Handling", "Developer Mode"],
        icons=["house", "file-earmark", "tools"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    # Home Classification Page
    if selected1 == "Home":
        with st.container(border=True):
            st.image('Images/image1.webp', use_column_width=True)
            st.write("<h1 style='text-align: center;'>IDump Classification System</h1>", unsafe_allow_html=True)

            with st.sidebar:
                st.header('Landfill Samples')

                if os.path.isdir(pickled_images_folder):
                    pickled_files = [f for f in os.listdir(pickled_images_folder) if f.lower().endswith('.pkl')]

                    if pickled_files:
                        selected_images = []

                        for pickled_file in pickled_files:
                            images = load_pickled_images(os.path.join(pickled_images_folder, pickled_file))
                            image = select_one_image(images)
                            selected_images.append(image)

                        num_columns = 2
                        cols = st.sidebar.columns(num_columns)
                        for idx, (image, filename) in enumerate(zip(selected_images, class_names)):
                            col = cols[idx % num_columns]
                            image = image.replace("\\", "/")
                            col.image(image, caption=filename, use_column_width=True)
                    else:
                        st.sidebar.warning('No pickled files found in the specified folder.')
                else:
                    st.sidebar.error(f'The directory "{pickled_images_folder}" does not exist. Please check the path and try again.')

            uploaded_files = st.file_uploader("Choose images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

            image_details = []

            if 'last_uploaded_image' not in st.session_state:
                st.session_state.last_uploaded_image = None

            col1, col2 = st.columns(2, gap="small")

            with col1:
                with st.container(border=True):
                    if uploaded_files:
                        image = Image.open(uploaded_files[-1])
                        st.session_state.last_uploaded_image = image

                        st.image(image, use_column_width=True)

                        image_details.append({
                            'Filename': uploaded_files[-1].name,
                            'Width': image.width,
                            'Height': image.height
                        })

                        if image_details:
                            df = pd.DataFrame(image_details)
                            df.set_index("Filename", inplace=True)
                            st.write(df)

            with col2:
                with st.container(border=True):
                    if st.session_state.last_uploaded_image:
                        with st.spinner('Classifying...'):
                            time.sleep(2)

                        st.success('Done!')

                        try:
                            label, confidence = predict(st.session_state.last_uploaded_image)

                            st.write(f"PREDICTED CLASS: {label}")
                            st.write(f"CONFIDENCE: {confidence * 100:.2f}%")

                            st.subheader('MATERIAL HANDLING', divider=True)
                            if label == 'Food Organics' or label == 'Vegetation':
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image('Images/Compost.jpg', width=120)
                                with col2:
                                    st.write('Divertible Organics')
                                st.write(st.session_state.instructions['compost'])
                            elif label in ['Glass', 'Metal', 'Paper', 'Plastic']:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image('Images/Recycle.jpg', width=120)
                                with col2:
                                    st.write('Divertible Recyclables')
                                st.write(st.session_state.instructions['recycle'])
                            else:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image('Images/Trash.jpg', width=120)
                                with col2:
                                    st.write('Residuals')
                                st.write(st.session_state.instructions['trash'])

                            
                        except Exception as e:
                            st.write(f"Error: {e}")
            with st.container(border=True):
                if st.session_state.last_uploaded_image:
                    st.write("Was the Image Classified Correctly?")
                    choice = st.radio("", ("Select an Option", "👍 Yes", "👎 No"), index=0)
                    if choice == "👍 Yes":
                        st.success("Classification correct")
                    elif choice == "👎 No":
                        st.session_state.wrongly_classified_image = st.session_state.last_uploaded_image
                        st.warning("Misclassified image has been moved to 'Developer Mode' for further action.")                

    elif selected1 == "Material Handling":
        st.write("<h1 style='text-align: center;'>Material Handling Guidelines</h1>", unsafe_allow_html=True)

        st.subheader('Compost Instructions', divider=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image('Images/Compost.jpg', width=120)
        with col2:
            st.write('Divertible Organics:')
            st.write('From which energy and fertilizer can be derived')
        st.write(st.session_state.instructions['compost'])

        st.subheader('Recycle Instructions', divider=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image('Images/Recycle.jpg', width=120)
        with col2:
            st.write("Recyclable Inorganics:")
            st.write("Fit for repurposing")
        st.write(st.session_state.instructions['recycle'])

        st.subheader('Trash Instructions', divider=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image('Images/Trash.jpg', width=120)
        with col2:
            st.write("Inorganic Materials:")
            st.write("Requiring Landfill")
        st.write(st.session_state.instructions['trash'])

    elif selected1 == "Developer Mode":
        with st.container(border=True): 
            PASSWORD = "0000"

            # Initialize session state for access control
            if 'access_granted' not in st.session_state:
                st.session_state.access_granted = False

            if not st.session_state.access_granted:
                # Password input section
                st.header("Developer Mode", divider="rainbow")
                password = st.text_input("Enter password to access Developer Mode:", type="password")

                if st.button("Enter"):
                    if password == PASSWORD:
                        st.session_state.access_granted = True
                        st.success("Password correct. Access granted.")
                        st.experimental_rerun() 
                    else:
                        st.error("Incorrect password. Please try again.")
            else:
                selected2 = option_menu(
                            menu_title=None,
                            options=["Teach Mode","Instructions Mode"],
                            icons=["bookmark", "pencil-square"],
                            menu_icon="cast",
                            default_index=0,
                            orientation="horizontal",
                        )

                if selected2=="Teach Mode":           
                    # Content for Developer Mode
                    st.header("Developer Mode Content", divider="rainbow")

                    # Initialize corrected_images in session state if not already present
                    if 'corrected_images' not in st.session_state:
                        st.session_state.corrected_images = []

                    # Function to handle image selection and class correction
                    def handle_image_correction():
                        if 'wrongly_classified_image' in st.session_state and st.session_state.wrongly_classified_image:
                            col1, col2 = st.columns(2, vertical_alignment="center")  
                            with col1:
                                image = st.session_state.wrongly_classified_image
                                st.image(image, caption="Wrongly Classified Image", use_column_width=True)
                            with col2:
                                class_names = ['Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
                                correct_class = st.selectbox("Select the correct class:", class_names)
                                if st.button("Confirm Correction"):
                                    if correct_class:
                                        moved_class=move_image_to_correct_folder(image, correct_class)
                                        
                                        st.session_state.corrected_images.insert(0,image)
                                        st.write(f"Moved to {moved_class} folder")

                                        # Refresh the container with the latest images
                                        st.experimental_rerun()

                                st.warning('WARNING:  \nTEACH ACTION TAKES TIME!!!', icon="⚠️")
                                #st.write("TEACH ACTION TAKES TIME!!!", divider= True)

                                # Placing a  "TEACH" button in the same column as the selectbo
                                #if st.session_state.corrected_images:
                                if st.button("TEACH",  key="teach-button"):    
                                    st.write("Training model has begun!")                           
                                    st.write("Kindly Be Patient......")
                                    st.write("This will take up to an hour!!!")
                                    retrain_model()
                                    # Clear the corrected images after retraining
                                    st.session_state.corrected_images.clear()
                            
                        else:
                            st.write("No wrongly classified image to correct.")


                    # Call the function to handle image correction
                    handle_image_correction()

                    # Container for displaying the latest corrected images
                    with st.container():
                        if st.session_state.corrected_images:
                            st.subheader("Images To Teach", divider="rainbow")
                            num_columns = 5
                            cols = st.columns(num_columns)
                            for idx, image in enumerate(st.session_state.corrected_images):
                                col = cols[idx % num_columns]
                                col.image(image, use_column_width=True)
                if selected2=="Instructions Mode":
                    st.header("Instruction Mode Content", divider="rainbow")
                    with st.container(border= True):    
                        st.write("Update the section required:")          
                        with st.expander("Compost Instructions"):
                            compost_instructions = st.text_area("Edit Compost Instructions", st.session_state.instructions['compost'])
                            if st.button("Update Compost Instructions"):
                                st.session_state.instructions['compost'] = compost_instructions
                                st.success("Compost instructions updated successfully!")

                        with st.expander("Recycle Instructions"):
                            recycle_instructions = st.text_area("Edit Recycle Instructions", st.session_state.instructions['recycle'])
                            if st.button("Update Recycle Instructions"):
                                st.session_state.instructions['recycle'] = recycle_instructions
                                st.success("Recycle instructions updated successfully!")

                        with st.expander("Trash Instructions"):
                            trash_instructions = st.text_area("Edit Trash Instructions", st.session_state.instructions['trash'])
                            if st.button("Update Trash Instructions"):
                                st.session_state.instructions['trash'] = trash_instructions
                                st.success("Trash instructions updated successfully!")
elif selected == "About":
    tab1, tab2 = st.tabs(["Software Used", "Hardware Used"])

    with tab1:
        st.header("Software Description", divider=True)
        st.write("COMING SOON......")

    with tab2:
        st.header("Hardware Description", divider=True)
        st.write("COMING SOON......")

elif selected == "Contact":
    st.header("DEVELOPER CONTACTS", divider="rainbow")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Annabellah Mbungu   \nannbellah.mbungu@student.moringaschool.com     \nNumber:  ")
        with col2:
            st.write("Brian Muthama  \nbrian.muthama@student.moringaschool.com   \nNumber:  ")
        with col3:
            st.write("Harris Lukundi  \nharris.lukundi@student.moringaschool.com  \nNumber:  ")
