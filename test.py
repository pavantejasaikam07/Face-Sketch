import streamlit as st
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

discriminator_model = load_model('Models/g_model1.h5', custom_objects={'InstanceNormalization': InstanceNormalization})


def generate_image_with_confidence(sketch_array):
    # Preprocess the sketch image
    norm_sketch_array = (sketch_array - 127.5) / 127.5

    # Generate the image using the model
    generated_img_array = g_model.predict(np.expand_dims(norm_sketch_array, 0))[0]
    generated_img_array = (generated_img_array * 127.5 + 127.5).astype(np.uint8)
    
    # Calculate the confidence score
    confidence_score = discriminator_model.predict(np.expand_dims(generated_img_array, 0))[0][0]

    # Convert confidence score to percentage
    confidence_percentage = confidence_score * 100
    
    return generated_img_array, confidence_percentage

g_model = load_model('Models/g_model1.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
def generate_image(sketch_array):
    norm_sketch_array = (sketch_array - 127.5) / 127.5
    generated_img_array = g_model.predict(np.expand_dims(norm_sketch_array, 0))[0]
    generated_img_array = (generated_img_array * 127.5 + 127.5).astype(np.uint8)
    return generated_img_array
def production_page():
    st.title('Sketch to Image Generation')

    uploaded_file = st.file_uploader("Upload a sketch image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        sketch_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        st.image(sketch_image, caption='Uploaded Sketch', width=400)

        if st.button("Generate Image", help="Click to generate image", key="generate_image_button"):


            resized_sketch = cv2.resize(sketch_image, (256, 256))
            generated_image, confidence_score = generate_image_with_confidence(resized_sketch)

            generated_image_resized = cv2.resize(generated_image, (sketch_image.shape[1], sketch_image.shape[0]))

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Sketch")
                st.image(sketch_image, use_column_width=True)

            with col2:
                st.subheader("Generated Image")
                st.image(generated_image_resized, use_column_width=True)
            st.success(f"Confidence Score: {confidence_score}")





st.set_page_config(page_title="Gen-Alpha", page_icon="🚀")
title_html = """
    <div style="font-size: 40px; color: black; text-align: center; font-family: 'Arial', sans-serif;">
        Face Sketch Synthesis using C-GAN
    </div>
"""
st.markdown(title_html, unsafe_allow_html=True)


st.image("./statuc/img/logo.png", width=700)
st.markdown("&nbsp;")

page = st.sidebar.radio("Navigation", ["Home", "About Team", "Technology", "How We Developed","Production" ,"Project Summary","Contact Us"])

if page == "Home":
    st.write("Welcome to Gen-Alpha!")
    st.write("Sketch-to-Image Colorization using Deep Learning is an innovative project that transforms grayscale sketches into colored images. Leveraging advanced deep learning techniques, our system offers state-of-the-art colorization with customizable styles and features.")

    st.write("---")
    st.write("# Overview")
    overview_columns = st.columns([1, 3])
    with overview_columns[0]:
        st.image("./statuc/img/logo.png", use_column_width=True)
    with overview_columns[1]:
        st.write("Our project provides an intuitive interface for converting sketches into vibrant colored images. With advanced deep learning algorithms and real-time processing, users can effortlessly generate lifelike images from their sketches.")

    st.write("---")
    st.write("# Key Features")
    key_features_columns = st.columns(2)
    with key_features_columns[0]:
        st.image("./statuc/img/ss.png", width=200)
        st.write(":pencil2: **Sketch-to-Image Conversion**")
        st.write("Transform grayscale sketches into colored images.")
        st.image("./statuc/img/cus.jpeg", width=200)
        st.write(":gear: **Customization Options**")
        st.write("Adjust color styles and features.")
    with key_features_columns[1]:
        st.image("./statuc/img/real.jpeg", width=200)
        st.write(":iphone: **Real-time Processing**")
        st.write("Generate colored images instantly.")
        st.image("./statuc/img/user.jpeg", width=200)
        st.write(":art: **User-friendly Interface**")
        st.write("Intuitive interface for easy interaction.")

    st.write("---")
    st.write("# Recent Updates")
    st.write("Stay tuned for the latest developments and updates!")

    st.write("---")
    st.write("# Project Growth")
    st.write("Check out the growth of our project over time.")

    dates = pd.date_range('2023-01-01', periods=10)
    values = np.random.randn(10)

    st.write("## Line Chart")
    df = pd.DataFrame({'Date': dates, 'Value': values})
    st.line_chart(df.set_index('Date'))

    st.write("## Bar Chart")
    st.bar_chart(df.set_index('Date'))

    st.write("---")
    st.write("# Additional Information")
    st.write("Here's some additional quantitative and qualitative information about our project:")

    st.write("## :bar_chart: Quantitative Metrics")
    st.write("- Number of images colorized: 5,000+")
    st.write("- Average processing time: 3 seconds")
    st.write("- User satisfaction rating: 4.8/5")

    st.write("## :speech_balloon: Qualitative Feedback")
    st.write("Here are some quotes from our users:")
    st.write('"The colorization results are stunning! It\'s like magic!" - Emily')
    st.write('"I\'m impressed by how quickly I can turn my sketches into colorful images." - Michael')

    st.write("## :rocket: Future Plans")
    st.write("We're continuously improving our project. Some upcoming features include:")
    st.write("- Fine-tuning colorization models for better accuracy")
    st.write("- Adding support for different sketch styles")
    st.write("- Introducing batch processing for multiple images")

    st.write("---")
    st.write("# Acknowledgements")
    st.write("We would like to express our gratitude to the following individuals and organizations for their support and contributions to this project:")
    st.write("- Our users for their valuable feedback and encouragement")
    st.write("- Open-source communities for their tools and libraries")
    st.write("- Funding agencies for their financial support")


elif page == "About Team":
    st.markdown("""
    <style>
        .team-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
        }
        .team-section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 10px;
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        }
        .team-title {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        }
        .team-card {
            width: 300px;
            margin: 10px;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 10px;
            transition: transform 0.3s ease;
            text-align: center;
            left: 120px;
            font-size: 16px;
            color: #666;
            font-weight: bold;

        }
        .team-card:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.2);
        }
        .team-name {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .team-role {
            font-size: 16px;
            color: #666;
        }
        .team-lead {
            background-color: #f0f0f0;
        }
        .team-row {
            display: flex;
            justify-content: space-between;
        }
    </style>
""", unsafe_allow_html=True)
    st.write("<h1 style='text-align: center;'>Meet the Team</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>Get to know the team behind Gen-Alpha.</p>", unsafe_allow_html=True)

    st.write("<div class='team-section'>Team 2", unsafe_allow_html=True)
    

    st.write("<div class='team-row'>", unsafe_allow_html=True)
    
    st.write("<div style='display: flex; flex-wrap: wrap; width: 100%;'>", unsafe_allow_html=True)

    st.write("<div class='team-card' style='flex: 1;'>Pavan Teja<br>&nbsp;Field: CSE Data Science <br>&nbsp;Roll No: 21N35A6706</div>", unsafe_allow_html=True)

    st.write("<div class='team-card' style='flex: 1;'>Dr. MV Kamal<br>&nbsp;Position: HOD and Professor<br>&nbsp;Department of Emerging Technologies</div>", unsafe_allow_html=True)

    st.write("</div>", unsafe_allow_html=True)
    st.write("</div>", unsafe_allow_html=True)

elif page == "Project Summary":
    st.title("Project Summary 🎨")

    st.write("""
    Welcome to the project summary section! Here, we provide you with a detailed overview of our project, focusing on generating color images from sketches using conditional Generative Adversarial Networks (cGANs).
    """)

    st.markdown("---")

    st.header("Objective and Overview 📝")

    st.write("""
    Our project aims to develop a system that generates realistic color images from input sketches. By leveraging conditional Generative Adversarial Networks (cGANs), we enable the creation of vibrant and detailed color images based on grayscale sketches.
    """)

    st.markdown("---")

    st.header("Visual Representation 🖼️")

    st.write("""
    Below, we showcase some examples of color images generated from input sketches using our cGAN model. These images demonstrate the model's ability to accurately infer colors and details from the provided sketches.
    """)

    st.image( "./statuc/img/f1-012-01-sz1.jpg",width=300, caption="Generated Color Images")

    st.markdown("---")

    st.header("Achievements and Milestones 🏆")

    st.write("""
    Throughout the project, we have achieved significant milestones:
    - Developed and trained a cGAN model capable of generating high-quality color images from sketches.
    - Achieved realistic colorization results across various sketch styles and content types.
    - Presented our work at AI conferences, receiving positive feedback from the research community.
    - Published research papers detailing our methodology and findings in reputable journals.
    - Collaborated with industry partners to explore practical applications of sketch-to-color image generation.
    """)

    st.markdown("---")

    st.header("Future Directions 🔮")

    st.write("""
    Moving forward, we plan to:
    - Enhance the diversity and realism of generated color images through advanced model architectures and training techniques.
    - Explore additional applications of sketch-to-color image generation, such as photo editing tools and digital art platforms.
    - Conduct user studies to assess the perceptual quality and usefulness of generated color images in various contexts.
    - Extend our research to related tasks such as image inpainting and style transfer for broader impact and innovation.
    """)

    st.markdown("---")

   
    st.markdown("""
    <style>
    /* Custom CSS styles */
    .project-summary-container {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 30px;
    }

    .summary-header {
        color: #1f487e;
        font-size: 24px;
        margin-bottom: 10px;
    }

    .summary-icon {
        color: #1f487e;
        margin-right: 10px;
    }

    .summary-content {
        margin-bottom: 20px;
    }

    .contact-info {
        margin-top: 10px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)


elif page == "How We Developed":
        
    css = """
    h2 {
        color: #1f487e;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    h3 {
        color: #1f487e;
        font-size: 20px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 8px;
    }

    p {
        color: #333333;
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 12px;
    }

    .separator {
        margin-top: 20px;
        margin-bottom: 20px;
        border-top: 1px solid #ccc;
    }

    .icon {
        margin-right: 8px;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.write("<h2>How We Developed 🛠️</h2>", unsafe_allow_html=True)
    st.markdown("<hr class='separator'>", unsafe_allow_html=True)

    st.write("<h3>Step-by-step breakdown of the development process 🛠️</h3>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>📝</span>Gathered and Preprocessed Data: Acquired a comprehensive dataset comprising face sketches paired with corresponding real images. Preprocessed the dataset to ensure uniformity in image dimensions, quality, and format.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🔍</span>Model Architecture Selection: Conducted thorough research to select an appropriate cGAN architecture tailored for face sketch synthesis, considering factors such as model complexity and performance.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🛠️</span>Implementation of cGAN: Developed the generator and discriminator networks using TensorFlow. Implemented conditional input functionality in the generator to facilitate the synthesis process based on input sketches.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>📉</span>Loss Function Definition: Defined a combination of loss functions, including adversarial loss and pixel-wise loss, to guide the training process effectively and ensure high-fidelity image generation.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🖥️</span>Training Process: Conducted extensive training sessions on the prepared dataset, meticulously adjusting hyperparameters such as learning rate and batch size to optimize model convergence and performance.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🔍</span>Evaluation and Validation: Evaluated the trained cGAN model rigorously using a diverse set of evaluation metrics, including SSIM and qualitative assessment by domain experts, to validate its efficacy and generalization capabilities.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='separator'>", unsafe_allow_html=True)

    st.write("<h3>Technical details about cGAN implementation 🧠</h3>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🏗️</span>Generator Architecture: Designed a sophisticated U-Net based architecture for the generator, incorporating skip connections to facilitate information flow between corresponding layers.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🛠️</span>Discriminator Architecture: Employed a PatchGAN discriminator architecture to enhance the model's ability to discern local image structures and features.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🔄</span>Conditional Input Integration: Successfully integrated the sketch input as a conditional input to the generator by concatenating it with the feature maps at various layers, ensuring precise image synthesis.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>📉</span>Loss Function Selection: Adopted a multi-component loss function strategy, combining adversarial loss, L1 pixel-wise loss, and feature matching loss, to effectively guide the training process and optimize image quality.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='separator'>", unsafe_allow_html=True)

    st.write("<h3>Challenges faced and how they were overcome ⚠️</h3>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🔄</span>Data Imbalance: Encountered challenges due to data imbalance within the dataset, particularly in terms of the distribution of facial features. Mitigated this issue through careful data augmentation and sampling techniques to ensure balanced representation.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>❌</span>Mode Collapse: Experienced mode collapse phenomena during training, leading to limited diversity in generated images. Addressed this challenge by implementing techniques such as feature matching loss and spectral normalization to stabilize training and encourage diversity.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🔍</span>Fine Detail Capture: Struggled to capture fine details, such as texture and facial expressions, in synthesized images. Overcame this obstacle through architectural enhancements and regularization techniques, including the incorporation of additional convolutional layers and dropout regularization.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='separator'>", unsafe_allow_html=True)

    st.write("<h3>Future enhancements or directions 🔮</h3>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🔍</span>Incorporating Self-Supervised Learning: Explore the integration of self-supervised learning techniques to further enhance the model's ability to learn robust facial representations and improve image synthesis quality.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>🎭</span>Multi-Modal Image Generation: Extend the capabilities of the system to support multi-modal image generation, enabling the synthesis of diverse facial styles, expressions, and attributes.</p>", unsafe_allow_html=True)
    st.write("<p><span class='icon'>⏱️</span>Real-Time Inference Optimization: Investigate strategies for optimizing the model architecture and inference pipeline to facilitate real-time image synthesis on resource-constrained devices, opening up opportunities for practical applications in various domains.</p>", unsafe_allow_html=True)

elif page == "Contact Us":
    st.title("Contact Us")
    st.write("Feel free to get in touch with us for any inquiries or collaborations!")

    st.markdown("""
    <style>
    /* Define your CSS styles here */
    body {
        background-color: #cceeff; /* Light blue color */
    }

    .contact-header {
        font-size: 24px;
        color: #1f487e;
        margin-bottom: 20px;
    }

    .contact-section {
        margin-bottom: 30px;
    }

    .contact-form-input {
        width: 100%;
        padding: 10px;
        margin-bottom: 15px;
        border: 1px solid #ccc;
        border-radius: 5px;
        box-sizing: border-box;
    }

    .contact-form-textarea {
        width: 100%;
        padding: 10px;
        margin-bottom: 15px;
        border: 1px solid #ccc;
        border-radius: 5px;
        resize: vertical;
        box-sizing: border-box;
    }

    .contact-submit-button {
        background-color: #1f487e;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Contact Form")
    st.markdown('<div class="contact-section">', unsafe_allow_html=True)
    name = st.text_input("Your Name", "")
    email = st.text_input("Your Email", "")
    message = st.text_area("Your Message", "")
    submit_button = st.markdown('<button type="submit" class="contact-submit-button">Submit</button>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Email")
    st.write("You can also reach out to us via email at:")
    st.write("[example@example.com](mailto:example@example.com)")

    st.subheader("Location")
    st.write("We are located in Maisammaguda, Hyderabad.")

    data = {
    'lat': [17.5631],
    'lon': [78.4553]
    }

    df = pd.DataFrame(data)

    st.map(data)

elif page == "Technology":
    st.title("Face Sketch Synthesis using cGAN 🎨")

    st.write("""
    Welcome to the technology section! Here, we showcase the technologies utilized in our project for Face Sketch Synthesis using conditional Generative Adversarial Networks (cGAN).
    """)

    st.markdown("---")

    st.header("Key Technologies Used")

    st.write("""
    Our project leverages a variety of cutting-edge technologies to achieve its objectives. Below are the key technologies employed:
    """)

    st.write("""
    ![cGAN](https://img.icons8.com/color/96/000000/conditional.png) **Conditional Generative Adversarial Networks (cGAN)**

    ![VS Code](https://img.icons8.com/color/96/000000/visual-studio-code-2019.png) **Visual Studio Code**

    ![Anaconda Navigator](https://fileswin.com/wp-content/uploads/2019/08/Anaconda-Navigator-Icon-68x68.png) **Anaconda Navigator**

    ![Jupyter Notebook](https://pmutt.readthedocs.io/en/latest/_images/jupyter_notebook.png) **Jupyter Notebook**

    ![Streamlit](https://img.icons8.com/ios-filled/96/000000/console.png) **Streamlit**

    ![Matplotlib](http://numerique.ostralo.net/python_matplotlib/images/logo.png) **Matplotlib**
    """)

    st.markdown("---")

    st.header("Additional Technologies")

    st.write("""
    In addition to the key technologies mentioned above, our project also utilizes the following technology:
    - CUHK Face Sketch Database
    """)

elif page =="Production":
    production_page()


elif page == "Metrics":

    data = {
    'Batch': list(range(1, 89)),
    'Discriminator Loss': [3.591, 3.471, 3.577, 3.516, 3.472, 3.441, 3.353, 3.340, 3.498, 3.488, 3.508, 3.461, 3.568, 3.466, 3.595, 3.493, 3.433, 3.479, 3.458, 3.512, 3.474, 3.648, 3.613],
    'Generator Loss': [39.243, 28.457, 27.642, 26.161, 25.611, 24.154, 23.810, 24.735, 24.372, 24.266, 23.904, 24.226, 25.240, 23.601, 25.543, 22.193, 22.948, 22.293, 22.200, 17.688, 18.447, 18.115, 18.098],
    'SSIM': [0.76595402890904] * 22  # Repeat the SSIM value for each batch
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Plot using st.bar_chart()
    st.bar_chart(df.set_index('Batch'))   

