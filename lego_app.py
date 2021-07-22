import cv2
import numpy as np
from itertools import product
from math import ceil
import streamlit as st
import tempfile
import streamlit.components.v1 as stc
import argparse

from PIL import Image


DEMO_IMAGE = 'input.jpg'
DEMO_VIDEO = 'night.mp4'
OUTPUT_VIDEO = 'output.mp4'
OUTPUT_IMAGE = 'output.jpg'


HTML_BANNER = """
<div style="background-color:Orange;padding:10px;border-radius:10px">
<h1 style="color:Black;text-align:center;">Lego App Generator</h1>
</div>
"""

stc.html(HTML_BANNER)

st.sidebar.title('Lego App Generator')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader('Parameters')

app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Run on Image','Run on Video'],index = 0
)

if app_mode == 'About App':

    st.markdown('''
    
    In this application we recreate the Images and Videos with **Legos** \n
    This code is heavily inspired from **[Viet Nguyen's](https://www.linkedin.com/in/vietnguyen-tum/)** 
    Lego Generator repo which you  can checkout from [here](https://github.com/uvipen/Lego-generator)
    ''')

    st.subheader('Demo Image')

    st.image(OUTPUT_IMAGE,use_column_width = True)

    st.subheader('Demo Video')
    dem = open(OUTPUT_VIDEO,'rb')
    out_vid = dem.read()

    


    st.video(out_vid)



    expander = st.beta_expander('About Author')

    auth = ('''
          
             Hey this is ** Pavan Kunchala ** I hope you like the application \n
            I am looking for ** Collabration ** or ** Freelancing ** in the field of ** Deep Learning ** and 
            ** Computer Vision ** \n
            I am also looking for ** Job opportunities ** in the field of** Deep Learning ** and ** Computer Vision** 
            if you are interested in my profile you can check out my resume from 
            [here](https://drive.google.com/file/d/1DyTBRvaXHExa1J5adUnt_7eTO-qy8NzC/view?usp=sharing)
            If you're interested in collabrating you can mail me at ** pavankunchalapk@gmail.com ** \n
            You can check out my ** Linkedin ** Profile from [here](https://www.linkedin.com/in/pavan-kumar-reddy-kunchala/) \n
            You can check out my ** Github ** Profile from [here](https://github.com/Pavankunchala) \n
            You can also check my technicals blogs in ** Medium ** from [here](https://pavankunchalapk.medium.com/) \n
            If you are feeling generous you can buy me a cup of ** coffee ** from [here](https://www.buymeacoffee.com/pavankunchala)
             
            ''')

    expander.write(auth)


elif app_mode == 'Run on Image':

    def get_args():


        parser = argparse.ArgumentParser("Lego-generator")
        parser.add_argument("--input", type=str, default="data/input.jpg", help="Path to input image")
        parser.add_argument("--output", type=str, default="data/output.jpg", help="Path to output image")
        parser.add_argument("--stride", type=int, default=15, help="size of each lego brick")
        parser.add_argument("--overlay_ratio", type=float, default=0.2, help="Overlay width ratio")
        args = parser.parse_args()
        return args

    st.subheader('Running on Image')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.subheader('Orignal Image')
    st.sidebar.image(image,use_column_width = True)

    stride = st.sidebar.slider('Size of Each Lego Brick',min_value = 0,max_value =100,value = 15)

    opt = get_args()

    opt.stride = stride

    lego_brick = cv2.imread("data/1x1.png", cv2.IMREAD_COLOR)
    lego_brick = cv2.resize(lego_brick, (opt.stride, opt.stride)).astype(np.int64)
    lego_brick[lego_brick < 33] = -100
    lego_brick[(33 <= lego_brick) & (lego_brick <= 233)] -= 133
    lego_brick[lego_brick > 233] = 100

        #image = cv2.imread(opt.input, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (ceil(image.shape[1]/opt.stride)*opt.stride, ceil(image.shape[0]/opt.stride)*opt.stride))
    height, width, num_channels = image.shape
    blank_image = np.zeros((height, width, 3), np.uint8)
    for i, j in product(range(int(width / opt.stride)), range(int(height / opt.stride))):
        partial_image = image[j * opt.stride: (j + 1) * opt.stride,
                            i * opt.stride: (i + 1) * opt.stride, :]
        avg_color = np.mean(np.mean(partial_image, axis=0), axis=0)
        blank_image[j * opt.stride: (j + 1) * opt.stride, i * opt.stride: (i + 1) * opt.stride,
        :] = np.clip(avg_color + lego_brick, 0, 255)
    if opt.overlay_ratio:
        height, width, _ = blank_image.shape
        overlay = cv2.resize(image, (int(width * opt.overlay_ratio), int(height * opt.overlay_ratio)))
        blank_image[height - int(height * opt.overlay_ratio):, width - int(width * opt.overlay_ratio):, :] = overlay
        #cv2.imshow('img',blank_image)
        #cv2.imwrite(opt.output, blank_image)

    st.subheader('Lego')

    st.image(blank_image,use_column_width = True)

elif app_mode == 'Run on Video':

    st.subheader('Running on Video')


    def get_args():
        parser = argparse.ArgumentParser("Lego-generator")
        parser.add_argument("--input", type=str, default="data/input.mp4", help="Path to input image")
        parser.add_argument("--output", type=str, default="data/output.mp4", help="Path to output image")
        parser.add_argument("--stride", type=int, default=10, help="size of each lego brick")
        parser.add_argument("--fps", type=int, default=0, help="frame per second")
        parser.add_argument("--overlay_ratio", type=float, default=0.2, help="Overlay width ratio")
        args = parser.parse_args()
        return args

    st.sidebar.text('Params For video')
    use_webcam = st.sidebar.button('Use Webcam')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True)

    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])

    tfflie = tempfile.NamedTemporaryFile(delete=False)

    stop_button = st.sidebar.button('Stop Processing')

    if stop_button:
        st.stop()



    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
            
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    

        
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)

    stride = st.sidebar.slider('Size of Each Lego Brick',min_value = 0,max_value =100,value = 15)


    opt = get_args()


    opt.stride = stride
    lego_brick = cv2.imread("data/1x1.png", cv2.IMREAD_COLOR)
    lego_brick = cv2.resize(lego_brick, (opt.stride, opt.stride)).astype(np.int64)
    lego_brick[lego_brick < 33] = -100
    lego_brick[(33 <= lego_brick) & (lego_brick <= 233)] -= 133
    lego_brick[lego_brick > 233] = 100

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','8')

    stframe = st.empty()


    while vid.isOpened():
        flag, frame = vid.read()
        if not flag:
            break
        frame = cv2.resize(frame, (
            ceil(frame.shape[1] / opt.stride) * opt.stride, ceil(frame.shape[0] / opt.stride) * opt.stride))
        height, width, num_channels = frame.shape
        blank_image = np.zeros((height, width, 3), np.uint8)
        for i, j in product(range(int(width / opt.stride)), range(int(height / opt.stride))):
            partial_frame = frame[j * opt.stride: (j + 1) * opt.stride,
                            i * opt.stride: (i + 1) * opt.stride, :]
            avg_color = np.mean(np.mean(partial_frame, axis=0), axis=0)
            blank_image[j * opt.stride: (j + 1) * opt.stride, i * opt.stride: (i + 1) * opt.stride,
            :] = np.clip(avg_color + lego_brick, 0, 255)
        if opt.overlay_ratio:
            height, width, _ = blank_image.shape
            overlay = cv2.resize(frame, (int(width * opt.overlay_ratio), int(height * opt.overlay_ratio)))
            blank_image[height - int(height * opt.overlay_ratio):, width - int(width * opt.overlay_ratio):, :] = overlay

        #cv2.imshow('img',blank_image)

        stframe.image(blank_image,channels = 'BGR',use_column_width=True)

    vid.release()
        #out.release()
    cv2.destroyAllWindows()
    st.success('Video is Processed')
    st.stop()




    





    

