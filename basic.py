import streamlit as st
import requests
import os.path
import torch
from PIL import Image
import utils_model
import utils_image as util
import numpy as np
import cv2
import pandas as pd
import pickle
from pathlib import Path 
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
import streamlit as st
users = ["admin"]
usernames = ["admin"]
file_path=Path(__file__).parent / "hashed_passwords.pkl"

with file_path.open("rb") as file:
	hashed_passwords=pickle.load(file)
authenticator = stauth.Authenticate(users, usernames, hashed_passwords, "demo_auth", "rkey1", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
	st.error("Username/password is incorrect")
if authentication_status == None:
		st.warning("Please enter your username and password")

if authentication_status:
    selected = option_menu(
        menu_title=None,
        options=["OCT"],
        icons=["book"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    if selected == "OCT":
        torch.cuda.empty_cache()
        st.write('The method uses State of the art DeepLearning Models for Image Denoising followed by ROI based segmentation for finding the disease detection on OCT Eye images.')

        upfile = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        device = st.radio('Select run in CPU or CUDA GPU', ('cpu','cuda'), index=1, horizontal=True)
        st.write('You selected:', device) 
        option1 = st.radio('Select the Denoising Model:', ('Version 1','other'), horizontal=True)
        st.write('You selected:', option1)  
        option2 = st.radio('Select the Segmentation Model:', ('Version 1', 'Version 2','Version 3'), horizontal=True)
        st.write('You selected:', option2)        
        proceed = st.button('Proceed')


        if upfile is not None:
            if proceed == True:
                torch.cuda.empty_cache()
                img = Image.open(upfile)
                img.save('input.png')
                n_channels = 3
                img_U = util.imread_uint('input.png', n_channels=n_channels)
                st.write("Original Input Image")
                st.image(upfile)
                image_original=img_U
                model_name = 'team15_SAKDNNet.pth'
                #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                from team15_SAKDNNet import SAKDNNet as net
                model = net(in_nc=n_channels,config=[4,4,4,4,4,4,4],dim=64)
                model.load_state_dict(torch.load(model_name), strict=True)
                model.eval()
                for k, v in model.named_parameters():
                    v.requires_grad = False      
                model = model.to(device)
                img_N = util.uint2tensor4(img_U)
                img_N = img_N.to(device)
                img_DN = utils_model.inference(model, img_N, refield=64, min_size=512, mode=2)
                img_DN = model(img_N)
                img_DN = util.tensor2uint(img_DN)
                st.write("Denoised Image")
                st.image(img_DN)

                from segment_anything import sam_model_registry, SamPredictor
                

                #sam_checkpoint = option2+'.pth'
                #model_type = option2
                option2 = "vit_b"
                sam = sam_model_registry[option2](checkpoint=option2+'.pth')
                sam.to(device=device)
                predictor = SamPredictor(sam)
                predictor.set_image(image_original)
                input_point = np.array([[200, 200], [900, 80]])
                input_label = np.array([1, 1])
                masks, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label, multimask_output=False)
                mask_input = logits[np.argmax(scores), :, :]
                new_masks = np.moveaxis(masks, [0], [2] )
                newarr = masks.reshape(masks.shape[1], masks.shape[2])
 

                from scipy import ndimage
                import numpy as np
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                newarr=ndimage.binary_opening(newarr,structure=np.ones((10,10)))
                #newarr = cv2.morphologyEx(newarr, cv2.MORPH_CLOSE, kernel)
                #display(Image.fromarray(newarr))

             

                new_image1 = np.zeros_like(newarr)
                for j in range(newarr.shape[1]):
                    for i in reversed(range(newarr.shape[0])):
                        if newarr[i,j] == True:
                            new_image1[0:i,j] = True
                #display(Image.fromarray(new_image1))

                            
                new_image2=new_image1.astype(np.uint8)
                new_image2=new_image2*255
                M2 = np.float32([[1, 0, 0], [0, 1, -50]])
                new_image2 = cv2.warpAffine(new_image2, M2, (new_image2.shape[1], new_image2.shape[0]))
                #display(Image.fromarray(new_image2))

                            
                new_image3=new_image1.astype(np.uint8)
                new_image3=new_image3*255
                M3 = np.float32([[1, 0, 0], [0, 1, -70]])
                new_image3 = cv2.warpAffine(new_image3, M3, (new_image3.shape[1], new_image3.shape[0]))
                #display(Image.fromarray(new_image3))

                img_DN = cv2.cvtColor(img_DN, cv2.COLOR_BGR2GRAY)
                new_image = np.zeros_like(img_DN)
                total_segmented_zone_pixel = 0
                for i in range(img_DN.shape[0]):
                    for j in range(img_DN.shape[1]):
                            if new_image1[i,j] == True:
                                total_segmented_zone_pixel = total_segmented_zone_pixel + 1;
                                new_image[i,j] = img_DN[i,j]
                ret,thresh1 = cv2.threshold(new_image,60,255,cv2.THRESH_BINARY)       
                st.write("Disease ROI Segmented Image")
                st.image(thresh1)
                number_of_white_pix = np.sum(thresh1 == 255)
                st.write("No of Disease Specific Pixels:")
                st.write(number_of_white_pix)
        torch.cuda.empty_cache()
     
  
    authenticator.logout("Logout", "sidebar")
        

