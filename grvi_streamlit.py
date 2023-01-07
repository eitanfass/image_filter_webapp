import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.colors as colors
import matplotlib
import os
from PIL import Image,ImageFilter
st.set_option('deprecation.showPyplotGlobalUse', False)

def GRVI(img):#
  '''functon gets img path and boolian 
  values to deermin the image 
  filtering before applying 
  GRVI index on the img in the path. returns: img,index (as float imgs)'''
  
  #img = img.convert("F")#format img to float_img
  img=np.array(img)
  return (img[:,:,1]-img[:,:,0])/(img[:,:,1]+img[:,:,0])# return img and indexed img



def show_index(img,index,index_name='RGVI'):# function that plots img and index, with colorbar, index_name will show in the title
  # define figure size: width 10 and height 15
  plt.figure(figsize=(10, 15))

  # subplot for the RGB
  ax1 = plt.subplot(121, title='Original RGB')
  im1 = ax1.imshow(img) 

  # subplot for the index
  ax2 = plt.subplot(122, title=f'{index_name} index, mean={np.nanmean(index):.3f}') # notice the position, and the title
  cmap=matplotlib.cm.get_cmap('Spectral_r',10)#set colormap to Spectral
  im2 = ax2.imshow(index*255,cmap=cmap)#show index in Spectral colormap
  #im2.set_clim(vmax=index.max(), vmin=index.min())# set min max values of color to max and min values of index img
  
  # add colorbar only to the image on the right
  divider = make_axes_locatable(ax2)
  colorbar_ax = divider.append_axes("right", size="5%", pad=0.05)  
  plt.colorbar(im2, cax=colorbar_ax);


def create_mask_from_index(index,thresh=None):# function that creats a mask for an index img from a threshhold if provided
  if thresh==None:
    thresh=index.mean()#if no threshold is given it is set to index mean
  
  mask=index.copy()#copy index to mask
  mask[mask>thresh]=1# if pixel value is larger than threshold set value to 1 
  mask[mask<=thresh]=0  # if pixel value is smaller than threshold set value to 0
  return mask #returns a binery one layer mask


# Set the app title
favicon = 'fabicon.jpg'

# main page
st.set_page_config(page_title='GRVI - Eitan Fass', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')

# add dropdown to select pages on left
app_mode = st.sidebar.selectbox('Navigate',
                                  ['About App', 'GRVI an Image'])

# Run image
if app_mode == 'GRVI an Image':
    st.title('GRVI Index and Mask Generator')
    st.sidebar.markdown('---') # adds a devider (a line)
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )



    # Allow the user to upload an image
    uploaded_image =st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_image == None:
        uploaded_image = 'demo.jpg'
    if uploaded_image is not None:
        
        # Read the image and convert to a NumPy array
        image = Image.open(uploaded_image)
        
        # Get the pixel values as a flat 1D array
        pixels = list(image.getdata())

        pixels = np.array(pixels).reshape(image.height, image.width, 3)  # 3D RGB image
        # Separate the green and red layers
        green_layer = pixels[:, :, 1]
        red_layer = pixels[:, :, 0]

        # Create sliders for the weights of the green and red layers
        green_weight = st.slider('Green weight', 0.0, 1.0, 0.5)
        red_weight = st.slider('Red weight', 0.0, 1.0, 0.5)
        if st.button('What is this?'):
          # Display the text as markdown when the button is clicked
          st.markdown('''If the index is not properly differentiating between green and red, adjust the color weights to improve the results.''')

        # Calculate the GRVI index
        grvi = (green_weight * green_layer - red_weight * red_layer) / (green_weight * green_layer + red_weight * red_layer)

        # Display the GRVI index      
        img=np.array(image)
        st.sidebar.image(image)
 
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 15))

        # Display the original image in the first subplot
        im1 = ax1.imshow(img)
        ax1.set_title('Original RGB')

        # Display the indexed image in the second subplot
        cmap = matplotlib.cm.get_cmap('Spectral_r', 10)
        im2 = ax2.imshow(grvi, cmap=cmap)
        ax2.set_title(f'GRVI index, mean={np.nanmean(grvi):.3f}')

        # Add a color bar to the second subplot
        divider = make_axes_locatable(ax2)
        colorbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=colorbar_ax)

        # Display the figure in the Streamlit app
        st.pyplot()
# This will display the original image and the indexed image with a color bar in a Streamlit app. The color map will be set to Spectral_r with 10 discrete colors. You can customize the color map by using a different value for the cmap parameter (e.g., 'viridis', 'plasma', 'inferno', etc.) and the number of discrete colors by modifying the second argument of get_cmap. You can also customize the appearance of the color bar by using the various options available in the colorbar function.
        # Show the indexed image, mask, and original image side by side
        #show_index(image, index)
#         st.pyplot()
        
 
#         st.image(image, caption='Original image', use_column_width=True)
#         st.image(index, caption='Indexed Image', use_column_width=True)
        mask_thresh= mask_threshold = st.slider('Mask sensetivity', -1.0, 1.0, 0.0, 0.02)# asks for input from the user
        if st.button('What is this? '):
          # Display the text as markdown when the button is clicked
          st.markdown('''Adjusting the mask threshold changes the sensitivity of the mask. A high threshold value results in a more sensitive mask, while a low threshold value results in a less sensitive mask.''')
        # Create a binary mask from the index using the mean value as the threshold
        mask = create_mask_from_index(grvi,mask_thresh)
        
        d_mask=img.copy()
        d_mask[:,:,0],d_mask[:,:,1],d_mask[:,:,2]=mask,mask,mask
        masked_img=img*d_mask
        st.image(masked_img, caption='Masked Image', use_column_width=True)
        
        # Allow the user to choose a destination folder for the saved images
#         save_folder = st.folder_selector('Save images to:', default='.')
        # Create a file selector dialog using the askdirectory function
        save_folder = os.path.abspath(st.sidebar.selectbox('Select a destenation folder for saved images:', os.listdir()))
        # Save the indexed image, mask, and plot to the chosen folder
#         if st.button('Save Indexed Image'):
#             im = Image.fromarray(grvi)
        
#             # Save the image to the folder
#             file_path = os.path.join(save_folder, f'Indexed Image.jpg')
#             pil_image.save(file_path)
#             st.success('Indexed image saved')
            
#         if st.button('Save Mask'):
#             im = Image.fromarray(masked_img)
#             file_path = os.path.join(save_folder, f'Indexed Image.jpg')
#             pil_image.save(file_path)
#             st.success('Indexed image saved')
#         if st.button('Save Plot'):
#             fig.savefig(f'{save_folder}/figure.png')
#             st.success('Plot saved')
# About page
if app_mode == 'About App':
    
    
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )

 
    st.markdown('''
                # About the app 
## Welcome to the GRVI Index and Mask Generator!

This app allows you to upload an image and apply various filters to it before calculating the GRVI index and generating a binary mask based on the index. You can also save the resulting indexed image, mask, and plot to a chosen folder.

The GRVI index is a measure of the greenness of a pixel in an image, and is calculated using the following formula:

GRVI = (G - R) / (G + R)

Where G is the green channel value and R is the red channel value. The resulting index values range from -1 to 1, with higher values indicating a higher degree of greenness.\n

We hope you find this app useful for your image processing needs. If you have any questions or suggestions, please don't hesitate to contact us.\n


                ''') 
