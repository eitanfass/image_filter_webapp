import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.colors as colors
import matplotlib
import os
from PIL import Image,ImageFilter


def GRVI(img):
  '''functon gets img path and boolian 
  values to deermin the image 
  filtering before applying 
  GRVI index on the img in the path. returns: img,index (as float imgs)'''
  
  img = img.convert("F")#format img to float_img
  img=np.array(img)
  return img,(img[:,:,1]-img[:,:,0])/(img[:,:,1]+img[:,:,0])# return img and indexed img



def show_index(img,index,index_name='RGVI'):# function that plots img and index, with colorbar, index_name will show in the title
  # define figure size: width 10 and height 15
  plt.figure(figsize=(10, 15))

  # subplot for the RGB
  ax1 = plt.subplot(121, title='Original RGB')
  im1 = ax1.imshow(img) 

  # subplot for the index
  ax2 = plt.subplot(122, title=f'{index_name} index, mean={np.nanmean(index):.3f} min={np.nanmin(index):.3f} max={np.nanmax(index):.3f}') # notice the position, and the title
  cmap=matplotlib.cm.get_cmap('Spectral_r',10)#set colormap to Spectral
  im2 = ax2.imshow(index,cmap=cmap)#show index in Spectral colormap
  im2.set_clim(vmax=index.max(), vmin=index.min())# set min max values of color to max and min values of index img
  
  # add colorbar only to the image on the right
  divider = make_axes_locatable(ax2)
  colorbar_ax = divider.append_axes("right", size="5%", pad=0.05)  
  plt.colorbar(im2, cax=colorbar_ax)


def create_mask_from_index(index,thresh=None):# function that creats a mask for an index img from a threshhold if provided
  if thresh==None:
    thresh=index.mean()#if no threshold is given it is set to index mean
  
  mask=index.copy()#copy index to mask
  mask[mask>thresh]=1# if pixel value is larger than threshold set value to 1 
  mask[mask<=thresh]=0  # if pixel value is smaller than threshold set value to 0
  return mask #returns a binery one layer mask


# Set the app title
st.title('GRVI Index and Mask Generator')
# add dropdown to select pages on left
app_mode = st.sidebar.selectbox('Navigate',
                                  ['About App', 'GRVI an Image'])
# About page
if app_mode == 'About App':
    st.markdown('In this app we will segment images using K-Means')
    
    
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
                ## About the app \n
                Welcome to the GRVI Index and Mask Generator!\n
                This app allows you to upload an image and apply various\n
                 filters to it before calculating the GRVI index \n
                 and generating a binary mask based on the index. \n
                 You can also save the resulting indexed image, mask,\n
                  and plot to a chosen folder.The GRVI index is a measure\n
                   of the greenness of a pixel in an image, and is calculated using the following formula:\n
$GRVI = (G - R) / (G + R)$\n
Where G is the green channel value and R is the red channel value. The resulting index values range from -1 to 1, with higher values indicating a higher degree of greenness.\n
We hope you find this app useful for your image processing needs. If you have any questions or suggestions, please don't hesitate to contact us.\n


                ''') 
# Run image
if app_mode == 'GRVI an Image':
    
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

    if uploaded_image is not None:
        # Read the image and convert to a NumPy array
        image = Image.open(uploaded_image)
        # Calculate the GRVI index
        _, index = GRVI(image)

        # Create a binary mask from the index using the mean value as the threshold
        mask = index.copy()
        mask[mask > index.mean()] = 1
        mask[mask <= index.mean()] = 0

        # Show the indexed image, mask, and original image side by side
        show_index(image, index, mask)
        st.pyplot()

        # Allow the user to choose a destination folder for the saved images
        save_folder = st.folder_selector('Save images to:', default='.')

        # Save the indexed image, mask, and plot to the chosen folder
        if st.button('Save Indexed Image'):
            im = Image.fromarray(index)
            im.save(os.path.join(save_folder, 'indexed_image.jpg'))
            st.success('Indexed image saved')
            
        # Allow the user to save the indexed image, mask, and plot
        if st.button('Save Indexed Image'):
            st.image(index, width=300)
            st.success('Indexed image saved')
        if st.button('Save Mask'):
            st.image(mask, width=300)
            st.success('Mask saved')
        if st.button('Save Plot'):
            st.pyplot(width=600, height=600)
            st.success('Plot saved')
