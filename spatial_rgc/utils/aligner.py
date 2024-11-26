import cv2
import matplotlib.pyplot as plt
import numpy as np
from cellpose import io
import os
from scipy import ndimage
import pandas as pd
import gc

from PIL import Image


class Aligner:
    def __init__(self,image_paths,centers=None,rotations=None):
        self.image_paths = image_paths
        if centers is None:
            self.centers  = dict()
        else:
            self.centers = centers

        self.subdirectories = []
        for ip in image_paths:
            subdirectory = ip.rsplit("/")[-2] # Retrives directory each image is in
            self.subdirectories.append(subdirectory)

        self.centers_micron = None
        self.all_pads = None
        self.rotations=rotations

    def set_center(self, i,center):
        self.centers[i] = center

    def visualize(self, indices,half_w=50,run="140g_rn3",figure_dir="spatial_figures",show_center=False,invert_xaxis=False,invert_yaxis=False):
        images = self._load_images(indices)
        for i in indices:
            im = images[i]
            """
            plt.figure(figsize=(16, 12), dpi=200)
            plt.imshow(im,cmap='Greys')
            plt.axis('off')
            plt.savefig(os.path.join("spatial_figures",run,f"{run}_{i}.tif"))
            plt.show()
            """
            print(self.centers)
            if show_center:
                if i in self.centers:
                    center = self.centers[i]
                    print(center,half_w)
                    im[center[0] - half_w : center[0]+half_w, center[1]-half_w: center[1]+half_w] = -1
            plt.imshow(im,cmap='Greys')
            if invert_yaxis:
                plt.gca().invert_yaxis()
            if invert_xaxis:
                plt.gca().invert_xaxis()
            plt.gca().axis('off')
            plt.savefig(os.path.join(figure_dir,f"{run}_img_{i}.png"))
            plt.show()


    def _load_images(self,indices):
        images = []
        for i in indices:
            im = io.imread(self.image_paths[i])
            images.append(im)
        return images

    # Center images, and plot image2 on top of image1
    def overlay(self, images, i1,i2,rotate2):
        im1 = images[i1]
        im2 = self._rotate(images[i2],angle=rotate2,center=(self.centers[i2][1],self.centers[i2][0]))
        
        l1,l2 = self.centers[i1][1],self.centers[i2][1]
        r1,r2 = (im1.shape[1] - self.centers[i1][1]), (im2.shape[1] - self.centers[i2][1])
        d1,d2  =self.centers[i1][0],self.centers[i2][0] # d = 0 to center
        u1,u2 = (im1.shape[0] - self.centers[i1][0]), (im2.shape[0] - self.centers[i2][0])

        l = np.max(l2-l1, 0)
        d = np.max(d2-d1, 0)
        u = np.max(u2-u1, 0)
        r = np.max(r2-r1, 0)

        pad_arg = ((d,u), (l,r))
        im1 = np.pad(im1, pad_arg)
        im2_pad = np.zeros((im1.shape[0], im1.shape[1])) # New bigger image for im2, which we will replace a section of with im2
        if l > 0:
            x = 0
        else:
            x= l1-l2
        if d > 0:
            y = 0
        else:
            y = d1-d2
        

        im2_pad[y:y+im2.shape[0], x:x+im2.shape[1]] = im2
        plt.imshow(im1,cmap='Blues',alpha=1)
        plt.imshow(im2_pad,cmap='Reds',alpha=1)
        # plt.imshow(np.maximum(im1,im2_pad))
        # r = self.centers[]
        del im1,im2,images,im2_pad
        gc.collect()


    def jigsaw(self,indices,rotate2=0):
        images = self._load_images(indices)
        self.overlay(images,0,1,rotate2)
    
    """Return transformation vector (x,y) that should be applied to coordinate system 2 to align it with 1"""
    def get_align_transform(self,i1,i2,dr,mappings_path=None):
        centers_micron = []
        for i in [i1,i2]:
            subdirectory = self.subdirectories[i]
            if mappings_path is None:
                A_p = os.path.join(os.getcwd(), "imaging_scripts", "mappings", subdirectory, "micron_to_mosaic_pixel_transform.csv")
            else:
                A_p = os.path.join(os.path.join(mappings_path,subdirectory,"micron_to_mosaic_pixel_transform.csv"))
            A = pd.read_csv(A_p, header=None, sep=' ').values
            center = np.array([self.centers[i][1], self.centers[i][0]])# Convert to (x,y)
            centers_micron.append(mosaic_to_micron(A,center,dr))
            
        self.centers_micron = centers_micron
        return centers_micron[0]-centers_micron[1]
    
    def overlay_images(self, base_idx, indices,save_f,use_multi_colors=True,run="140g_rn3",invert_xaxis=False,invert_yaxis=False):
        assert self.rotations is not None, "Set rotation variable in aligner first"
        assert base_idx in indices
        images = self._load_images(indices)
        all_pads = np.zeros((len(indices),4),dtype=np.int64) # Stores padding for each image

        for idx in indices:
            # Note: Compares based image to itself, which should always return [0,0,0,0]
            image_pad = self._determine_padding(images,base_idx,idx)
            all_pads[idx] = image_pad

        d,u,l,r = np.max(all_pads,axis=0) # Get maximum padding (e.g. padding that allows all images to fit)
        base_im = np.pad(images[base_idx], ((d,u),(l,r)))
        if use_multi_colors:
            cmaps = ['Reds', 'Blues','Greens']
        else:
            cmaps = ['Greys']
        center_base = self.centers[base_idx][1],self.centers[base_idx][0] # makes it (x,y) aka (col, row)
        im2pad=None
        for idx in indices:
            im2 = images[idx]
            center = self.centers[idx][1],self.centers[idx][0] # makes it (x,y) aka (col, row)
            rotation = self.rotations[idx]
            cmap = cmaps[idx%len(cmaps)]
            if im2pad is None:
                im2pad = self._create_padded_image(base_im, im2,center_base, center,(d,u,l,r),rotation,cmap)
            else:
                im2pad = np.maximum(im2pad,self._create_padded_image(base_im, im2,center_base, center,(d,u,l,r),rotation,cmap))
        """
        plt.savefig(save_f)
        plt.show()
        plt.figure(figsize=(16, 12), dpi=200)
        # cv2.imwrite(os.path.join("spatial_figures",run,f"overlayed_img.tiff"),im2pad)
        """
        plt.imshow(im2pad)
        plt.savefig(save_f)
        plt.show()
        plt.imshow(im2pad,cmap='Greys')
        plt.axis('off')
        if invert_xaxis:
            plt.gca().invert_xaxis()
        if invert_yaxis:
            plt.gca().invert_yaxis()
        # plt.savefig(os.path.join("spatial_figures",run,f"overlayed_img.tif"))
        # im = Image.fromarray(im2pad, mode='F') # float32
        # im.save(os.path.join("spatial_figures",run,f"overlayed_img.tiff"), "TIFF")
        del base_im, images
        gc.collect()
    
    def _determine_padding(self,images,i1,i2):
        im1 = images[i1]
        im2 = images[i2]
        
        l1,l2 = self.centers[i1][1],self.centers[i2][1]
        d1,d2  =self.centers[i1][0],self.centers[i2][0] # d = 0 to center
        u1,u2 = (im1.shape[0] - self.centers[i1][0]), (im2.shape[0] - self.centers[i2][0])
        r1,r2 = (im1.shape[1] - self.centers[i1][1]), (im2.shape[1] - self.centers[i2][1])

        l = np.max(l2-l1, 0)
        d = np.max(d2-d1, 0)
        u = np.max(u2-u1, 0)
        r = np.max(r2-r1, 0)
        return np.array([d,u,l,r])
    

        
    def _create_padded_image(self,base_im,im2,center_base,center,pad,rotation,cmap):
        im2 = self._rotate(im2,angle=rotation,center=center)
        im2_pad = np.zeros((base_im.shape[0], base_im.shape[1])) # New bigger image for im2, which we will replace a section of with im2
        d,u,l,r = pad
        
        x = center_base[0]+l-center[0]
        y = center_base[1]+d-center[1]

        im2_pad[y:y+im2.shape[0], x:x+im2.shape[1]] = im2
        # plt.imshow(im2_pad,cmap=cmap,alpha=0.1)
        #del im2_pad
        #gc.collect()
        return im2_pad


    
    # Center = (col, row)
    def _rotate(self, image, angle, center = None, scale = 1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

"""
Input: micron to mosaic transformation matrix,coords np.array([x,y]) of downsampled mosaic point, downsample ratio used
Return: micron coordinates (x,y) after resampling and inverse transformation
"""
def mosaic_to_micron(A,mosaic_coords, dr):
    mosaic_coords_aug = np.array([mosaic_coords[0]*dr,mosaic_coords[1]*dr,1])
    A_inv = np.linalg.inv(A)
    micron_coords = (A_inv@mosaic_coords_aug)[:-1]
    return micron_coords


def rotation_matrix(rotat_degree):
    rad =  rotat_degree/180*np.pi
    return np.array([[np.cos(rad), -np.sin(rad)],[np.sin(rad),np.cos(rad)]])


