import cv2
import numpy as np
from skimage import morphology
from skimage.color import rgb2gray
import histomicstk.preprocessing.color_deconvolution as htk_cd



class decon_segmentor:
    def __init__(self, stain_matrix, th_psr = 0.5, th_yellow = 0.9):
        self.stain_matrix = stain_matrix
        self.th_psr = th_psr
        self.th_yellow = th_yellow
        
    def get_psr_stain(self, patch):
        decon_img =rgb2gray(htk_cd.color_deconvolution(im_rgb=patch, w = self.stain_matrix)[0])
        gray_psr = rgb2gray(patch)
        mask = decon_img <= self.th_psr
        mask = morphology.remove_small_objects(mask)
        mask = morphology.area_closing(mask)
        return gray_psr*mask, mask
    
    
    def get_yellow_stain(self, patch):
        decon_img =rgb2gray(htk_cd.color_deconvolution(im_rgb=patch, w = self.stain_matrix)[0])
        gray_psr = rgb2gray(patch)
        mask = (gray_psr>self.th_psr) & (gray_psr<=self.th_yellow)
        mask = morphology.remove_small_objects(mask)
        mask = morphology.area_closing(mask)
        return gray_psr*mask, mask
        

    def generate_rois(self, img, roi, white_background=True, thickness=3):
        """
        Generate ROI by masking the background of an image.

        Args:
            img (numpy.ndarray): Original image
            roi (numpy.ndarray): Binary mask defining the ROI
            white_background (bool): If True, use white background, else black
            thickness (int): Thickness of the border

        Returns:
            numpy.ndarray: Image with background masked out
        """
        # Ensure roi is a grayscale image
        if roi.ndim > 2 and roi.shape[2] > 1:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Resize roi to match image dimensions if needed
        if img.shape[:2] != roi.shape:
            roi = cv2.resize(roi, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Set background color based on preference
        background_color = [228, 228, 228] if white_background else [0, 0, 0]

        # Invert roi for processing
        roi = cv2.bitwise_not(roi)

        # Create a dilated mask for border pixels
        kernel = np.ones((thickness, thickness), np.uint8)
        eroded_roi = cv2.dilate(roi, kernel, iterations=1)
        (x_idx, y_idx) = np.where(eroded_roi == 255)

        # Apply background color to masked regions
        img_roi = img.copy()
        for row, col in zip(list(x_idx), list(y_idx)):
            img_roi[row, col, :] = np.array(background_color)

        return img_roi