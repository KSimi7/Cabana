import os
import cv2
import csv
import imutils
import convcrf
import argparse
import numpy as np
import torch.nn.init
from glob import glob
from tqdm import tqdm
from skimage import measure
import torch.optim as optim
from log import Log
from torch.autograd import Variable
from skimage.morphology import remove_small_objects, remove_small_holes
from models import BackBone, LightConv3x3
from utils import mean_image, cal_color_dist, save_result_video, read_bar_format
from typing import List, Tuple, Optional, Union


# Set fixed seed for reproducible results
SEED = 0
torch.use_deterministic_algorithms(True)

class RedFiberSegmenter:
    """
    Self-supervised segmentation for red fiber extraction in histological images.
    Designed for notebook use with numpy arrays as input/output.
    """
    
    def __init__(self, 
                 num_channels: int = 24,
                 max_iter: int = 35,
                 min_labels: int = 3,
                 hue_value: float = 0.0,
                 lr: float = 0.08,
                 sz_filter: int = 5,
                 rt: float = 0.18,
                 min_size: int = 32,
                 seed: int = 0):
        """
        Initialize the Red Fiber Segmenter.
        
        Args:
            num_channels: Number of channels in segmentation model (default: 24)
            max_iter: Maximum training iterations (default: 35)
            min_labels: Minimum labels to stop early (default: 3)
            hue_value: Target hue for red detection (default: 0.0 for red)
            lr: Learning rate (default: 0.08)
            sz_filter: CRF filter size (default: 5)
            rt: Relative threshold for color detection (default: 0.18)
            min_size: Minimum object size in pixels (default: 32)
            seed: Random seed for reproducibility (default: 0)
        """
        self.num_channels = num_channels
        self.max_iter = max_iter
        self.min_labels = min_labels
        self.hue_value = hue_value
        self.lr = lr
        self.sz_filter = sz_filter
        self.rt = rt
        self.min_size = min_size
        self.seed = seed
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(True)
    
    def segment_single_image(self, 
                           image: np.ndarray,
                           save_intermediate: bool = False,
                           save_dir: Optional[str] = None,
                           image_name: str = "image",
                           verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Segment a single image to extract red fibers.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            save_intermediate: Whether to save intermediate results
            save_dir: Directory to save results (if save_intermediate=True)
            image_name: Name for saved files
            verbose: Whether to show progress bar
            
        Returns:
            Tuple containing:
            - Binary mask (H, W) with red fiber regions as True
            - ROI image (H, W, 3) with background masked
            - Dictionary with metrics and intermediate results
        """
        
        # Validate input
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be (H, W, 3) RGB format")
        
        # Store original dimensions
        ori_height, ori_width = image.shape[:2]
        
        # Resize to 512 width while maintaining aspect ratio for processing
        if ori_width > 512:
            scale_factor = 512 / ori_width
            new_height = int(ori_height * scale_factor)
            img = cv2.resize(image, (512, new_height))
        else:
            img = image.copy()
            
        rgb_image = img.copy()
        img_size = img.shape[:2]
        
        # Prepare image for PyTorch processing
        img_tensor = img.transpose(2, 0, 1)  # Convert to channels-first
        data = torch.from_numpy(np.array([img_tensor.astype('float32') / 255.]))
        img_var = torch.Tensor(img_tensor.reshape([1, 3, *img_size]))
        
        # Initialize CRF
        config = convcrf.default_conf
        config['filter_size'] = self.sz_filter
        
        gausscrf = convcrf.GaussCRF(conf=config,
                                    shape=img_size,
                                    nclasses=self.num_channels,
                                    use_gpu=torch.cuda.is_available())
        
        # Initialize model
        model = BackBone([LightConv3x3], [2], [self.num_channels // 2, self.num_channels])
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        img_var = img_var.to(device)
        gausscrf = gausscrf.to(device)
        model = model.to(device)
        
        data = Variable(data)
        img_var = Variable(img_var)
        
        # Set up training
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        
        # Training loop
        intermediate_results = []
        pbar = tqdm(range(self.max_iter), desc="Segmenting", disable=not verbose)
        
        for batch_idx in pbar:
            # Forward pass
            optimizer.zero_grad()
            output = model(data)[0]
            unary = output.unsqueeze(0)
            prediction = gausscrf.forward(unary=unary, img=img_var)
            target = torch.argmax(prediction.squeeze(0), axis=0).reshape(img_size[0] * img_size[1])
            output = output.permute(1, 2, 0).contiguous().view(-1, self.num_channels)
            
            # Process prediction
            im_target = target.data.cpu().numpy()
            image_labels = im_target.reshape(img_size[0], img_size[1]).astype("uint8")
            num_labels = len(np.unique(im_target))
            
            # Store intermediate results if requested
            if save_intermediate:
                labels = measure.label(image_labels)
                mean_img = mean_image(rgb_image, labels)
                abs_color_dist, rel_color_dist = cal_color_dist(mean_img, self.hue_value)
                intermediate_results.append({
                    'iteration': batch_idx,
                    'num_labels': num_labels,
                    'mean_image': mean_img.copy(),
                    'color_distance': rel_color_dist.copy()
                })
            
            # Backward pass
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            # Update progress
            pbar.set_description(f"Iter {batch_idx}: {num_labels} labels, loss: {loss.item():.3f}")
            
            # Early stopping
            if num_labels <= self.min_labels:
                if verbose:
                    print(f"Early stopping: {num_labels} labels reached minimum {self.min_labels}")
                break
        
        # Final segmentation processing
        labels = measure.label(image_labels)
        mean_img = mean_image(rgb_image, labels)
        abs_color_dist, rel_color_dist = cal_color_dist(mean_img, self.hue_value)
        
        # Apply threshold and clean up
        thresholded = rel_color_dist > self.rt
        thresholded = remove_small_holes(thresholded, self.min_size)
        thresholded = remove_small_objects(thresholded, self.min_size)
        
        # Resize mask back to original size
        if ori_width > 512:
            mask = cv2.resize(thresholded.astype(np.uint8), (ori_width, ori_height), cv2.INTER_NEAREST)
        else:
            mask = thresholded.astype(np.uint8)
        
        # Generate ROI image
        roi_img = self._generate_roi(image, mask, white_background=True)
        
        # Calculate metrics
        total_pixels = ori_width * ori_height
        fiber_pixels = np.sum(mask)
        fiber_percentage = fiber_pixels / total_pixels
        
        results_dict = {
            'fiber_area_pixels': fiber_pixels,
            'fiber_percentage': fiber_percentage,
            'total_pixels': total_pixels,
            'final_num_labels': num_labels,
            'iterations_completed': batch_idx + 1,
            'intermediate_results': intermediate_results if save_intermediate else None
        }
        
        # Save results if requested
        if save_intermediate and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save mask
            cv2.imwrite(os.path.join(save_dir, f"{image_name}_mask.png"), 
                       (mask * 255).astype(np.uint8))
            
            # Save ROI
            roi_bgr = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"{image_name}_roi.png"), roi_bgr)
            
            # Save intermediate visualization if available
            if intermediate_results:
                self._save_intermediate_results(intermediate_results, save_dir, image_name)
        
        return mask.astype(bool), roi_img, results_dict
    
    def segment_batch(self, 
                     images: List[np.ndarray],
                     image_names: Optional[List[str]] = None,
                     save_intermediate: bool = False,
                     save_dir: Optional[str] = None,
                     verbose: bool = True) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
        """
        Segment a batch of images.
        
        Args:
            images: List of input images as numpy arrays (H, W, 3) in RGB format
            image_names: Optional list of names for each image
            save_intermediate: Whether to save intermediate results
            save_dir: Directory to save results
            verbose: Whether to show progress
            
        Returns:
            List of tuples, each containing (mask, roi_image, results_dict)
        """
        
        if image_names is None:
            image_names = [f"image_{i:03d}" for i in range(len(images))]
        
        if len(image_names) != len(images):
            raise ValueError("Number of image names must match number of images")
        
        results = []
        
        for i, (image, name) in enumerate(zip(images, image_names)):
            if verbose:
                print(f"\nProcessing image {i+1}/{len(images)}: {name}")
            
            try:
                mask, roi_img, results_dict = self.segment_single_image(
                    image=image,
                    save_intermediate=save_intermediate,
                    save_dir=save_dir,
                    image_name=name,
                    verbose=verbose
                )
                results.append((mask, roi_img, results_dict))
                
                if verbose:
                    print(f"✓ Completed {name}: {results_dict['fiber_percentage']:.2%} red fibers")
                    
            except Exception as e:
                print(f"✗ Error processing {name}: {str(e)}")
                # Return empty results for failed images
                empty_mask = np.zeros(image.shape[:2], dtype=bool)
                empty_roi = image.copy()
                empty_results = {'error': str(e)}
                results.append((empty_mask, empty_roi, empty_results))
        
        return results
    
    def _generate_roi(self, img: np.ndarray, mask: np.ndarray, 
                     white_background: bool = True) -> np.ndarray:
        """Generate ROI by masking background."""
        background_color = [228, 228, 228] if white_background else [0, 0, 0]
        
        roi_img = img.copy()
        roi_img[mask == 0] = background_color
        
        return roi_img
    
    def _save_intermediate_results(self, intermediate_results: List[dict], 
                                 save_dir: str, image_name: str):
        """Save intermediate results as plots."""
        if not intermediate_results:
            return
        
        # Plot convergence
        iterations = [r['iteration'] for r in intermediate_results]
        num_labels = [r['num_labels'] for r in intermediate_results]
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(iterations, num_labels, 'b-o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Number of Labels')
        plt.title('Segmentation Convergence')
        plt.grid(True, alpha=0.3)
        
        # Show final result
        plt.subplot(1, 2, 2)
        final_result = intermediate_results[-1]
        plt.imshow(final_result['color_distance'], cmap='hot')
        plt.title('Final Color Distance Map')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{image_name}_convergence.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()

# Simple wrapper functions for quick use:

def segment_single_image_simple(image: np.ndarray, 
                              hue_value: float = 0.0,
                              rt: float = 0.18,
                              verbose: bool = True) -> Tuple[np.ndarray, float]:
    """
    Simple wrapper for single image segmentation.
    
    Args:
        image: Input RGB image as numpy array
        hue_value: Target hue (0.0 for red)
        rt: Relative threshold (0.15-0.25 range)
        verbose: Show progress
        
    Returns:
        Binary mask and fiber percentage
    """
    segmenter = RedFiberSegmenter(hue_value=hue_value, rt=rt)
    mask, _, results = segmenter.segment_single_image(image, verbose=verbose)
    return mask, results['fiber_percentage']

def segment_batch_simple(images: List[np.ndarray],
                        hue_value: float = 0.0,
                        rt: float = 0.18) -> List[Tuple[np.ndarray, float]]:
    """
    Simple wrapper for batch segmentation.
    
    Args:
        images: List of RGB images as numpy arrays
        hue_value: Target hue (0.0 for red)
        rt: Relative threshold
        
    Returns:
        List of (mask, fiber_percentage) tuples
    """
    segmenter = RedFiberSegmenter(hue_value=hue_value, rt=rt)
    results = segmenter.segment_batch(images)
    return [(mask, res['fiber_percentage']) for mask, _, res in results]
