import os
import cv2
import yaml
import shutil
import warnings
import numpy as np
from scipy import stats
import pandas as pd
from log import Log
from hdm import HDM
from glob import glob
from PIL import Image
import tifffile as tiff
import imageio.v3 as iio
import json
from pathlib import Path
from detector import FibreDetector
from analyzer import SkeletonAnalyzer
from skimage.feature import peak_local_max
from skimage.morphology import area_closing
from orientation import OrientationAnalyzer
from skimage.color import rgb2hed, hed2rgb, rgb2gray
from sklearn.metrics.pairwise import euclidean_distances
from utils import create_folder, join_path, mask_color_map
from segmenter import parse_args, segment_single_image, visualize_fibres
from utils import overlay_colorbar, color_survey_with_colorbar
from color_decon_segmentor import decon_segmentor



class Cabana_stain_analyzer:
    def __init__(self, param_file, stain_matrix, th_psr = 0.5, 
                th_yellow = 0.9, ims_res = 1):
        self.param_file = param_file
        self.stain_matrix = stain_matrix
        self.th_psr = th_psr
        self.th_yellow = th_yellow

        # Initialize parameters
        self.args = None  # args for Cabana program
        self.seg_args = parse_args()  # args for segmentation
        self.ims_res = ims_res  # µm/pixel
        
        self.metrics = self.get_metric_names()
        
        self.initialize_params()
        self.color_segmentor = decon_segmentor(self.stain_matrix, self.th_psr, self.th_yellow)
        self.detector = self.getFiberDetector()
        self.orient_analyzer = OrientationAnalyzer(2.0)
        self.hdmanalyzer = self.getHDManalyzer()
        
        
    def initialize_params(self):
        """Initialize parameters from the parameter file"""
        with open(self.param_file) as pf:
            try:
                self.args = yaml.safe_load(pf)
            except yaml.YAMLError as exc:
                print(exc)

        # overwrite specific fields of seg_args with those in the parameter file
        setattr(self.seg_args, 'rt', float(self.args['Segmentation']["Color Threshold"]))
        setattr(self.seg_args, 'max_size', int(self.args['Segmentation']["Max Size"]))
        setattr(self.seg_args, 'min_size', int(self.args['Segmentation']["Min Size"]))
        setattr(self.seg_args, 'white_background', self.args['Detection']["Dark Line"])    


    def get_metric_names(self):
        with open('metrics.txt', 'r') as f:
            metrics = json.load(f)
        return metrics
    
    def getFiberDetector(self):
        """Get Fiber Detector"""
        dark_line = self.args["Detection"]["Dark Line"]
        extend_line = self.args["Detection"]["Extend Line"]
        min_line_width = self.args["Detection"]["Min Line Width"]
        max_line_width = self.args["Detection"]["Max Line Width"]
        line_width_step = self.args["Detection"]["Line Width Step"]
        line_widths = np.arange(min_line_width, max_line_width + line_width_step, line_width_step)
        low_contrast = self.args["Detection"]["Low Contrast"]
        high_contrast = self.args["Detection"]["High Contrast"]
        min_len = self.args["Detection"]["Minimum Line Length"]
        max_len = self.args["Detection"]["Maximum Line Length"]
        
        det = FibreDetector(line_widths=line_widths,
                            low_contrast=low_contrast,
                            high_contrast=high_contrast,
                            dark_line=dark_line,
                            extend_line=extend_line,
                            correct_pos=False,
                            min_len=min_len,
                            max_len=max_len)
        return det
    
    
    def getHDManalyzer(self):
        """Get high density matrix analyzer"""
        max_hdm = self.args["Quantification"]["Maximum Display HDM"]
        sat_ratio = self.args["Quantification"]["Contrast Enhancement"]
        dark_line = self.args["Detection"]["Dark Line"]

        hdm = HDM(max_hdm=max_hdm, sat_ratio=sat_ratio, dark_line=dark_line)
        return hdm
    
    
    def get_psr_roi(self, image):
        _, mask = self.color_segmentor.get_psr_stain(image)
        mask = area_closing(mask, area_threshold=16)
        _, yellow_mask = self.color_segmentor.get_yellow_stain(image)
        yellow_mask = area_closing(yellow_mask, area_threshold=16) 
        psr_roi = self.color_segmentor.generate_rois(image, mask.astype(np.uint8), thickness=1)
        yellow_roi = self.color_segmentor.generate_rois(image, yellow_mask.astype(np.uint8), thickness=1)
        return mask, psr_roi, yellow_roi, yellow_mask
    
    
    def get_fiber_detector_output(self, roi, mask):
        
        self.detector.detect_lines(roi)
        contour_img, width_img, binary_contours, binary_widths, int_width_img = self.detector.get_results()

        ## filter artifacts in the fiber detection
        contour_img[~mask] = contour_img.max()
        width_img[~mask] = width_img.max()
        binary_contours[~mask] = binary_contours.max()
        int_width_img[~mask] = 0
        binary_widths[~mask] = binary_widths.max()
        binary_widths = area_closing(binary_widths, area_threshold=16)
        
        contours = self.detector.contours
        junctions = self.detector.junctions
        
        return contour_img, width_img, binary_contours, binary_widths, int_width_img, contours, junctions
    
    
    def get_hdm_output(self, image):
        
        result, contrast_img = self.hdmanalyzer.quantify_black_space_single_image(image)
        return result, contrast_img
    
    
    def get_Orientation_results(self, roi, mask):
        
        self.orient_analyzer.compute_orient(roi)
        energy_img = self.orient_analyzer.get_energy_image()
        coherency_img = self.orient_analyzer.get_coherency_image()
        orientation_img = self.orient_analyzer.get_orientation_image()
        orient_color_survey = self.orient_analyzer.draw_color_survey()
        vector_field = self.orient_analyzer.draw_vector_field(mask)
        angular_hist = self.orient_analyzer.draw_angular_hist(mask = mask.astype(np.uint8))
        
        orientation_metrics  = {}
        
        orientation_metrics['mean coherency - anisotropy'] = self.orient_analyzer.mean_coherency(mask = mask)
        orientation_metrics['dominant angle'] = self.orient_analyzer.mean_orientation(mask = mask)
        orientation_metrics['Orient. Variance'] = self.orient_analyzer.circular_variance(mask = mask)
        # orientation_metrics['randomness_orientation'] = self.orient_analyzer.randomness_orientation(mask=mask)
        
        return energy_img, coherency_img, orientation_img, orient_color_survey, vector_field, angular_hist, orientation_metrics
    
    
    def get_skeleton_analysis(self, binary_contours):
        
        min_curve_win = int(self.args["Quantification"]["Minimum Curvature Window"])
        max_curve_win = int(self.args["Quantification"]["Maximum Curvature Window"])
        curve_win_step = int(self.args["Quantification"]["Curvature Window Step"])
        
        self.skeletonAnalyzer.analyze_image(binary_contours)
        keypnts_img = self.skeletonAnalyzer.key_pts_image
    
        
        fiber_metrics = {}
        
        fiber_metrics['Area of Fiber Spines'] = (self.skeletonAnalyzer.proj_area) * self.ims_res**2
        fiber_metrics['Lacunarity'] = self.skeletonAnalyzer.lacunarity
        fiber_metrics['Total Length (µm)'] = self.skeletonAnalyzer.total_length * self.ims_res
        fiber_metrics['Endpoints'] = self.skeletonAnalyzer.num_tips
        fiber_metrics['Avg Length (µm)'] = self.skeletonAnalyzer.growth_unit * self.ims_res
        fiber_metrics['Branchpoints'] = self.skeletonAnalyzer.num_branches
        fiber_metrics['Fractal Dim'] = self.skeletonAnalyzer.frac_dim
        
        image_area = (512**2)*(self.ims_res**2)
        ratio_norm = fiber_metrics['Total Length (µm)'] * self.ims_res / image_area
        if 0<ratio_norm<1:
            fiber_metrics['Lacunarity Normalized'] = fiber_metrics['Lacunarity']*((1-ratio_norm)/ratio_norm)
        else:
            fiber_metrics['Lacunarity Normalized'] = 0
            
        fiber_metrics['Branchpoints Density'] = fiber_metrics['Branchpoints']/fiber_metrics['Total Length (µm)']
        fiber_metrics['Endpoints Density'] = fiber_metrics['Endpoints']/fiber_metrics['Total Length (µm)']
        
        # fiber_metrics['total image area'] = np.prod(self.skeletonAnalyzer.skel_image.shape[:2]) * self.ims_res ** 2
        
        curvature_maps = []
        for win_sz in np.arange(min_curve_win, max_curve_win + curve_win_step, curve_win_step):
            self.skeletonAnalyzer.calc_curve_all(win_sz)
            curve_key = f"Curvature (win_sz={win_sz})"
            fiber_metrics[curve_key] = self.skeletonAnalyzer.avg_curve_all
            curvature_maps.append(self.skeletonAnalyzer.curve_map_all)
            
        return self.skeletonAnalyzer.skel_image, keypnts_img, fiber_metrics, curvature_maps
    
    
    def analyze_gaps(self, image, binary_contours):
        """Analyze gaps in the image"""
        min_gap_diameter = self.args["Gap Analysis"]["Minimum Gap Diameter"]
        if min_gap_diameter == 0:
            warnings.warn("minimum gap diameter = 0 pixels. Skipping gap analysis.")
            return
        
        min_gap_radius = min_gap_diameter / 2
        min_dist = int(np.max([1, min_gap_radius]))

        mask = binary_contours.copy()

        # Set border pixels to zero to avoid partial circles
        mask[0, :] = mask[-1, :] = mask[:, :1] = mask[:, -1:] = 0

        final_circles = []
        downsample_factor = 2

        while True:
            dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

            # Downsample distance map and upscale detected centers to original image size
            dist_map_downscaled = cv2.resize(dist_map, None, fx=1 / downsample_factor, fy=1 / downsample_factor)
            centers_downscaled = peak_local_max(dist_map_downscaled, min_distance=min_dist, exclude_border=False)
            centers = centers_downscaled * downsample_factor

            radius = dist_map[centers[:, 0], centers[:, 1]]

            eligible_centers = centers[radius > min_gap_radius, :]
            eligible_radius = radius[radius > min_gap_radius]
            eligible_circles = np.hstack([eligible_centers, eligible_radius[:, None]])

            if len(eligible_circles) == 0:
                break

            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            while len(eligible_circles) > 0:
                if eligible_circles[1:, :].size > 0:
                    pw_euclidean_dist = \
                        euclidean_distances(eligible_circles[[0], :2], eligible_circles[1:, :2])[0]
                    pw_radius_sum = eligible_circles[0, 2] + eligible_circles[1:, 2]
                    neighbor_idx = np.nonzero(pw_euclidean_dist < pw_radius_sum)[0] + 1
                    eligible_circles = np.delete(eligible_circles, neighbor_idx, axis=0)

                circle = eligible_circles[0, :]
                result = cv2.circle(result, (int(circle[1]), int(circle[0])), int(circle[2]), (0, 0, 0), -1)
                final_circles.append(eligible_circles[0, :])
                eligible_circles = np.delete(eligible_circles, 0, axis=0)

            mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Create visualizations
        final_result = cv2.cvtColor(binary_contours, cv2.COLOR_GRAY2BGR)
        color_result = image.copy()

        for circle in final_circles:
            final_result = cv2.circle(final_result, (int(circle[1]), int(circle[0])),
                                    int(circle[2]), (0, 255, 255), 2)
            color_result = cv2.circle(color_result, (int(circle[1]), int(circle[0])),
                                    int(circle[2]), (0, 255, 255), 2)
            
        areas = np.pi * (np.array(final_circles)[:, 2] ** 2) * self.ims_res ** 2
        stats = {}

        if len(areas) > 0:
            # Store gap metrics
            stats['Mean (All gaps area in µm²)'] = np.mean(areas)
            stats['Std (All gaps area in µm²)'] = np.std(areas)
            stats['Percentile5 (All gaps area in µm²)'] = np.percentile(areas, 5)
            stats['Median (All gaps area in µm²)'] = np.median(areas)
            stats['Percentile95 (All gaps area in µm²)'] = np.percentile(areas, 95)
            stats['Gap Circles Count (All)'] = areas.size

            # Convert to radius metrics
            radius_values = np.sqrt(areas / np.pi)
            stats['Mean (All gaps radius in µm)'] = np.mean(radius_values)
            stats['Std (All gaps radius in µm)'] = np.std(radius_values)
            stats['Median (All gaps radius in µm)'] = np.median(radius_values)
            stats['Percentile5 (All gaps radius in µm)'] = np.percentile(radius_values, 5)
            stats['Percentile95 (All gaps radius in µm)'] = np.percentile(radius_values, 95)
        else:
            # Fill with zeros if no gaps found
            gap_metrics = ['Mean (All gaps area in µm²)', 'Std (All gaps area in µm²)',
                        'Percentile5 (All gaps area in µm²)', 'Median (All gaps area in µm²)',
                        'Percentile95 (All gaps area in µm²)', 'Gap Circles Count (All)',
                        'Mean (All gaps radius in µm)', 'Std (All gaps radius in µm)',
                        'Median (All gaps radius in µm)', 'Percentile5 (All gaps radius in µm)',
                        'Percentile95 (All gaps radius in µm)']

            for metric in gap_metrics:
                stats[metric] = 0

        # Save gap data to CSV
        if len(final_circles) > 0:
            final_circles = np.array(final_circles)
            data = {
                'Area (µm²)': areas,
                'X': final_circles[:, 1],
                'Y': final_circles[:, 0]
            }
            final_circles_df = pd.DataFrame(data)

        return final_circles_df, stats, final_result, color_result
    
    
    def analyze_intra_gaps(self, final_circles_df, binary_mask, binary_contours, image):
        """Analyze gaps within the ROI"""
        if not self.args["Configs"]["Gap Analysis"] or self.args["Gap Analysis"]["Minimum Gap Diameter"] == 0:
            return
        
        color_img_fibre = cv2.cvtColor(binary_contours, cv2.COLOR_GRAY2BGR)
        color_img = image
        areas = []
        circle_cnt = 0

        for index, row in final_circles_df.iterrows():
            area, x, y = row['Area (µm²)'], int(row['X']), int(row['Y'])
            radius = int(np.sqrt(area / np.pi) / self.ims_res)  # convert back to measurements in pixels
            if binary_mask[y, x] > 0:
                color_img_fibre = cv2.circle(color_img_fibre, (x, y), radius, (0, 255, 255), 1)
                color_img = cv2.circle(color_img, (x, y), radius, (0, 255, 255), 1)
                areas.append(area)
                circle_cnt += 1

        areas = np.array(areas)
        stats = {}

        if len(areas) > 0:
            radius = np.sqrt(areas / np.pi)

            # Store metrics
            stats['Mean (ROI gaps area in µm²)'] = np.mean(areas)
            stats['Std (ROI gaps area in µm²)'] = np.std(areas)
            stats['Percentile5 (ROI gaps area in µm²)'] = np.percentile(areas, 5)
            stats['Median (ROI gaps area in µm²)'] = np.median(areas)
            stats['Percentile95 (ROI gaps area in µm²)'] = np.percentile(areas, 95)
            stats['Mean (ROI gaps radius in µm)'] = np.mean(radius)
            stats['Std (ROI gaps radius in µm)'] = np.std(radius)
            stats['Percentile5 (ROI gaps radius in µm)'] = np.percentile(radius, 5)
            stats['Median (ROI gaps radius in µm)'] = np.median(radius)
            stats['Percentile95 (ROI gaps radius in µm)'] = np.percentile(radius, 95)
            stats['Gap Circles Count (ROI)'] = circle_cnt
        else:
            # Fill with zeros if no gaps found
            intra_gap_metrics = ['Mean (ROI gaps area in µm²)', 'Std (ROI gaps area in µm²)',
                                'Percentile5 (ROI gaps area in µm²)', 'Median (ROI gaps area in µm²)',
                                'Percentile95 (ROI gaps area in µm²)', 'Gap Circles Count (ROI)',
                                'Mean (ROI gaps radius in µm)', 'Std (ROI gaps radius in µm)',
                                'Median (ROI gaps radius in µm)', 'Percentile5 (ROI gaps radius in µm)',
                                'Percentile95 (ROI gaps radius in µm)']

            for metric in intra_gap_metrics:
                stats[metric] = 0

        return stats, color_img, color_img_fibre
    
    
    def calc_fiber_areas(self, binary_mask, binary_widths):
        
        area_roi = np.count_nonzero(binary_mask)
        percent_roi = area_roi / binary_mask.shape[0] / binary_mask.shape[1]  # % ROI area
        
        area_width = np.count_nonzero(binary_widths)
        percent_width = area_width/ binary_mask.shape[0] / binary_mask.shape[1]
        
        stats = {}
        
        stats['perct_area_roi'] = percent_roi
        stats['perct_area_width'] = percent_width
        
        return stats
    
    
    def get_fiber_orientation_metrics(self, contour_angles, weights = None):
        """
        takes angles from detector functions for each fiber and outputs
        mean_angle - circular mean direction of all angles in radian
        Resultant_length (R) - consistency of the angular distribution [0, angles uniformly distributed no direction 1, concentrated around mean_angle]
        anisotropy - deviation from isotropy [positive - alignment, negative - perpendicular orientations, 0 isotropic]
        
        Parameters
        ----------
        contour_angles : list
            list of angles from contours 
        weights : np.ndarray
            weight to be assigned to each line segment
            
        
        Returns
        -------
        mean_angle : float 
            in radian
        R : float [0,1]
        anisotropy : float
        
        """
        
        all_angles_flat = np.concatenate(contour_angles)  # Flatten all angles
        if len(all_angles_flat) == 0:
            return 0.0, 0.0, 0.0
        
        n = len(contour_angles)
        segment_lengths = np.array([len(seg) for seg in contour_angles])
        
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.asarray(weights)
            weights /= np.sum(weights)
        
        # Repeat weights per angle in each segment
        angle_weights = np.repeat(weights / segment_lengths, segment_lengths)
        angle_weights /= np.sum(angle_weights)  # Normalize total
        
        C = np.sum(angle_weights * np.cos(all_angles_flat))
        S = np.sum(angle_weights * np.sin(all_angles_flat))
        
        R = np.sqrt(C**2 + S**2)
        mean_angle = np.arctan2(S, C)
        
        # Anisotropy
        anisotropy = np.sum(angle_weights * np.cos(2 * (all_angles_flat - mean_angle)))
        
        return mean_angle, R, anisotropy
    
    
    def get_contour_metrics(self, contours, junctions):
        
        lengths = np.array([contour.estimate_length() for contour in contours])
        angles = [contour.get_angle() for contour in contours]
        
        thickness = np.empty(len(contours))
        asymmetries = np.empty(len(contours))
        
        for i, contour in enumerate(contours):
            width_l = np.asarray(contour.get_line_width_l())
            width_r = np.asarray(contour.get_line_width_r())
            total_width = width_l + width_r
            thickness[i] = np.mean(total_width)
            if np.any(total_width > 0):
                asymmetry = np.abs(width_l - width_r) / total_width
                asymmetries[i] = np.mean(asymmetry)
            else:
                asymmetries[i] = 0.0
                
        return lengths, thickness, asymmetries, angles, len(contours), len(junctions)
    
    
    def get_enhanced_distribution_features(self, values_array, prefix):
        """Optimized: Early exit, vectorized stats."""
        values = np.asarray(values_array)
        if len(values) == 0 or np.all(values == 0):
            return {f'{prefix}_{stat}': 0 for stat in 
                    ['mean', 'median', 'std', 'skew', 'kurtosis', 'q25', 'q75', 'iqr', 'entropy', 'cv']}
        
        features = {}
        features[f'{prefix}_mean'] = np.mean(values)
        features[f'{prefix}_median'] = np.median(values)
        features[f'{prefix}_std'] = np.std(values)
        features[f'{prefix}_cv'] = features[f'{prefix}_std'] / features[f'{prefix}_mean'] if features[f'{prefix}_mean'] > 0 else 0
        features[f'{prefix}_skew'] = stats.skew(values)
        features[f'{prefix}_kurtosis'] = stats.kurtosis(values)
        q25, q75 = np.percentile(values, [25, 75])
        features[f'{prefix}_q25'] = q25
        features[f'{prefix}_q75'] = q75
        features[f'{prefix}_iqr'] = q75 - q25
        
        hist, _ = np.histogram(values, bins=min(20, len(values)//2 + 1), density=True)
        hist = hist[hist > 0]
        features[f'{prefix}_entropy'] = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
        
        return features
    
    
    def get_all_contour_metrics_normalized(self, contours, junctions, im_res = None):
        '''
        '''
        lengths, thickness, asymmetries, angles, num_fibers, num_branches = self.get_contour_metrics(contours, junctions)
        mean_angle, R, anisotropy = self.get_fiber_orientation_metrics(angles)
        
        metrics = [lengths, thickness, asymmetries]
        metric_names = ['length_contour', 'thickness_contour', 'asymmetry_contour']
        
        if im_res:
            metrics.append(lengths*im_res)
            metrics.append(thickness*im_res)
            metrics.append(asymmetries*im_res)
            metric_names.append('lengths_contour_um')
            metric_names.append('thickness_contour_um')
            metric_names.append('asymm_contour_um')
            
        stats = {}
        for metric, metric_name in zip(metrics, metric_names):
            stats.update(self.get_enhanced_distribution_features(metric, metric_name))
            
        stats['mean_angle'] = mean_angle
        stats['R'] = R
        stats['anisotropy'] = anisotropy
        stats['num fibers'] = num_fibers
        stats['num branches'] = num_branches
        
        return stats
    
        
    
    def output_for_blank_patches(self):
        blank_stats = {}
        for i in self.metrics:
            blank_stats[i] = 0
        return blank_stats
    
    
    def perform_analysis_on_single_image(self, image):
        
        output_images = {}
        collagen_morphometrics = {}
        binary_mask, roi = self.get_psr_roi(image)
        ratio_roi = np.count_nonzero(binary_mask)/(image.shape[0]**2)
        if ratio_roi <= 0.025:
            return [], self.output_for_blank_patches()
        
        else:
            
            contour_img, width_img, binary_contours, binary_widths, int_width_img, contours, junctions = self.get_fiber_detector_output(roi, binary_mask)
            contour_metrics = self.get_all_contour_metrics_normalized(contours, junctions, self.ims_res)
            hdm_metrics = {}
            hdm_roi, contrast_img_roi = self.get_hdm_output(roi)
            hdm_image, contrast_img = self.get_hdm_output(image)
            hdm_metrics['perct_hdm_area_roi'] = hdm_roi['% HDM Area']
            hdm_metrics['perct_hdm_area_image'] = hdm_image['% HDM Area']
            
            # TODO: Fix angular hist -- it is a blank image
            energy_img, coherency_img, orientation_img, orient_color_survey, vector_field, angular_hist, orientation_metrics = self.get_Orientation_results(roi, binary_mask)
            
            skel_img, keypnts_img, fiber_metrics, curvature_maps = self.get_skeleton_analysis(binary_contours)
            
            area_metrics = self.calc_fiber_areas(binary_mask, binary_widths)
            
            final_circle_df, gap_metrics, gap_image_fiber, gap_image = self.analyze_gaps(image, binary_contours)
            intra_gap_metrics, intra_gap_image, intra_gap_image_fiber = self.analyze_intra_gaps(final_circle_df, binary_mask, binary_contours, image)
            
            
            output_images['binary_mask'] = binary_mask
            output_images['roi'] = roi
            output_images['contour img'] = contour_img
            output_images['width_img'] = width_img
            output_images['binary_contours'] = binary_contours
            output_images['binary_widths'] = binary_widths
            output_images['contrast_image'] = contrast_img
            output_images['contrast_image_roi'] = contrast_img_roi
            output_images['gap_image_fiber'] = gap_image_fiber
            output_images['gap_image'] = gap_image
            output_images['intra_gap_image'] = intra_gap_image
            output_images['intra_gap_image_fiber'] = intra_gap_image_fiber
            output_images['energy_images'] = energy_img
            output_images['coherence_img'] = coherency_img
            output_images['orientation_img'] = orientation_img
            output_images['orient_color_survey'] = orient_color_survey
            output_images['vector_field'] = vector_field
            # output_images['angular_hist'] = angular_hist
            output_images['skel_img'] = skel_img
            output_images['keypnts_img'] = keypnts_img
            output_images['curvature_maps'] = curvature_maps
            
            for i in [hdm_metrics, orientation_metrics, fiber_metrics, gap_metrics, intra_gap_metrics, area_metrics, contour_metrics]:
                collagen_morphometrics.update(i)
            
            return output_images, collagen_morphometrics