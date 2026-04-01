# from skimage.color import rgb2gray
# from skimage.util import invert
# import astropy.units as u
# from fil_finder import FilFinder2D
# from scipy.stats import circmean, circvar
# import numpy as np


# class FiberSegmentDetector:
#     def __init__(self, length_th = 15):
#         self.length_th = length_th
        
        
#     def filfinder_fiber_metrics(img, foreground_value = 255, th = 10):
#         roi, skeleton_img, coherency_img, orientation_metrics, vector_field, binary_mask, mask_yellow = get_orientation_metrics(img)
#         skeleton_img = filter_skeleton_img(skeleton_img, th = th)
#         prop_area_red = np.count_nonzero(binary_mask)/(512*512)
#         prop_area_yellow = np.count_nonzero(mask_yellow)/(512*512)

#         if prop_area_red >= 0.01:
#             img = invert(rgb2gray(roi))
#             fil = FilFinder2D(img, distance=250 * u.pc, mask=skeleton_img)
#             fil.preprocess_image(flatten_percent=85)
#             fil.create_mask(border_masking=True, verbose=False,
#             use_existing_mask=True)
#             fil.medskel(verbose=False)
#             fil.analyze_skeletons(branch_thresh=20* u.pix, skel_thresh=20 * u.pix, prune_criteria= 'length', )
        
#             filament_coords = []
#             num_branches = []
#             branch_length = []
#             for filament in fil.filaments:
#                 filament_coords.append(filament.pixel_coords)
#                 num_branches.append(filament.branch_properties['number'])
#                 branch_length.append(filament.branch_properties['length'])
        
#             fil.exec_rht()
#             lengths = fil.lengths().value
#             curvatures = fil.curvature.value
#             orientations = fil.orientation.value
#             num_intersections = [len(i) for i in fil.intersec_pts]
        
#             orientation_mean = circmean(orientations,high=np.pi/2, low=-np.pi/2)
#             orientation_var = circvar(orientations, high=np.pi/2, low=-np.pi/2)
        
#             curvature_weighted_mean = np.average(curvatures, weights=lengths/np.sum(lengths))
#             curvature_std = np.std(curvatures)
        
#             curvature_mean = np.mean(curvatures)
        
#             frac_dim, lacunarity_per_scale, sizes, counts = calc_frac_dim_and_lacunarity(skeleton_img, foreground_value=foreground_value)
#             slope, auc, max_lac, std_lac, mean_lac = summarize_lacunarity(lacunarity_per_scale, sizes)
        
#             branches_average = np.average(num_branches, weights = lengths/np.sum(lengths))
#             branching_density = np.sum(num_branches)/np.sum(lengths)
        
#             prop_area_red = np.count_nonzero(binary_mask)/(512*512)
#             prop_area_yellow = np.count_nonzero(mask_yellow)/(512*512)
        
#         summary_metrics = {'fiber_length_mean' : np.mean(lengths), 'fiber_length_std' : np.std(lengths), 
#                             'orientation_mean' : orientation_mean, 'orientation_var' : orientation_var, 
#                             'curvature_weighted_mean' : curvature_weighted_mean, 'curvature_std' : curvature_std,
#                             'curvature_mean' : curvature_mean, 'fractal_dim' : frac_dim, 'lacunarity_slope' : slope, 
#                             'lacunarity_mean' : mean_lac, 'lacunarity_auc': auc, 'lacunarity_max' : max_lac, 'lacunarity_std' : std_lac,
#                             'average_branching' : branches_average, 'branching_density' : branching_density, 
#                             'mean_coherency' : orientation_metrics['mean coherency - anisotropy'], 
#                             'dominant angle' : orientation_metrics['dominant angle'], 
#                             'orientation_var_coherence' : orientation_metrics['Orient. Variance'], 
#                             'prop_area_red' : prop_area_red, 'prop_area_yellow': prop_area_yellow, 
#                             'num_fibers' : len(lengths)}
        
#     return filament_coords, lengths, num_branches, branch_length, summary_metrics


# def calc_frac_dim_and_lacunarity(pruned_image, foreground_value=255):
#     """
#     Calculate both fractal dimension (box-counting) and lacunarity for a binary image.
#     Returns: frac_dim (float), lacunarity_per_scale (list of float), sizes (list of int)
#     """
#     height, width = pruned_image.shape[:2]
#     p = min(pruned_image.shape)
#     n = 2 ** np.floor(np.log(p) / np.log(2)) - 2
#     n = int(np.log(n) / np.log(2))
#     sizes = 2 ** np.arange(n, 1, -1)

#     counts = []
#     lacunarity_per_scale = []
#     Sum = []
#     for size in sizes:
#         # Sum the foreground pixels in boxes of a given size
#         S = np.add.reduceat(
#             np.add.reduceat(pruned_image == foreground_value,
#                             np.arange(0, height, size), axis=0),
#             np.arange(0, width, size), axis=1)

#         Sum.append(S)

#         # Count boxes that contain foreground pixels but aren't completely filled
#         counts.append(np.count_nonzero(S))
#         # Calculate lacunarity for this scale
#         masses = S.flatten()
#         masses = masses[masses > 0]  # Only non-empty boxes
#         # Lacunarity formula: (variance / mean^2) + 1
#         mean_mass = np.mean(masses)
#         var_mass = np.var(masses)
#         if mean_mass > 0:
#             lacunarity = var_mass / (mean_mass ** 2) + 1
#         else:
#             lacunarity = np.nan
#         lacunarity_per_scale.append(lacunarity)
    
#     # Handle all counts zero
#     if np.all(np.array(counts) == 0):
#         frac_dim = None
#     else:
#         counts = np.maximum(counts, 1e-10)
#         frac_dim = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]
    
#     return frac_dim, lacunarity_per_scale, sizes, counts


# def summarize_lacunarity(lacunarity_per_scale, sizes):
    
#     log_sizes = np.log(sizes)
#     log_lac = np.log(lacunarity_per_scale)
#     coeffs = np.polyfit(log_sizes, log_lac, 1)
#     slope = coeffs[0]  # Power-law exponent (negative for typical behavior)
#     intercept = coeffs[1]  # Prefactor or offset

#     auc = np.trapezoid(lacunarity_per_scale, x=sizes)     # Area under curve
#     max_lac = np.max(lacunarity_per_scale)                # Maximum
#     std_lac = np.std(lacunarity_per_scale)                # Variation
#     mean_lac = np.nanmean(lacunarity_per_scale)

#     return slope, auc, max_lac, std_lac, mean_lac