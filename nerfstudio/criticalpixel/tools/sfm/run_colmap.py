from nerfstudio.criticalpixel.utils.subprocess_controller import run_cmd_with_log
import os
from pathlib import Path
from dataclasses import dataclass
import argparse
import os
from pathlib import Path
import shutil
from struct import unpack
import numpy as np
import sys
from rich import print
import cv2
from dataclasses import dataclass, field
import tyro


@dataclass
class ImageReaderConfig:
    # Optional root path to folder which contains image masks. For a given image, the corresponding mask must have the same sub-path below this root as the image has below image_path. The filename must be equal, aside from the added extension .png. For example, for an image image_path/abc/012.jpg, the mask would be mask_path/abc/012.jpg.png. No features will be extracted in regions where the mask image is black (pixel intensity value 0 in grayscale).
    camera_model = "OPENCV"  # (default: SIMPLE_RADIAL)
    # Possible values: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, FOV, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, THIN_PRISM_FISHEYE
    # Name of the camera model. See: Camera Models

    single_camera = 0  # (default: 0)
    # Whether to use the same camera for all images.

    single_camera_per_folder = 1  # (default: 0)
    # Whether to use the same camera for all images in the same sub-folder.

    single_camera_per_image = 0  # (default: 0)
    # Whether to use a different camera for each image.

    existing_camera_id = -1  # (default: -1)
    # Whether to explicitly use an existing camera for all images. Note that in this case the specified camera model and parameters are ignored.

    camera_params = ""
    # Manual specification of camera parameters. If empty, camera parameters will be extracted from EXIF, i.e. principal point and focal length.

    default_focal_length_factor = 1.2  # (default: 1.2)
    # If camera parameters are not specified manually and the image does not have focal length EXIF information, the focal length is set to the value default_focal_length_factor * max(width, height).

    camera_mask_path = ""
    # Optional path to an image file specifying a mask for all images. No features will be extracted in regions where the mask is black (pixel intensity value 0 in grayscale).


@dataclass
class SiftExtractionConfig:
    num_threads = -1  # (default: -1)
    # Number of threads for feature extraction.

    use_gpu = 1  # (default: 1)
    # Whether to use the GPU for feature extraction.

    gpu_index = -1  # (default: -1)
    # Index of the GPU used for feature extraction. For multi-GPU extraction, you should separate multiple GPU indices by comma, e.g. "0,1,2,3". See: Multi-GPU support in feature extraction/matching

    max_image_size = 3200  # (default: 3200)
    # Maximum image size, otherwise image will be down-scaled.

    max_num_features = 10240  # (default: 8192)
    # Maximum number of features to detect, keeping larger-scale features.

    first_octave = -1  # (default: -1)
    # First octave in the pyramid, i.e. -1 upsamples the image by one level. By convention, the octave of index 0 starts with the image full resolution. Specifying an index greater than 0 starts the scale space at a lower resolution (e.g. 1 halves the resolution). Similarly, specifying a negative index starts the scale space at an higher resolution image, and can be useful to extract very small features (since this is obtained by interpolating the input image, it does not make much sense to go past -1).

    num_octaves = 4  # (default: 4)
    # Number of octaves. Increasing the scale by an octave means doubling the size of the smoothing kernel, whose effect is roughly equivalent to halving the image resolution. By default, the scale space spans as many octaves as possible (i.e. roughly log2(min(width, height))), which has the effect of searching keypoints of all possible sizes.

    octave_resolution = 3  # (default: 3)
    # Number of levels per octave. Each octave is sampled at this given number of intermediate scales. Increasing this number might in principle return more refined keypoints, but in practice can make their selection unstable due to noise.

    peak_threshold = 0.0067  # (default: 0.0067)
    # Peak threshold for detection. This is the minimum amount of contrast to accept a keypoint. Increase to eliminate more keypoints.

    edge_threshold = 10  # (default: 10)
    # Edge threshold for detection. Decrease to eliminate more keypoints.

    estimate_affine_shape: int = 1  # (default: 0)
    # Estimate affine shape of SIFT features in the form of oriented ellipses as opposed to original SIFT which estimates oriented disks.

    max_num_orientations = 2  # (default: 2)
    # aximum number of orientations per keypoint if not SiftExtraction.estimate_affine_shape.

    upright = 0  # (default: 0)
    # Fix the orientation to 0 for upright features.

    domain_size_pooling: int = 1  # (default: 0)
    # Enable the more discriminative DSP-SIFT features instead of plain SIFT. Domain-size pooling computes an average SIFT descriptor across multiple scales around the detected scale. DSP-SIFT outperforms standard SIFT in most cases.
    # This was proposed in Domain-Size Pooling in Local Descriptors: DSP-SIFT, J. Dong and S. Soatto, CVPR 2015. This has been shown to outperform other SIFT variants and learned descriptors in Comparative Evaluation of Hand-Crafted and Learned Local Features, Schönberger, Hardmeier, Sattler, Pollefeys, CVPR 2016.

    # dsp_min_scale (default: 0.1667)
    # dsp_max_scale (default: 3)
    # dsp_num_scales (default: 10)
    # Domain-size pooling parameters. See: SiftExtraction.domain_size_pooling


# @dataclass
# class SiftExtractionConfig:
#     image_path = ""
#     # Root path to folder which contains the images.

#     image_list_path = ""
#     # Optional list of images to read. The list must contain the relative path of the images with respect to the image_path.

#     descriptor_normalization = "l1_root"  # (default: l1_root)
#     # Possible values: l1_root, l2
#     # Whether to use L1 normalization of each descriptor followed by element-wise square rooting (RootSIFT) or standard L2 normalization.
#     # RootSIFT descriptors are usually better than standard SIFT. Proposed in Three things everyone should know to improve object retrieval, R. Arandjelovic and A. Zisserman, CVPR 2012.

#     ImageReader = ImageReaderConfig()
#     SiftExtraction = SiftExtraction()


@dataclass
class SiftMatching:
    num_threads = -1  # (default: -1)
    # Number of threads for feature matching and geometric verification.

    use_gpu = 1  # (default: 1)
    # Whether to use the GPU for feature matching.

    gpu_index = -1  # (default: -1)
    # Index of the GPU used for feature matching. For multi-GPU matching, you should separate multiple GPU indices by comma, e.g. "0,1,2,3". See: Multi-GPU support in feature extraction/matching

    max_ratio = 0.8  # (default: 0.8)
    # Maximum distance ratio between first and second best match.

    max_distance = 0.7  # (default: 0.7)
    # Maximum distance to best match.

    cross_check = 1  # (default: 1)
    # Whether to enable cross checking in matching.

    max_error = 4  # (default: 4)
    # Maximum epipolar error in pixels for geometric verification.

    max_num_matches = 32768  # (default: 32768)
    # Maximum number of matches.

    confidence = 0.999  # (default: 0.999)
    # Confidence threshold for geometric verification.

    max_num_trials = 10000  # (default: 10000)
    # Maximum number of RANSAC iterations. Note that this option overrules the SiftMatching.min_inlier_ratio option.

    min_inlier_ratio = 0.25  # (default: 0.25)
    # A priori assumed minimum inlier ratio, which determines the maximum number of iterations.

    min_num_inliers = 15  # (default: 15)
    # Minimum number of inliers for an image pair to be considered as geometrically verified.

    multiple_models = 0  # (default: 0)
    # Whether to attempt to estimate multiple geometric models per image pair.

    guided_matching = 1  # (default: 0)
    # Whether to perform guided matching, if geometric verification succeeds.

    planar_scene = 0  # (default: 0)
    # Force Homography use for Two-view Geometry (can help for planar scenes).

    compute_relative_pose = 0  # (default: 0)
    # Whether to estimate the relative pose between the two images and save them to the database.


@dataclass
class ExhaustiveMatching:
    block_size = 50  # (default: 50)
    # Block size, i.e. number of images to simultaneously load into memory.


@dataclass
class FeatureMatchingConfig:
    # match mode in ["exhaustive","sequential","spatial","transitive","vocab_tree"]
    # ref: https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification

    SiftMatching = SiftMatching()
    ExhaustiveMatching = ExhaustiveMatching()


@dataclass
class MapperConfig:
    #! General
    min_num_matches = 15  # (default: 15)
    # The minimum number of matches for inlier matches to be considered.

    ignore_watermarks = 0  # (default: 0)
    # Whether to ignore the inlier matches of watermark image pairs.

    multiple_models = 0  # (default: 1)
    # Whether to reconstruct multiple sub-models.

    max_num_models = 50  # (default: 50)
    # The number of sub-models to reconstruct.

    max_model_overlap = 20  # (default: 20)
    # The maximum number of overlapping images between sub-models. If the current sub-models shares more than this number of images with another model, then the reconstruction is stopped.

    min_model_size = 10  # (default: 10)
    # The minimum number of registered images of a sub-model, otherwise the sub-model is discarded.

    extract_colors = 1  # (default: 1)
    # Whether to extract colors for reconstructed points.

    num_threads = -1  # (default: -1)
    # The number of threads to use during reconstruction.

    snapshot_path = ""
    snapshot_images_freq = 0  # (default: 0)
    # Path to a folder with reconstruction snapshots during incremental reconstruction. Snapshots will be saved according to the specified frequency of registered images.

    fix_existing_images = 0  # (default: 0)
    # If reconstruction is provided as input, fix the existing image poses.

    #! Init

    init_image_id1 = -1  # (default: -1)
    init_image_id2 = -1  # (default: -1)
    # The image identifiers used to initialize the reconstruction. Note that only one or both image identifiers can be specified. In the former case, the second image is automatically determined.

    init_num_trials = 200  # (default: 200)
    # The number of trials to initialize the reconstruction.

    init_min_num_inliers = 100  # (default: 100)
    # Minimum number of inliers for initial image pair.

    init_max_error = 4  # (default: 4)
    # Maximum error in pixels for two-view geometry estimation for initial image pair.

    init_max_forward_motion = 0.95  # (default: 0.95)
    # Maximum forward motion for initial image pair.

    init_min_tri_angle = 16  # (default: 16)
    # Minimum triangulation angle for initial image pair.

    init_max_reg_trials = 2  # (default: 2)
    # Maximum number of trials to use an image for initialization.
    #! Registration
    abs_pose_max_error = 12  # (default: 12)
    # Maximum reprojection error in absolute pose estimation.

    abs_pose_min_num_inliers = 30  #  (default: 30)
    # Minimum number of inliers in absolute pose estimation.

    abs_pose_min_inlier_ratio = 0.25  #  (default: 0.25)
    # Minimum inlier ratio in absolute pose estimation.

    max_reg_trials = 3  # (default: 3)
    # Maximum number of trials to register an image.
    #! Triangulation

    tri_max_transitivity = 1  # (default: 1)
    # Maximum transitivity to search for correspondences.

    tri_create_max_angle_error = 2  # (default: 2)
    # Maximum angular error to create new triangulations.

    tri_continue_max_angle_error = 2  # (default: 2)
    # Maximum angular error to continue existing triangulations.

    tri_merge_max_reproj_error = 4  # (default: 4)
    # Maximum reprojection error in pixels to merge triangulations.

    tri_complete_max_reproj_error = 4  # (default: 4)
    # Maximum reprojection error to complete an existing triangulation.

    tri_complete_max_transitivity = 5  # (default: 5)
    # Maximum transitivity for track completion.

    tri_re_max_angle_error = 5  # (default: 5)
    # Maximum angular error to re-triangulate under-reconstructed image pairs.

    tri_re_min_ratio = 0.2  # (default: 0.2)
    # Minimum ratio of common triangulations between an image pair over the number of correspondences between that image pair to be considered as under-reconstructed.

    tri_re_max_trials = 1  # (default: 1)
    # Maximum number of trials to re-triangulate an image pair.

    tri_min_angle = 1.5  # (default: 1.5)
    # Minimum pairwise triangulation angle for a stable triangulation. If your images are taken from far distance with respect to the scene, you can try to reduce the minimum triangulation angle

    tri_ignore_two_view_tracks = 1  # (default: 1)
    # Whether to ignore two-view feature tracks in triangulation, resulting in fewer 3D points than possible. Triangulation of two-view tracks can in rare cases improve the stability of sparse image collections by providing additional constraints in bundle adjustment.

    #! BA

    ba_refine_focal_length = 1  # (default: 1)
    ba_refine_principal_point = 0  # (default: 0)
    ba_refine_extra_params = 1  # (default: 1)
    # Which intrinsic parameters to optimize during the reconstruction.

    ba_min_num_residuals_for_multi_threading = 50000  # (default: 50000)
    # The minimum number of residuals per bundle adjustment problem to enable multi-threading solving of the problems.

    ba_local_num_images = 6  # (default: 6)
    # The number of images to optimize in local bundle adjustment.

    ba_local_function_tolerance = 0  # (default: 0)
    # Ceres solver function tolerance for local bundle adjustment

    ba_local_max_num_iterations = 25  # (default: 25)
    # The maximum number of local bundle adjustment iterations.

    ba_global_use_pba = 0  # (default: 0)
    # Whether to use PBA (Parralel Bundle Adjustment) in global bundle adjustment. See: https://grail.cs.washington.edu/projects/mcba/, https://github.com/cbalint13/pba

    ba_global_pba_gpu_index = -1  # (default: -1)
    # The GPU index for PBA bundle adjustment.

    ba_global_images_ratio = 1.1  # (default: 1.1)
    ba_global_points_ratio = 1.1  # (default: 1.1)
    ba_global_images_freq = 500  # (default: 500)
    ba_global_points_freq = 250000  # (default: 250000)
    # The growth rates after which to perform global bundle adjustment.

    ba_global_function_tolerance = 0  # (default: 0)
    # Ceres solver function tolerance for global bundle adjustment

    ba_global_max_num_iterations = 50  # (default: 50)
    # The maximum number of global bundle adjustment iterations.

    ba_global_max_refinements = 5  # (default: 5)
    ba_global_max_refinement_change = 0.005  # (default: 0.0005)
    ba_local_max_refinements = 2  # (default: 2)
    ba_local_max_refinement_change = 0.001  # (default: 0.001)
    # The thresholds for iterative bundle adjustment refinements.

    local_ba_min_tri_angle = 6  # (default: 6)
    # Minimum triangulation for images to be chosen in local bundle adjustment.

    ba_use_cuda = True

    #! Filter

    min_focal_length_ratio = 0.1  # (default: 0.1)
    max_focal_length_ratio = 0.1  # (default: 10)
    max_extra_param = 0.1  # (default: 1)
    # Thresholds for filtering images with degenerate intrinsics.

    filter_max_reproj_error = 4  # (default: 4)
    # Maximum reprojection error in pixels for observations.

    filter_min_tri_angle = 1.5  # (default: 1.5)
    # Minimum triangulation angle in degrees for stable 3D points.


@dataclass
class SfmConfig:
    data_dir: Path

    use_gpu: bool = True
    gpu_id: int = 0
    log_reldir: Path = Path("logs/sfm")
    images_reldir = Path("images")
    colmap_model_dir = Path("colmap")
    pixsfm_model_dir = Path("pixsfm")
    final_model_dir = Path("sparse")
    database_relpath = Path("database.db")
    num_images = -1
    match_mode = "exhaustive"
    dsp_sift = False
    camera_params = ""

    colmap_bin: Path = Path("/data/lidong/devlibs/bin/colmap")

    image_reader: ImageReaderConfig = ImageReaderConfig()
    feature_extractor: SiftExtractionConfig = SiftExtractionConfig()
    matcher: FeatureMatchingConfig = FeatureMatchingConfig()
    mapper: MapperConfig = MapperConfig()


def run_feature_matching(sfm_config: SfmConfig, feature_config: FeatureMatchingConfig):
    """
    match mode in ["exhaustive","sequential","spatial","transitive","vocab_tree"]
    ref: https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification
    """
    SMALL_VOCAB_TREE_PATH = "/mnt/g/libs/vocab_tree_flickr100K_words256K.bin"
    LARGE_VOCAB_TREE_PATH = "/mnt/g/libs/vocab_tree_flickr100K_words256K.bin"
    if sfm_config.match_mode == "exhaustive":
        # For hundreds of images
        cmd = f"""
        colmap exhaustive_matcher \
        --SiftMatching.use_gpu 1 \
        --database_path {sfm_config.database_path} \
        --SiftMatching.guided_matching {feature_config.SiftMatching.guided_matching} \
        --ExhaustiveMatching.block_size {feature_config.ExhaustiveMatching.block_size} \
        --SiftMatching.max_num_matches {feature_config.SiftMatching.max_num_matches} \
        --SiftMatching.min_inlier_ratio {feature_config.SiftMatching.min_inlier_ratio} \
        --SiftMatching.max_ratio {feature_config.SiftMatching.max_ratio} \
        """
    elif mode == "sequential":
        # For images from video.
        cmd = f"""
        colmap sequential_matcher \
        --SiftMatching.guided_matching=true \
        --SiftMatching.use_gpu 1 \
        --database_path {database_path}
        --SequentialMatching.overlap 50
        """
    elif mode == "vocab_tree":
        # For 1000 - 10,000 images: SMALL_VOCAB_TREE_PATH
        # For 10,000 - 100,000 images

        cmd = f"""
            colmap vocab_tree_matcher \
            --SiftMatching.use_gpu 1 \
            --database_path {database_path} \
            --VocabTreeMatching.vocab_tree_path {SMALL_VOCAB_TREE_PATH}
        """
    else:
        raise NotImplementedError(f"unknown feature match mode: {mode}")
    print(cmd)
    os.system(cmd)


def run_mapper(sfm_config: SfmConfig, mapper_config: MapperConfig):
    cmd = f"""
    colmap mapper \
    --database_path {sfm_config.database_path} \
    --image_path {sfm_config.image_dir} \
    --output_path {sfm_config.colmap_model_dir} \
    --Mapper.multiple_models {mapper_config.multiple_models} \
    --Mapper.init_min_tri_angle {mapper_config.init_min_tri_angle} \
    --Mapper.tri_re_max_trials {mapper_config.tri_re_max_trials} \
    --Mapper.init_max_reg_trials {mapper_config.init_max_reg_trials} \
    --Mapper.abs_pose_min_inlier_ratio {mapper_config.abs_pose_min_inlier_ratio} \
    --Mapper.tri_min_angle {mapper_config.tri_min_angle} \
    --Mapper.max_reg_trials {mapper_config.max_reg_trials} \
    --Mapper.local_ba_min_tri_angle {mapper_config.local_ba_min_tri_angle} \
    --Mapper.abs_pose_min_num_inliers {mapper_config.abs_pose_min_num_inliers} \
    --Mapper.filter_min_tri_angle {mapper_config.filter_min_tri_angle} \
    --Mapper.ba_local_num_images {mapper_config.ba_local_num_images} \
    --Mapper.tri_ignore_two_view_tracks {mapper_config.tri_ignore_two_view_tracks} \
    --Mapper.abs_pose_max_error {mapper_config.abs_pose_max_error} \
    """
    print(cmd)
    os.system(cmd)


def run_point_triangulation(database_path, image_dir, input_dir, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    cmd = f"""
    colmap point_triangulator --database_path {database_path} --image_path {image_dir} \
    --input_path {input_dir} \
    --output_path {output_dir} \
    --Mapper.init_min_tri_angle 2
    """
    os.system(cmd)


def run_ba(input_dir, output_dir):
    cmd = f"""colmap bundle_adjuster \
        --input_path {input_dir}  \
        --output_path {output_dir} \
        --BundleAdjustment.refine_principal_point 1
        """
    os.system(cmd)


def run_patchmatch(workspace_dir, depth_max, max_image_size):
    cmd = f"""
        colmap patch_match_stereo \
        --workspace_path {workspace_dir} \
        --PatchMatchStereo.depth_min 0.0 \
        --PatchMatchStereo.depth_max {depth_max} \
        --PatchMatchStereo.max_image_size {max_image_size} \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.geom_consistency_max_cost 10 \
        --PatchMatchStereo.geom_consistency_regularizer 0.5 \
        --PatchMatchStereo.filter 1 \
        --PatchMatchStereo.filter_min_triangulation_angle 1 \
        --PatchMatchStereo.filter_min_num_consistent 2 \
        --PatchMatchStereo.filter_geom_consistency_max_cost 2
    """
    os.system(cmd)


def run_stereo_fusion(workspace_dir, output_pointcloud_file):
    cmd = f"""
        colmap stereo_fusion \
        --workspace_path {workspace_dir} \
        --output_path {output_pointcloud_file}
    """
    os.system(cmd)


def run_meshing(input_dir, output_path):
    cmd = f"""
        colmap delaunay_mesher \
        --input_path {input_dir} \
        --output_path {output_path}
        """
    os.system(cmd)


def run_pixsfm(sfm_config: SfmConfig):
    cmd = f"""
    python -m pixsfm.refine_colmap bundle_adjuster \
        --input_path {sfm_config.colmap_model_dir/'0'} \
        --output_path {sfm_config.pixsfm_model_dir} \
        --image_dir {sfm_config.image_dir}
    """
    print(cmd)
    os.system(cmd)


def run_pipeline(args):
    sfm_config = get_sfm_config(args)
    sfm_config.num_images = get_num_images(sfm_config.image_dir)
    create_dirs(sfm_config)
    print(repr(sfm_config))

    feature_extraction_config = SiftExtractionConfig()
    feature_extraction_config.ImageReader.single_camera = 1
    feature_extraction_config.SiftExtraction.max_image_size = 4096
    feature_extraction_config.SiftExtraction.max_num_features = 20480
    feature_extraction_config.SiftExtraction.num_octaves = 8
    feature_extraction_config.SiftExtraction.octave_resolution = 5

    feature_matching_config = FeatureMatchingConfig()
    feature_matching_config.SiftMatching.max_num_matches = 40960
    feature_matching_config.SiftMatching.min_inlier_ratio = 0.25
    feature_matching_config.SiftMatching.max_ratio = 0.8

    mapper_config = MapperConfig()
    mapper_config.init_min_tri_angle = 12
    mapper_config.init_max_reg_trials = 5

    mapper_config.abs_pose_min_inlier_ratio = 0.1
    mapper_config.abs_pose_min_num_inliers = 5
    mapper_config.abs_pose_max_error = 12
    mapper_config.max_reg_trials = 20

    mapper_config.tri_re_max_trials = 20
    mapper_config.tri_min_angle = 1.0
    mapper_config.local_ba_min_tri_angle = 2
    mapper_config.ba_local_num_images = 4
    mapper_config.filter_min_tri_angle = 1.0
    mapper_config.tri_ignore_two_view_tracks = 0

    if sfm_config.dsp_sift:
        print("in dsp_sift mode")
        feature_extraction_config.SiftExtraction.estimate_affine_shape = 1
        feature_extraction_config.SiftExtraction.domain_size_pooling = 1
        feature_matching_config.SiftMatching.guided_matching = 1
    if sfm_config.camera_params:
        print("camera params=", sfm_config.camera_params)
        feature_extraction_config.ImageReader.camera_params = sfm_config.camera_params

    run_feature_extraction(sfm_config, feature_extraction_config)
    run_feature_matching(sfm_config, feature_matching_config)
    run_mapper(sfm_config, mapper_config)
    run_pixsfm(sfm_config)
    print(f"copy {sfm_config.pixsfm_model_dir} to {sfm_config.sparse_model_dir}")
    if sfm_config.sparse_model_dir.exists():
        shutil.rmtree(sfm_config.sparse_model_dir)
    shutil.copytree(sfm_config.pixsfm_model_dir, sfm_config.sparse_model_dir)


class ColmapSfm:
    config: SfmConfig

    def __init__(self, config) -> None:
        self.config = config

    @property
    def workspace(self) -> Path:
        return self.config.data_dir

    def get_cuda_prefix(self):
        cuda_prefix = f"CUDA_VISIBLE_DEVICES={self.config.gpu_id}"
        return cuda_prefix

    def run_feature_extractor(self):
        use_gpu = 1 if self.config.use_gpu else 0

        image_dir: Path = self.workspace / self.config.images_reldir
        assert image_dir.exists(), image_dir
        log_dir = self.workspace / self.config.log_reldir
        log_dir.mkdir(parents=True, exist_ok=True)

        print("extract feature")
        cmd_feature = f"""{self.get_cuda_prefix()} \
        {self.config.colmap_bin} feature_extractor \
            --database_path {self.workspace}/database.db \
            --image_path {image_dir} \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model {self.config.image_reader.camera_model} \
            --SiftExtraction.use_gpu {use_gpu} \
            --SiftExtraction.estimate_affine_shape {self.config.feature_extractor.estimate_affine_shape} \
            --SiftExtraction.domain_size_pooling {self.config.feature_extractor.domain_size_pooling} \
            --SiftExtraction.max_num_features {self.config.feature_extractor.max_num_features}
        """

        if self.config.image_reader.camera_params:
            cmd_feature = cmd_feature.strip() + f" --ImageReader.camera_params {self.config.image_reader.camera_params}"
        run_cmd_with_log(cmd_feature, "feature_extractor", log_dir=log_dir, timeout=100000)

    def run_matcher(self):
        use_gpu = 1 if self.config.use_gpu else 0
        log_dir = self.workspace / self.config.log_reldir
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd_matcher = f""" {self.get_cuda_prefix()} \
        {self.config.colmap_bin} exhaustive_matcher \
            --database_path {self.workspace/self.config.database_relpath} \
            --SiftMatching.use_gpu {use_gpu} \
            --SiftMatching.guided_matching {self.config.matcher.SiftMatching.guided_matching} \
        """
        run_cmd_with_log(cmd_matcher, "exhaustive_matcher", log_dir=log_dir, timeout=100000)

    def execute(self):
        print("video sfm workspace:", self.workspace)
        use_gpu = 1 if self.config.use_gpu else 0

        image_dir: Path = self.workspace / self.config.images_reldir
        assert image_dir.exists(), image_dir
        log_dir = self.workspace / self.config.log_reldir
        log_dir.mkdir(parents=True, exist_ok=True)

        self.run_feature_extractor()

        self.run_matcher()

        sparse_dir = self.workspace / self.config.colmap_model_dir
        sparse_dir.mkdir(parents=True, exist_ok=True)
        cmd_mapper = f""" {self.get_cuda_prefix()} \
        {self.config.colmap_bin} mapper \
            --database_path {self.workspace}/{self.config.database_relpath} \
            --image_path {image_dir} \
            --output_path {sparse_dir} \
            --Mapper.ba_use_cuda {self.config.mapper.ba_use_cuda}
        """
        run_cmd_with_log(cmd_mapper, "mapper", log_dir=log_dir, timeout=1000000)

def main():
    config = tyro.cli(SfmConfig)

    sfm = ColmapSfm(config)
    sfm.execute()


if __name__ == "__main__":
    main()
