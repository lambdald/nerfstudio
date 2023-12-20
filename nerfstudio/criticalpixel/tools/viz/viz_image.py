'''
Description: file content
Usage:
python -m html4vision.server {port}
'''
from html4vision import Col, imagetable
# python -m html4vision.server 23333
# cols = [
#     Col('img', 'Image', 'img/*/*.jpg'),
#     Col('img', 'depth', 'depth_colormaps/*/*.png'),
#     # Col('img', 'depth', 'overlap_masks/*.png'),
# ]
cols = [
    Col('img', 'Image', 'img/*.png'),
    # Col('img', 'Image', 'undistorted_data/img/*/*.jpg'),
    # Col('img', 'Semantic', 'undistorted_data/semantics/*/*.png'),
    # Col('img', 'sky mask', 'undistorted_data/sky_masks/*/*.png'),
    # # Col('img', 'depth', 'overlap_masks/*.png'),
    # Col('img', 'semantic_binary_masks', 'undistorted_data/semantic_binary_masks/*/*.png'),
    # Col('img', 'mask', 'undistorted_data/masks/*/*.png'),
    # Col('img', 'color_masks', 'undistorted_data/color_masks/*/*.jpg'),
    # Col('img', 'depth map', 'undistorted_data/depth_colormaps/*/*.jpg'),
]
# cols = [
#     Col('img', 'Src Image', 'undistorted_images/*/*/*.JPG'),
#     Col('img', 'Dst Image', 'undistorted_images_mapping/*/*/*.JPG'),
#     # Col('img', 'depth', 'overlap_masks/*.png'),
# ]
# cols = [
#     Col('img', 'Src Image', 'undistorted_images/*.JPG'),
#     Col('img', 'Dst Image', 'undistorted_images_mapping/*.JPG'),
#     # Col('img', 'depth', 'overlap_masks/*.png'),
# ]
imagetable(
    cols,               
    'image.html',
    imsize=0,      # resize sketch svg to match corresponding image size (column 0)
    imscale=0.2,   # scale all images to 50%
    # adding image border and box shadows to the entire table
    style='img {border: 1px solid black;-webkit-box-shadow: 2px 2px 1px #ccc; box-shadow: 2px 2px 1px #ccc;}',
)