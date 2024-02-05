cat $0
USER_NAME=lidong
USER_DIR=/data/$USER_NAME
# your python env
CONDA_ENV_DIR=$USER_DIR/miniconda3/envs
CONDA_ENV_NAME=nerfstudio


PYTHON_PATH=$CONDA_ENV_DIR/$CONDA_ENV_NAME/bin/python
dateTime="`date +%Y-%m-%d_%H%M%S`"




# scene data
SCENE_NAME=insta360/shuyuan/video20240108
DATA_NAME=VID_20240108_140609_00_001_pano_sift
DATA_DIR=$USER_DIR/data/$SCENE_NAME/$DATA_NAME



SCENE_NAME=nerfuser/A
DATA_NAME=undistorted_colmap
DATA_DIR=$USER_DIR/data/$SCENE_NAME/$DATA_NAME



# ouput dir
TRAIN_OUTPUT_DIR=outputs/$SCENE_NAME
EXPORT_OUTPUT_DIR=exports/$SCENE_NAME
RENDER_OUTPUT_DIR=renders/$SCENE_NAME

# exp
MODEL_NAME=meta-colmap-nerfacto
EXP_NAME=test

HOST_IP='192.168.0.12'

echo data_dir=$DATA_DIR

GPU_IDS=3


CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
$PYTHON_PATH nerfstudio/criticalpixel/scene/common_colmap/pipeline.py \
colmap-gsplat --data $DATA_DIR \
--viewer.websocket_port 8410 --viewer.websocket_host $HOST_IP \
meta-colmap --sparse_reldir sparse
