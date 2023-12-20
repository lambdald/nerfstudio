cat $0
USER_NAME=lidong
USER_DIR=/data/lidong
# your python env
CONDA_ENV_DIR=$USER_DIR/miniconda3/envs
CONDA_ENV_NAME=py310dev


PYTHON_PATH=$CONDA_ENV_DIR/$CONDA_ENV_NAME/bin/python
dateTime="`date +%Y-%m-%d_%H%M%S`"




# scene data
SCENE_NAME=lanrundasha
DATA_DIR=$USER_DIR/data/$SCENE_NAME
DATA_NAME=2023_12_13_10_51_IMG_0506

DATA_DIR=$DRONE_DATA_DIR/$DATA_NAME

# ouput dir
TRAIN_OUTPUT_DIR=outputs/$SCENE_NAME
EXPORT_OUTPUT_DIR=exports/$SCENE_NAME
RENDER_OUTPUT_DIR=renders/$SCENE_NAME

# exp
MODEL_NAME=meta-colmap-nerfacto
EXP_NAME=test


$PYTHON_PATH nerfstudio/criticalpixel/scene/common_colmap/pipeline.py \
--data-config.data-dir $DATA_DIR \
--timestamp $dateTime --exp_name $EXP_NAME  \
train-config:$MODEL_NAME meta-colmap --data-dir $DATA_DIR --train-config.viewer.websocket_port 8410 --train-config.viewer.websocket_host `hostname -I` \
--train-config.output_dir $TRAIN_OUTPUT_DIR --train-config.experiment-name $DATA_NAME

