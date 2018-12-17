
# This is the Deep Learning Image Models training module. Shenzhi DeepKnowledge, Inc.
## This document will be updated.

## So far, when a user specifies 3 things:
    1. a image directory, which contains subfolders that category image data to different categories
    2. Pre-trained model name
    3. Output Directory

## the user should be able to have a model trained specifically for the pointed image data.

---

# Only the Tensorflow Slim version is available at the moment.
# Due to the fact that Tensorflow is Officially supporting Keras now and would very likely prefer tf.Keras over TF.Slim, so we will switch this Bottleneck Model Building library to Keras soon.


---


Step 1.1: Train Model
```
cd 1.Train_a_Model/

CHECKPOINT_DIR=/path/to/the/checkpoint/directory/
BOTTLENECK_PATH=/path/to/the/bottleneck/directory/
MODEL_NAME=the_model_name
SAVE_SUMMARIES_SECS=10

python train_image_classifier.py \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${BOTTLENECK_PATH} \
    --save_summaries_secs=${SAVE_SUMMARIES_SECS} \
    --save_interval_secs=300 \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits
```

## Step 1.2: Evaluate Model

```
CHECKPOINT_FILE=/path/to/the/saved/checkpoint/file/

python eval_image_classifier.py \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --model_name=${MODEL_NAME}
```




Use & modify the command block below to easily train on your dataset.

```
DATASET_DIR=/path/to/the/folder/where/TFRecords/are/located
CHECKPOINT_DIR=/path/to/the/folder/where/you/want/your/training/checkpoints/graphs/to/be/saved/to
BOTTLENECK_PATH=/path/to/the/folder/where/the/pre-trained/.ckpt/file/is/located
MODEL_NAME=inception_v3   # or any other model name from the "Pre_trained_nets_list" below
SAVE_SUMMARIES_SECS=10    # for real time (every 10 seconds) graph displaying

python train_image_classifier.py \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${BOTTLENECK_PATH} \
    --save_summaries_secs=${SAVE_SUMMARIES_SECS} \
    --save_interval_secs=300 \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits


Pre_trained_nets_list = [
    'alexnet_v2', 'cifarnet', 'overfeat',
    'vgg_a', 'vgg_16', 'vgg_19',
    'inception_v1', 'inception_v2', 'inception_v3', 'inception_v4',
    'inception_resnet_v2', 'lenet',
    'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v1_200',
    'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200',
    'mobilenet_v1', 'mobilenet_v1_075', 'mobilenet_v1_050', 'mobilenet_v1_025',
    'mobilenet_v2', 'mobilenet_v2_140', 'mobilenet_v2_035',
    'nasnet_cifar', 'nasnet_mobile', 'nasnet_large',
    'pnasnet_large', 'pnasnet_mobile',
]
```




