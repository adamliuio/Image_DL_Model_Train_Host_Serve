
# Deep Learning Image Classifier Creator

This repo can be used to:

## To create a image classifier based on your own image dataset, you only need to have the dataset ready in order to use this tool.

`I created this repo for fun & my own use, feel free to let me know any problems you run into.`

## So far, in order to use this tool, a user needs to specify 3 things:
	1. An image directory, which contains subfolders that category image data to different categories
	2. The pre-trained model name
	3. An output Directory

## With those 3 things, the user should be able to have a model trained specifically for the pointed image data.

---

# Only the Tensorflow Slim version is available at the moment.


# User Manual on a high-level:


## Step 0: Convert Data
```
cd 0.Convert_Data/
DATASET_DIR=/path/to/the/dataset/directory/

python convert_existing_img_folder.py \
	--dataset_dir=${DATASET_DIR} \
	--dataset_name=flowers
```

## Step 1.1: Train Model
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

## Step 2.1: Convert Model

```
cd 2.Convert_Ckpt_TFServe/

CHECKPOINT_DIR=/path/to/the/checkpoint/directory/
EXPORT_DIR=/where/to/export/the/converted/servable(.pb)/to/

python slim_inception_v4_saved_model.py \
	--checkpoint_dir=${CHECKPOINT_DIR} \
	--output_dir=${EXPORT_DIR} \
	--num_classes=5
```

## Step 2.2: Serve Model
### the servables should be saved as:
models/model_name
&emsp;|- 1
&emsp;&emsp;|- saved_model.pb
&emsp;&emsp;|- variables
&emsp;|- 2
&emsp;&emsp;|- saved_model.pb
&emsp;&emsp;|- variables
&emsp;|- ...

```
docker run -p 8500:8500 \
--mount type=bind,source=$(pwd)/models/inception_v4,target=/models/inception_v4 \
-e MODEL_NAME=inception_v4 -t tensorflow/serving &
```



## Step 3: Use Model

```
cd 3.Go_Client/

./Go_Client --serving-address localhost:8500 /Users/AdamLiu/Downloads/Images/dandelion.jpg
```
































