

# Converting Tensorflow models from checkpoints to TF Servables for using in TF Serving.


**For now, only the models based on Inception V4 network could be converted. We will cover more networks soon.**


# Hosting the TF Servable with docker.

```
SOURCE_PATH=$(pwd)/models/inception_v3
TARGET_PATH=/models/inception_v3
docker run -p 8500:8500 --mount type=bind,source=${SOURCE_PATH},target=${TARGET_PATH} -e MODEL_NAME=inception_v3 -t tensorflow/serving &
```

