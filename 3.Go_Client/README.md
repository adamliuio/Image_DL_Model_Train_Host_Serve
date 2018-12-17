

## How To Use:

**Preparing the necessary stuff.**
The code in the block below will create 3 folders in the current folder:
> 1. serving
> 2. tensorflow
> 3. vendor

```
git clone -b r1.7 https://github.com/tensorflow/serving.git
git clone -b r1.7 https://github.com/tensorflow/tensorflow.git

mkdir -p vendor
PROTOC_OPTS='-I tensorflow -I serving --go_out=plugins=grpc:vendor'

eval "protoc $PROTOC_OPTS serving/tensorflow_serving/apis/*.proto"
eval "protoc $PROTOC_OPTS serving/tensorflow_serving/config/*.proto"
eval "protoc $PROTOC_OPTS serving/tensorflow_serving/util/*.proto"
eval "protoc $PROTOC_OPTS serving/tensorflow_serving/sources/storage_path/*.proto"
eval "protoc $PROTOC_OPTS tensorflow/tensorflow/core/framework/*.proto"
eval "protoc $PROTOC_OPTS tensorflow/tensorflow/core/example/*.proto"
eval "protoc $PROTOC_OPTS tensorflow/tensorflow/core/lib/core/*.proto"
eval "protoc $PROTOC_OPTS tensorflow/tensorflow/core/protobuf/{saver,meta_graph}.proto"
```

**Then generate the executable file.**

```
go build
```

**Use the model hosted by Tensorflow Serving through GRPC.**

```

./Go_Client \
	--serving_address 47.110.179.36:9090 \
	--model_version 1539762720 \
	--img_path /path/to/image

```


