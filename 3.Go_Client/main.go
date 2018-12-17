package main

import (
	"context"
	"flag"
	// "fmt"
	"io/ioutil"
	"log"
	// "os"
	// "path/filepath"
	tf_core_framework "tensorflow/core/framework"
	pb "tensorflow_serving/apis"

	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"

	"google.golang.org/grpc"
)

func main() {

	servingAddress := flag.String("serving_address", "localhost:8500", "The tensorflow serving address")
	modelVersion := flag.Int64("model_version", 1, "The tensorflow serving model version")
	modelName := flag.String("model_name", "", "The tensorflow serving model name")
	imgPath := flag.String("img_path", "", "The absolute path to the image in the file system")

	flag.Parse()

	if len(*modelName) == 0 {
		panic("The target model name can not be empty.")
	}

	imageBytes, err := ioutil.ReadFile(*imgPath)
	if err != nil {
		log.Fatalln(err)
	}

	request := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name:          *modelName,
			SignatureName: "predict_images",
			Version: &google_protobuf.Int64Value{
				Value: *modelVersion,
			},
		},
		Inputs: map[string]*tf_core_framework.TensorProto{
			"images": &tf_core_framework.TensorProto{
				Dtype: tf_core_framework.DataType_DT_STRING,
				TensorShape: &tf_core_framework.TensorShapeProto{
					Dim: []*tf_core_framework.TensorShapeProto_Dim{
						&tf_core_framework.TensorShapeProto_Dim{
							Size: int64(1),
						},
					},
				},
				StringVal: [][]byte{imageBytes},
			},
		},
	}

	conn, err := grpc.Dial(*servingAddress, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Cannot connect to the grpc server: %v\n", err)
	}
	defer conn.Close()

	client := pb.NewPredictionServiceClient(conn)

	resp, err := client.Predict(context.Background(), request)
	if err != nil {
		log.Fatalln(err)
	}

	log.Println(resp)
}
