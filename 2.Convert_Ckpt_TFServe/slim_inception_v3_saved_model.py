
#!/usr/bin/env python
"""
	Export inception model given existing training checkpoints.

	The model is exported as SavedModel with proper signatures that can be loaded by
	standard tensorflow_model_server.
"""

"""
	###############
	VERY IMPORTANT:
	###############
		Recording what GPU you used in the training process in order to have a successful Model Exportation.


	Usage:
	python slim_inception_v3_saved_model.py \
		--checkpoint_dir /path/to/model/checkpoint/directory \
		--output_dir /path/to/where/to/export/the/servable \
		--num_classes=5
"""

import os
import logging
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from nets import inception
from datasets import imagenet

slim = tf.contrib.slim
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S' )


def main(unused_argv=None):
	export()


def preprocess_image(image_buffer):

	"""Preprocess JPEG encoded bytes to 3D float Tensor."""

	# Decode the string as an RGB JPEG.
	# Note that the resulting image contains an unknown height and width
	# that is set dynamically by decode_jpeg. In other words, the height
	# and width of image is unknown at compile-time.
	image = tf.image.decode_jpeg(image_buffer, channels=3)

	# After this point, all image pixels reside in [0,1)
	# until the very end, when they're rescaled to (-1, 1).  The various
	# adjust_* ops all require this range for dtype float.
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)

	# Crop the central region of the image with an area containing 87.5% of
	# the original image.
	image = tf.image.central_crop(image, central_fraction=0.875)

	# Resize the image to the original height and width.
	image = tf.expand_dims(image, 0)
	image = tf.image.resize_bilinear(
		image, [FLAGS.image_size, FLAGS.image_size], align_corners=False
	)
	image = tf.squeeze(image, [0])

	# Finally, rescale to [-1,1] instead of [0, 1)
	image = tf.subtract(image, 0.5)
	image = tf.multiply(image, 2.0)

	return image


def export():

	with tf.Graph().as_default():

		# build inference model

		# imagenet labels
		names = imagenet.create_readable_names_for_imagenet_labels()

		names_tensor = tf.constant(list(names.values()))

		names_lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(names_tensor)

		# input transformation
		serialized_tf_example = tf.placeholder(tf.string, name="tf_example")
		feature_configs = {
			"image/encoded": tf.FixedLenFeature(shape=[], dtype=tf.string),
		}
		tf_example = tf.parse_example(serialized_tf_example, feature_configs)
		jpegs      = tf_example["image/encoded"]
		images     = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

		# run inference
		with slim.arg_scope(inception.inception_v3_arg_scope()):
			# inception v3 models
			logits, end_points = inception.inception_v3(images, num_classes=NUM_CLASSES, is_training=False)
			# logits = tf.Print(logits, [logits])

		probs = tf.nn.softmax(logits)

		# transform output to topk result
		topk_probs, topk_indices = tf.nn.top_k(probs, NUM_TOP_CLASSES)

		topk_names = names_lookup_table.lookup(tf.to_int64(topk_indices))

		init_fn = slim.assign_from_checkpoint_fn(
			tf.train.latest_checkpoint(FLAGS.checkpoint_dir),
			slim.get_model_variables(),
		)

		# sess config
		config = tf.ConfigProto(
			gpu_options={ "allow_growth": 1, },
			allow_soft_placement=True,
			log_device_placement=False,
		)

		with tf.Session(config=config) as sess:

			init_fn(sess)

			# # to print out all the tensornames in the attached layers to inception V3
			for node_tensor in tf.get_default_graph().as_graph_def().node:
				if str(node_tensor.name).startswith('InceptionV3/Logits'):
					print(str(node_tensor.name))

			prelogits = sess.graph.get_tensor_by_name("InceptionV3/Logits/SpatialSqueeze:0")
			
			# an optional alternative
			# prelogits = end_points['PreLogitsFlatten']

			# export inference model.
			output_path = os.path.join(
				tf.compat.as_bytes(FLAGS.output_dir),
				tf.compat.as_bytes(str(FLAGS.model_version))
			)
			print("Exporting trained model to", output_path)
			builder = tf.saved_model.builder.SavedModelBuilder(output_path)

			# build the signature_def_map.
			predict_inputs_tensor_info   = tf.saved_model.utils.build_tensor_info(jpegs)
			classes_output_tensor_info   = tf.saved_model.utils.build_tensor_info(topk_names)
			scores_output_tensor_info    = tf.saved_model.utils.build_tensor_info(topk_probs)
			prelogits_output_tensor_info = tf.saved_model.utils.build_tensor_info(prelogits)

			prediction_signature = (
				tf.saved_model.signature_def_utils.build_signature_def(
					inputs={
						"images":      predict_inputs_tensor_info
					},
					outputs={
						"classes":     classes_output_tensor_info,
						"scores":       scores_output_tensor_info,
						"prelogits": prelogits_output_tensor_info,
					},
					method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
				)
			)

			legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")

			builder.add_meta_graph_and_variables(
				sess, [tf.saved_model.tag_constants.SERVING],
				signature_def_map={ "predict_images": prediction_signature, },
				legacy_init_op=legacy_init_op
			)

			builder.save()

			print("Successfully exported model to %s" % FLAGS.output_dir)


def define_flags():

	tf.app.flags.DEFINE_string(
		"checkpoint_dir", "",
		"Directory where to read training checkpoints."
	)
	tf.app.flags.DEFINE_string(
		"output_dir", "",
		"Directory where to export inference model."
	)
	tf.app.flags.DEFINE_integer(
		"model_version", 1,
		"Version number of the model."
	)
	tf.app.flags.DEFINE_integer(
		"image_size", 299,
		"Needs to provide same value as in training."
	)
	tf.app.flags.DEFINE_integer(
		"num_classes", 0,
		"Needs to provide same value as in training."
	)
	tf.app.flags.DEFINE_integer(
		"num_top_classes", 5,
		"Needs to provide same value as in training."
	)

	FLAGS = tf.app.flags.FLAGS

	assert len(FLAGS.checkpoint_dir) > 0
	assert len(FLAGS.output_dir) > 0
	assert FLAGS.num_classes != 0

	return FLAGS



if __name__ == "__main__":

	FLAGS           = define_flags()
	NUM_CLASSES     = FLAGS.num_classes
	NUM_TOP_CLASSES = FLAGS.num_top_classes

	WORKING_DIR = os.path.dirname(os.path.realpath(__file__))

	tf.app.run()








