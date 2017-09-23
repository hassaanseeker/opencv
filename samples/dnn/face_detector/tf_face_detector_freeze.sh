#!/bin/sh

# Converts variables to constants.
python ~/tensorflow/tensorflow/python/tools/freeze_graph.py \
  --input_graph face_detector.pb \
  --input_checkpoint face_detector.ckpt \
  --output_graph face_detector.pb \
  --output_node_names DetectionOutput

# Removes extra Identity layers. Perhabs, fuses weights of batch normalization and convolutions.
python ~/tensorflow/tensorflow/python/tools/optimize_for_inference.py \
  --input face_detector.pb \
  --output face_detector.pb \
  --frozen_graph True \
  --input_names data \
  --output_names DetectionOutput

# Fuses constant ops.
~/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=face_detector.pb \
  --out_graph=face_detector.pb \
  --inputs=data \
  --outputs=DetectionOutput \
  --transforms="remove_nodes(op=Cast) fold_constants(ignore_errors=true) sort_by_execution_order"

# Optimization precuderes restores fp16 weights into `half_val` field. We replace
# it into `tensor_content` manually.
python <<END
import tensorflow as tf
import struct

with tf.gfile.FastGFile('face_detector.pb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    for node in graph_def.node:
        if 'value' in node.attr:
            halfs = node.attr["value"].tensor.half_val
            if not node.attr["value"].tensor.tensor_content and halfs:
                node.attr["value"].tensor.tensor_content = struct.pack('H' * len(halfs), *halfs)
                node.attr["value"].tensor.ClearField('half_val')
    tf.train.write_graph(graph_def, "", 'face_detector.pb', as_text=False)
END
