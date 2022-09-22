import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.python.platform import gfile

model = 'nv-logdir-823/plugins/profile/2022_08_23_19_08_40/NV3090-32-41.input_pipeline.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('log/', graph)