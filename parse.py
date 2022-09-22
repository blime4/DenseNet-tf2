import tensorflow as tf
import os
from google.protobuf import text_format
import json



def _parse_file(pbtxt_file, target_device="DPU"):
    graph_def = tf.compat.v1.GraphDef()

    with tf.compat.v1.gfile.FastGFile(pbtxt_file, 'r') as f:
        text_format.Parse(f.read(), graph_def)

    ops = {}
    if target_device == "DPU" :
        target = "/job:localhost/replica:0/task:0/device:DPU:0"
    elif target_device == "GPU" :
        target = "/job:localhost/replica:0/task:0/device:GPU:0"
    elif target_device == "CPU" :
        target = "/job:localhost/replica:0/task:0/device:CPU:0"

    for node in graph_def.node:
        if node.device == target :
            if node.op in ops.keys():
                ops[node.op] += 1
            else:
                ops[node.op] = 1

    return ops

def _get_all_pbtxt_file(path):
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files if file.endswith(".pbtxt")]
    return files

def run():
    dl_pbtxt_files = _get_all_pbtxt_file("dl-dump-four-check")
    dl_ops = {}
    for file in dl_pbtxt_files:
        dl_ops[file] = _parse_file(file)
    dl_ops_json = json.dumps(dl_ops,sort_keys=False, indent=4, separators=(',', ': '))
    with open("dl-dump-four-check.json", "w") as f:
        f.write(dl_ops_json)
    f.close()
    print(dl_ops)


    # nv_pbtxt_files = _get_all_pbtxt_file("nv-dump-layout-false")
    # nv_ops = {}
    # for file in nv_pbtxt_files:
    #     nv_ops[file] = _parse_file(file, "GPU")
    # nv_ops_json = json.dumps(nv_ops,sort_keys=False, indent=4, separators=(',', ': '))
    # with open("nv-dump-layout-false_ops.json", "w") as f:
    #     f.write(nv_ops_json)
    # f.close()


def _count_all_ops(path, dst):
    count = {}
    with open(path, "r") as f:
        data = json.load(f)
    for ops in data:
        for op in data[ops]:
            if op in count.keys():
                count[op] += data[ops][op]
            else:
                count[op] = data[ops][op]

    count = sorted(count.items(), key=lambda s:s[0])
    print(count)
    ops_json = json.dumps(count,sort_keys=False, indent=4, separators=(',', ': '))
    with open(dst + "_ops_count.json", "w") as f:
        f.write(ops_json)
    f.close()




if __name__ == "__main__":
    # _parse_file("dl-dump/after_MetaOptimizer_iteration_0_remapper_140735384714672.pbtxt")
    # _get_all_pbtxt_file("dl-dump")
    run()
    _count_all_ops("dl-dump-four-check.json", "dl-dump-four-check")
    # _count_all_ops("dl-dump-double-check-GPU.json", "dl-double-check-GPU")
    # _count_all_ops("nv_ops.json", "nv")
    # _count_all_ops("nv-dump-layout-false_ops.json", "nv-dump-layout-false")