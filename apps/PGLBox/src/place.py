
import os
import paddle.fluid.core as core


def get_cuda_places():
    gpus_env = os.getenv("FLAGS_selected_gpus")
    if gpus_env:
        device_ids = [int(s) for s in gpus_env.split(",")]
    else:
        device_ids = list(range(core.get_cuda_device_count()))
        os.environ["FLAGS_selected_gpus"] = ",".join([str(w) for w in device_ids])
    return device_ids

get_cuda_places()


