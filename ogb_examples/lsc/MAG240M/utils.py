import os
import paddle
import paddle.distributed
import glob
import shutil
from pgl.utils.logger import log

def _create_if_not_exist(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

def save_model(output_path, model, steps, opt, lr_scheduler, max_ckpt=2):
    if paddle.distributed.get_rank() == 0:
        output_dir = os.path.join(output_path,
                                        "model_%d" % steps)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        paddle.save(model.state_dict(), os.path.join(output_dir, "ckpt.pdparams" ))
#         paddle.save({ "global_step": steps }, os.path.join(output_dir, "step"))
#         paddle.save(opt.state_dict(), os.path.join(output_dir, "opt.pdparams"))
#         paddle.save(lr_scheduler.state_dict(), os.path.join(output_dir, "lr_scheduler.pdparams"))

        log.info("save model %s" % output_dir)

        ckpt_paths = glob.glob(os.path.join(output_path, "model_*"))
        if len(ckpt_paths) > max_ckpt:
            def version(x):
                x = int(x.split("_")[-1])
                return x
            rm_ckpt_paths = sorted(ckpt_paths, key=version, reverse=True)[max_ckpt:]
            for ckpt_dir in rm_ckpt_paths:
                if os.path.exists(ckpt_dir):
                    shutil.rmtree(ckpt_dir)


def load_model(output_path, model, opt=None, lr_scheduler=None):
    def version(x):
        x = int(x.split("_")[-1])
        return x

    ckpt_paths = glob.glob(os.path.join(output_path, "model_*"))
    steps = 0
    if len(ckpt_paths) > 0:
        output_dir = sorted(ckpt_paths, key=version, reverse=True)[0]

        model_state_dict = paddle.load(os.path.join(output_dir, "ckpt.pdparams" ))
        model.set_state_dict(model_state_dict)
        log.info("load model from  %s" % output_dir)

        if opt is not None and os.path.exists(os.path.join(output_dir, "opt.pdparams" )):
            opt_state_dict = paddle.load(os.path.join(output_dir, "opt.pdparams" ))
            opt.set_state_dict(opt_state_dict)
            log.info("restore optimizer")

        if lr_scheduler is not None and os.path.exists(os.path.join(output_dir, "lr_scheduler.pdparams" )):
            lr_scheduler_state_dict = paddle.load(os.path.join(output_dir, "lr_scheduler.pdparams" ))
            lr_scheduler.set_state_dict(lr_scheduler_state_dict)
            log.info("restore lr_scheduler")

        if os.path.exists(os.path.join(output_dir, "lr_scheduler.pdparams" )):
            steps = paddle.load(os.path.join(output_dir, "step" ))["global_step"]
            log.info("restore steps")
    return steps
    

def infinite_loop(train_ds):
    while True:
        for data in train_ds.generator():
            yield data


        
