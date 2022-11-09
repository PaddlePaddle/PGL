import paddle as P
import re

def get_warmup_and_linear_decay(max_steps, warmup_steps):
    warmup_steps = max(warmup_steps, 1)
    def fn(step):
        if step < warmup_steps:
            return 1
        elif step == warmup_steps:
            return 0.5
        else:
            pos_step = step - warmup_steps 
            return 0.5 * 0.75 ** (pos_step // 5)
    return fn
def get_optimizer(parameters, learning_rate, max_steps, 
                  weight_decay, warmup_proportion, clip=-1,
                  use_lr_decay=True):
    
    if clip > 0:
        g_clip = P.nn.ClipGradByGlobalNorm(clip)
    else:
        g_clip = None

    param_name_to_exclue_from_weight_decay = re.compile(
        r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')

    if use_lr_decay:
        #lr_scheduler = P.optimizer.lr.StepDecay(learning_rate=learning_rate,
        #                                        step_size=25, gamma=0.5,)
        #opt = P.optimizer.AdamW(
        #    lr_scheduler,
        #    parameters=parameters,
        #    weight_decay=weight_decay,
        #    apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n),
        #    grad_clip=g_clip)
        lr_scheduler = P.optimizer.lr.LambdaDecay(
            learning_rate,
            get_warmup_and_linear_decay(
                max_steps, 25))
        opt = P.optimizer.AdamW(
            lr_scheduler,
            parameters=parameters,
            weight_decay=weight_decay,
            apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n),
            grad_clip=g_clip)
    else:
        lr_scheduler = None
        opt = P.optimizer.AdamW(
            learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n),
            grad_clip=g_clip)
    return opt, lr_scheduler


