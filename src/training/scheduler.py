import numpy as np
from torch.optim.swa_utils import get_ema_multi_avg_fn

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def const_lr_cooldown(optimizer, base_lr, warmup_length, steps, cooldown_steps, cooldown_power=1.0, cooldown_end_lr=0.):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e/es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
def get_ema_with_warmup(args):
    ema = get_ema_multi_avg_fn(args.elr_ema_decay)
    num_rampup_epochs = args.elr_teacher_warmup
    steps_per_epoch = np.ceil(args.train_num_samples / (args.log_every_n_steps * args.batch_size) / args.world_size)
    
    def avg_fn(ema_param_list, current_param_list, num_avg):
        if num_avg < num_rampup_epochs * steps_per_epoch:
            decay = min(sigmoid_rampup(num_avg.cpu(), num_rampup_epochs * steps_per_epoch), args.elr_ema_decay)
            return get_ema_multi_avg_fn(decay)(ema_param_list, current_param_list, num_avg)
        else:
            return ema(ema_param_list, current_param_list, num_avg)
        
    return avg_fn
        

