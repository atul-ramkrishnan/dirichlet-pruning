import numpy as np
import vgg
from train import train


def prune_and_retrain(switch_save_path, thresholds, model_save_dir, cpu, resume, eval, batch_size, workers, lr, momentum, weight_decay, start_epoch, epochs, print_freq):
    pruned_arch_layer = [int(n) for n in thresholds.split(",")]
    switches =list(np.load(switch_save_path,  allow_pickle=True))

    model = vgg.vgg16_bn()
    for name, param in model.named_parameters():
        if (("Conv2d" in name) or ("fc" in name)) and ("weight" in name):
            it += 1
            param.data[switches[it - 1]] = 0
            # print(param.data)
        if (("Conv2d" in name) or ("fc" in name)) and ("bias" in name):
            param.data[switches[it - 1]] = 0
            # print(param.data)
        if ("BatchNorm2d" in name) and ("weight" in name):
            param.data[switches[it - 1]] = 0
        if ("BatchNorm2d" in name) and ("bias" in name):
            param.data[switches[it - 1]] = 0
        if ("BatchNorm2d" in name) and ("running_mean" in name):
            param.data[switches[it - 1]] = 0
        if ("BatchNorm2d" in name) and ("running_var" in name):
            param.data[switches[it - 1]] = 0

    def gradi_new(combs_num):
        def hook(module):
            module[switches[combs_num]] = 0
        return hook

    # Backward hooks
    for name, param in model.named_parameters():
        if "Conv2d" in name or "BatchNorm" in name:
            param.register_hook(gradi_new(name[name.find('_') + 1:name.rfind('.')]))
        elif "fc" in name:
            param.register_hook(gradi_new(16 + name[name.find('_') + 1:name.rfind('.')]))
    print("Retraining")
    train("prune_and_retrain",
        model_save_dir,
        cpu, resume,
        eval,
        batch_size,
        workers,
        lr,
        momentum,
        weight_decay,
        start_epoch,
        epochs,
        print_freq)