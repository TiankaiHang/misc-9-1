import os
import numpy as np
import matplotlib.pylab as plt


def get_grad_norm_from_file(fp):
    if not os.path.exists(fp):
        return -1
    
    with open(fp, "r") as f:
        grad_norms = []
        for line in f.readlines():
            line = line.strip()
            if 'grad_norm' in line:
                grad_norm = line.split('|')[-2]
                grad_norm = float(grad_norm)

                grad_norms.append(grad_norm)
    try:
        return sum(grad_norms) / len(grad_norms)
    except ZeroDivisionError:
        return -1


def parse_exp_dir(dir):
    sub_dirs = os.listdir(dir)
    sub_dirs = [int(x) for x in sub_dirs]
    sub_dirs.sort()
    sub_dirs = [f"{x:06d}" for x in sub_dirs]
    print(sub_dirs)

    # different iters
    grad_norm_at_iters = []
    for _sub_dir in sub_dirs:
        # from-to dir

        # different timesteps for each iter
        grad_norm_at_steps = []
        for i in range(100):
            _from = 10 * i
            _to = 10 * (i + 1) - 1

            _fp = f"{dir}/{_sub_dir}/from-to-{_from}-{_to}/log.txt"
            grad_norm = get_grad_norm_from_file(_fp)
            # print(grad_norm)
            if grad_norm > 0:
                grad_norm_at_steps.append(grad_norm)

        grad_norm_at_iters.append(grad_norm_at_steps)

    return grad_norm_at_iters


def main():
    # x = get_grad_norm_from_file("/home/v-tiahang/code_base/others/analyze_gradnorm/data/28/nov11_imnet_beit_base_layer12_lr1e-4_099_099_img64_pred_x0_loss-eps_fp16_bs8x64/070000/from-to-150-159/log.txt")
    # print(x)

    exp_list = [
        "nov3_imnet_beit_base_layer12_lr1e-4_099_099_img64_pred_noise_fp16_bs8x64",
        "nov11_imnet_beit_base_layer12_lr1e-4_099_099_img64_pred_x0_loss-eps_fp16_bs8x64",
        "nov15_imnet_beit_base_layer12_lr1e-4_099_099_img64_pred_x0__min_snr_1__fp16_bs8x64",
        "nov15_imnet_beit_base_layer12_lr1e-4_099_099_img64_pred_x0__min_snr_5__fp16_bs8x64",
        "nov23_imnet_beit_base_layer12_lr1e-4_099_099_img64_pred_noise__min_snr_1__fp16_bs8x64",
        "nov23_imnet_beit_base_layer12_lr1e-4_099_099_img64_pred_noise__min_snr_5__fp16_bs8x64",
    ]
    
    for exp in exp_list:
        result = parse_exp_dir(f"data/28/{exp}")
        # import pdb; pdb.set_trace()

        for iter, norms_at_iter in enumerate(result):
            plt.plot(norms_at_iter, label=f"{(iter + 1)*10000}")

        plt.legend()
        plt.savefig(f"{exp}.png")

        plt.close(); plt.clf()
        # import pdb; pdb.set_trace()
        

if __name__ == '__main__':
    main()