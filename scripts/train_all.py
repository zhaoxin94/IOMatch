import os
import os.path as osp
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", default="iomatch", help="Method")
    parser.add_argument("--n_trials",
                        "-n",
                        default=1,
                        type=int,
                        help="Repeat times")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
    parser.add_argument("--backbone",
                        "-b",
                        default="resnet18",
                        help="Backbone")

    args = parser.parse_args()

    exp_info = args.exp_name
    if exp_info:
        exp_info = '_' + exp_info

    exp_info = exp_info + f'_lr={args.lr}_epoch={args.epoch}'

    base_dir = osp.join('output/DAS', args.method, args.dataset,
                        args.backbone + exp_info)

    for i in range(args.n_trials):
        output_dir = osp.join(base_dir, str(i + 1))
        seed = args.seed
        seed += i

        os.system(
            f'CUDA_VISIBLE_DEVICES={args.gpu} '
            f'python train.py '
            f'--c config/openset_cv/{args.method}/{args.method}_das6_pretrain.yaml')
