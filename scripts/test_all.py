import os
import os.path as osp
import argparse


def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", default="iomatch", help="Method")
    parser.add_argument("--n_trials",
                        "-n",
                        default=1,
                        type=int,
                        help="Repeat times")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_labels', type=int, default=75)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
    parser.add_argument("--backbone",
                        "-b",
                        default="resnet18",
                        help="Backbone")
    parser.add_argument('--use_pretrain', default=True, type=str2bool)
    parser.add_argument('--staged_lr', default=False, type=str2bool)
    parser.add_argument('--lr', default=0.01, type=float)

    args = parser.parse_args()

    # experiment name
    exp_info = '_' + str(args.lr)
    if args.use_pretrain:
        exp_info += '_pretrain'

    if args.staged_lr:
        exp_info += '_stagedlr'

    if args.exp_name:
        exp_info = exp_info + '_' + args.exp_name

    # base directory
    base_dir = osp.join('output/das6', args.method, str(args.num_labels),
                        args.backbone + exp_info)

    # multiple trials
    for i in range(args.n_trials):
        # path setting
        output_dir = osp.join(base_dir, str(i + 1))
        load_path = osp.join(output_dir, 'latest_model.pth')

        # specify random seed
        seed = args.seed
        seed += i

        os.system(
            f'CUDA_VISIBLE_DEVICES={args.gpu} '
            f'python train.py '
            f'--net {args.backbone} '
            f'--use_pretrain {args.use_pretrain} '
            f'--save_dir {output_dir} '
            f'--load_path {load_path} '
            f'--lr {args.lr} '
            f'--seed {seed} '
            f'--staged_lr {args.staged_lr} '
            f'--c config/openset_cv/{args.method}/{args.method}_das6_{args.num_labels}.yaml '
            f'--eval-only'
        )
