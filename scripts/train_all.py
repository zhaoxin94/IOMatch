import os
import os.path as osp
import argparse

from semilearn.algorithms.utils import str2bool


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

    args = parser.parse_args()

    # experiment name
    exp_info = ''
    if args.use_pretrain:
        exp_info += '_pretrain'
    
    if args.exp_name:
        exp_info = exp_info + '_' + args.exp_name

    # base directory
    base_dir = osp.join('output/das6', args.num_labels, args.method, 
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
            f'--seed {args.seed} '
            f'--c config/openset_cv/{args.method}/{args.method}_das6_{args.num_labels}_pretrain.yaml')
