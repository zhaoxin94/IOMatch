import os.path as osp
import argparse
from collections import defaultdict, OrderedDict
import re
import numpy as np

from dassl.utils.tools import listdir_nohidden, check_isfile


def write_row(row, colwidth=10):
    sep = "  "

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.2f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    return sep.join([format_val(x) for x in row]) + "\n"


# compute results across different seeds
def parse_function(*metrics, directory="", end_signal=None):
    
    print(f"Parsing files in {directory}")
    
    fpath = osp.join(directory, "log.txt")
    assert check_isfile(fpath)
    good_to_go = False
    output = OrderedDict()

    with open(fpath, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()

            if end_signal in line:
                good_to_go = True

            for metric in metrics:
                match = metric["regex"].search(line)
                if match and good_to_go:
                    if "file" not in output:
                        output["file"] = fpath
                    num = float(match.group(1))
                    name = metric["name"]
                    output[name] = num

    print(good_to_go)
    assert output, f"Nothing found in {directory}"

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default='', help="Method")
    
    args = parser.parse_args()

    base_dir = args.path
    method = base_dir.split('/')[2]

    print('*****************************************************************')
    print(f'Extract results from {base_dir}')
    print('*****************************************************************\n')

    # parse results
    end_signal = "Final Evaluation"

    metrics = []
    metric_names = ["C_Acc", "C_Precision", "C_Recall", "C_F1-score", 'AUROC']
    for metric_name in metric_names:
        regex_str = re.compile(fr"{metric_name}:([\.\deE+-]+)")
        metric = {"name": metric_name, "regex": regex_str}
        metrics.append(metric)

    results = []
    for directory in listdir_nohidden(base_dir, sort=True):
        directory = osp.join(base_dir, directory)
        if osp.isdir(directory):
            result = parse_function(*metrics,
                                     directory=directory,
                                     end_signal=end_signal)
            results.append(result)

    metrics_results = defaultdict(list)

    for result in results:
        msg = ""
        for key, value in result.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {base_dir}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
    print("===")

    row1 = ['Method']
    row2 = [method]

    for key, value in output_results.items():
        row1.append(key)
        row2.append(value)

    results_path = osp.join(base_dir, 'collect_results.txt')
    with open(results_path, 'w') as f:
        f.write(write_row(row1))
        f.write(write_row(row2))




            

        
