import numpy as np
import os
import glob
import time
import argparse
import copy
import mcubes, trimesh
import torch
from chamfer_distance import ChamferDistance
import json

#NOTE: this evaluation code is taken from previous work PatchComplete https://github.com/yuchenrao/PatchComplete/blob/main/evaluation/evaluation.py


CATEGORY_LIST = [
    ("04379243", "table"),
    ("03593526", "jar"),
    ("04225987", "skateboard"),
    ("02958343", "car"),
    ("02876657", "bottle"),
    ("04460130", "tower"),
    ("03001627", "chair"),
    ("02871439", "bookshelf"),
    ("02942699", "camera"),
    ("02691156", "airplane"),
    ("03642806", "laptop"),
    ("02801938", "basket"),
    ("04256520", "sofa"),
    ("03624134", "knife"),
    ("02946921", "can"),
    ("04090263", "rifle"),
    ("04468005", "train"),
    ("03938244", "pillow"),
    ("03636649", "lamp"),
    ("02747177", "trash bin"),
    ("03710193", "mailbox"),
    ("04530566", "watercraft"),
    ("03790512", "motorbike"),
    ("03207941", "dishwasher"),
    ("02828884", "bench"),
    ("03948459", "pistol"),
    ("04099429", "rocket"),
    ("03691459", "loudspeaker"),
    ("03337140", "file cabinet"),
    ("02773838", "bag"),
    ("02933112", "cabinet"),
    ("02818832", "bed"),
    ("02843684", "birdhouse"),
    ("03211117", "display"),
    ("03928116", "piano"),
    ("03261776", "earphone"),
    ("04401088", "telephone"),
    ("04330267", "stove"),
    ("03759954", "microphone"),
    ("02924116", "bus"),
    ("03797390", "mug"),
    ("04074963", "remote"),
    ("02808440", "bathtub"),
    ("02880940", "bowl"),
    ("03085013", "keyboard"),
    ("03467517", "guitar"),
    ("04554684", "washer"),
    ("02834778", "bicycle"),
    ("03325088", "faucet"),
    ("04004475", "printer"),
    ("02954340", "cap"),
]

CATEGORY_NAME_MAP = dict(CATEGORY_LIST)


class Evaluation:
    """
    This class caluate Chamfer Distance and IOU between GTs and predictions

    Steps:
    1. Read GT and prediction
    2. Convert voxels to meshes
    3. Sample points on the meshes and then calcualte chamfer distance
    4. Calculate IOU and F1
    """
    def __init__(self, args):
        """
        Initializes Evaluation object. Sets up related pathes and other parameters.

        :param args: arguments from the commandline
        :type args: argparse.Namespace
        """
        # init path for related files
        self._root = args.root
        if os.path.exists(self._root) is not True:
            print('Please enter valid data root path')
            return
        self._pred_path = args.pred_path
        if os.path.exists(self._pred_path) is not True:
            print('Please enter valid pred path')
            return
        self._test_file = args.test_file
        if os.path.exists(self._test_file) is not True:
            print('Please enter valid test file path')
            return

        # set paths based on dataset
        self._data_path = os.path.join(self._root, args.dataset)
        self._pred_path = os.path.join(self._pred_path, args.dataset)
        # get output file
        self._points_n = args.points_n
        self._chamfer_dist = ChamferDistance()

    def calculate_iou(self, gt, pred, threshold):
        bool_true_voxels = gt > threshold
        bool_pred_voxels = pred > threshold
        total_union = (bool_true_voxels | bool_pred_voxels).sum()
        total_intersection = (bool_true_voxels & bool_pred_voxels).sum()
        return (total_intersection / total_union)

    def calculate_f1(self, gt_mask, pred_mask):
        mask_t = copy.deepcopy(gt_mask)
        mask_t[np.where(pred_mask == 1)] += 10
        miss = len(np.where(mask_t == 1)[0]) / np.sum(gt_mask)
        redundant = len(np.where(mask_t == 10)[0]) / np.sum(gt_mask)
        f1 = np.sum(np.logical_and(gt_mask, pred_mask)) / (
            np.sum(np.logical_and(gt_mask, pred_mask)) +
            0.5 * np.sum(np.logical_xor(gt_mask, pred_mask))
        )
        return miss, redundant, f1

    def calculate_cd(self, gt_mask, pred_mask):
        # get points for prediction
        if np.sum(pred_mask) == 0:
            pred_points = np.zeros((self._points_n, 3))
        else:
            pred_points, _ = self.get_surface_points(pred_mask, 0, 32)
        # get points for gt
        gt_points, _ = self.get_surface_points(gt_mask, 0, 32)
        # calcualte CD
        gt_points_torch = torch.from_numpy(gt_points).cuda().unsqueeze(0).float()
        pred_points_torch = torch.from_numpy(pred_points).cuda().unsqueeze(0).float()
        dist1, dist2 = self._chamfer_dist(gt_points_torch, pred_points_torch)
        eps = 1e-10
        loss = torch.sqrt(dist1 + eps).mean(1) + torch.sqrt(dist2 + eps).mean(1)
        return loss.detach().cpu().numpy()

    def get_surface_points(self, V, threshold, voxel_res):
        # padding
        logits = np.pad(V, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        vertices, triangles = mcubes.marching_cubes(logits, threshold)
        # recale to [0,1]
        vertices -= 1
        step = 1 / (voxel_res - 1)
        vertices = np.multiply(vertices, step)
        mesh = trimesh.Trimesh(vertices, triangles)
        points = mesh.sample(self._points_n)

        return points, mesh

    def evaluate_shapenet(self):
        """
        This function gets final cd evaluation results
        """
        start_time = time.time()
        eval_res = {}
        # iterate all the files
        with open(self._test_file, 'r') as data:
            gt_files = data.readlines()
            for gt_file in gt_files:
                gt_file = gt_file.split('\n')[0]
                pred_model_path = os.path.join(self._pred_path, gt_file)
                gt_model_path = os.path.join(self._data_path, gt_file)
                gt_name = os.path.join(gt_model_path, "gt.npz")
                with np.load(gt_name, 'rb') as data_npz:
                    gt = data_npz['tsdf']
                    gt_mask = np.zeros(gt.shape)
                    gt_mask[np.where(gt <= 1e-10)] = 1

                for pred_name in glob.glob(os.path.join(pred_model_path, "*_pred.npz")):
                    with np.load(pred_name, allow_pickle=True) as raw_data:
                        data_item = raw_data["data"].item()
                        pred = data_item['output'].squeeze(0)
                        pred_mask = np.zeros(pred.shape)
                        pred_mask[np.where(pred <= 1e-10)] = 1

                    # evaluate IOU, f1 and cd
                    iou = self.calculate_iou(gt_mask, pred_mask, 0.5)
                    miss, redundant, f1 = self.calculate_f1(gt_mask, pred_mask)
                    cd = self.calculate_cd(gt_mask, pred_mask)
                    eval_res[pred_name] = [cd[0], iou, miss, redundant, f1, pred_name]

        print("costing time:")
        print(time.time() - start_time)
        return eval_res

    def evaluate_scannet(self):
        """
        This function gets final cd evaluation results
        """
        start_time = time.time()
        eval_res = {}
        missing_gt = 0
        missing_pred = 0
        total = 0

        # iterate all the files
        with open(self._test_file, 'r') as data:
            gt_files = data.readlines()
            for gt_file in gt_files:
                total += 1
                gt_file = gt_file.split('\n')[0]
                mask_file = os.path.join(self._data_path, gt_file)
                pred_file = os.path.join(self._pred_path, gt_file)
                pred_file = pred_file[:-4] + '_pred.npz'
                gt_sdf_file = mask_file[:-8] + "scaled_sdf_gt.npy"

                if not os.path.exists(gt_sdf_file):
                    missing_gt += 1
                    continue

                if not os.path.exists(pred_file):
                    missing_pred += 1
                    continue

                with open(gt_sdf_file, 'rb') as data_file:
                    gt = np.load(data_file)
                    gt_mask = np.zeros(gt.shape)
                    gt_mask[np.where(gt <= 1e-10)] = 1

                with np.load(pred_file, allow_pickle=True) as raw_data:
                    data_item = raw_data["data"].item()
                    pred = data_item['output'].squeeze(0)
                    pred_mask = np.zeros(pred.shape)
                    pred_mask[np.where(pred <= 1e-10)] = 1

                # evaluate IOU, f1 and cd
                iou = self.calculate_iou(gt_mask, pred_mask, 0.5)
                miss, redundant, f1 = self.calculate_f1(gt_mask, pred_mask)
                cd = self.calculate_cd(gt_mask, pred_mask)
                eval_res[gt_file] = [cd[0], iou, miss, redundant, f1, gt_file]

        print("costing time:")
        print(time.time() - start_time)
        print(f"total test entries: {total}")
        print(f"missing gt: {missing_gt}")
        print(f"missing pred: {missing_pred}")
        print(f"evaluated: {len(eval_res)}")
        return eval_res


def parse_arguments():
    """
    Generates a command line parser that supports all arguments used by the
    tool.

    :return: Parser with all arguments.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    # required parameters
    parser.add_argument("--dataset",
                        help="dataeset to be trained",
                        default="shapenet",
                        required=True,
                        type=str)
    parser.add_argument("--root",
                        help="root for reading and saving data",
                        default='data_samples',
                        type=str)
    parser.add_argument("--pred_path",
                        help="path for reading mesh data",
                        default='output',
                        type=str)
    # optional parameters
    parser.add_argument("--test_file",
                        help="test samples names",
                        default="test.txt",
                        type=str)
    parser.add_argument("--points_n",
                        help="the number of sampling points for chamfer distance",
                        default=10240,
                        type=int)
    return parser


def print_results(eval_res, dataset, pred_root=None):
    import os
    import json
    import numpy as np

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def format_entry(entry, rank_idx=None):
        cd, iou, miss, red, f1, name = entry
        prefix = f"{rank_idx}. " if rank_idx is not None else ""
        return (
            f"{prefix}{name}\n"
            f"    CD   : {cd:.6f}\n"
            f"    IoU  : {iou:.6f}\n"
            f"    Miss : {miss:.6f}\n"
            f"    Red  : {red:.6f}\n"
            f"    F1   : {f1:.6f}\n"
        )

    def resolve_report_dir(dataset, pred_root):
        if pred_root is None:
            out_dir = os.getcwd()
        else:
            pred_root = os.path.abspath(pred_root)
            if os.path.basename(pred_root) == dataset:
                out_dir = os.path.dirname(pred_root)
            else:
                out_dir = pred_root
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    if len(eval_res) == 0:
        print("No valid prediction/GT pairs found. Check pred_path, root, and test file.")
        summary = {
            "dataset": dataset,
            "instance_cd": None,
            "category_cd": None,
            "instance_iou": None,
            "category_iou": None,
            "instance_miss": None,
            "category_miss": None,
            "instance_red": None,
            "category_red": None,
            "instance_f1": None,
            "category_f1": None,
            "n": 0,
        }
        if os.environ.get("EVAL_JSON", "0") == "1":
            print("EVAL_SUMMARY_JSON=" + json.dumps(summary))
        return summary

    cats_iou = {}
    if dataset == 'shapenet':
        for name, eval_r in eval_res.items():
            cat = name.split('/')[-3]
            cats_iou.setdefault(cat, []).append(eval_r)

    elif dataset == 'scannet':
        for name, eval_r in eval_res.items():
            cat = name.split('/')[-1].split('_')[0]
            cats_iou.setdefault(cat, []).append(eval_r)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    sum_cd = 0.0
    sum_iou = 0.0
    sum_miss = 0.0
    sum_red = 0.0
    sum_f1 = 0.0
    sum_len = 0

    cat_cds = []
    cat_ious = []
    cat_misses = []
    cat_reds = []
    cat_f1s = []

    per_category_summary = {}
    all_entries = []

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append(f"EVALUATION REPORT | dataset={dataset}")
    report_lines.append("=" * 100)
    report_lines.append("")

    for cat, data_l in sorted(cats_iou.items()):
        cat_name = CATEGORY_NAME_MAP.get(cat, "unknown")

        cat_cd = []
        cat_iou = []
        cat_miss = []
        cat_red = []
        cat_f1 = []

        for cd, iou, miss, red, f1, name in data_l:
            sum_cd += cd
            sum_iou += iou
            sum_miss += miss
            sum_red += red
            sum_f1 += f1

            cat_cd.append(cd)
            cat_iou.append(iou)
            cat_miss.append(miss)
            cat_red.append(red)
            cat_f1.append(f1)

            all_entries.append((cd, iou, miss, red, f1, name, cat, cat_name))

        cat_cd_mean = float(np.mean(cat_cd))
        cat_iou_mean = float(np.mean(cat_iou))
        cat_miss_mean = float(np.mean(cat_miss))
        cat_red_mean = float(np.mean(cat_red))
        cat_f1_mean = float(np.mean(cat_f1))

        cat_cds.append(cat_cd_mean)
        cat_ious.append(cat_iou_mean)
        cat_misses.append(cat_miss_mean)
        cat_reds.append(cat_red_mean)
        cat_f1s.append(cat_f1_mean)
        sum_len += len(data_l)

        sorted_all = sorted(data_l, key=lambda x: x[1], reverse=True)

        per_category_summary[cat] = {
            "category_name": cat_name,
            "n": len(data_l),
            "cd": cat_cd_mean,
            "iou": cat_iou_mean,
            "miss": cat_miss_mean,
            "red": cat_red_mean,
            "f1": cat_f1_mean,
            "all_entries_sorted_by_iou": [
                {
                    "name": e[5],
                    "cd": safe_float(e[0]),
                    "iou": safe_float(e[1]),
                    "miss": safe_float(e[2]),
                    "red": safe_float(e[3]),
                    "f1": safe_float(e[4]),
                }
                for e in sorted_all
            ],
        }

        print(f"{cat} ({cat_name}): n={len(data_l)} | "
              f"CD={cat_cd_mean:.6f} | "
              f"IoU={cat_iou_mean:.6f} | "
              f"Miss={cat_miss_mean:.6f} | "
              f"Red={cat_red_mean:.6f} | "
              f"F1={cat_f1_mean:.6f}")

        report_lines.append("-" * 100)
        report_lines.append(f"Category: {cat} ({cat_name})")
        report_lines.append(
            f"n={len(data_l)} | CD={cat_cd_mean:.6f} | IoU={cat_iou_mean:.6f} | "
            f"Miss={cat_miss_mean:.6f} | Red={cat_red_mean:.6f} | F1={cat_f1_mean:.6f}"
        )
        report_lines.append("")
        report_lines.append("ALL RESULTS IN THIS CATEGORY (sorted by IoU descending):")
        report_lines.append("")

        for idx, entry in enumerate(sorted_all, 1):
            report_lines.append(format_entry(entry, idx))

        report_lines.append("")

    if sum_len == 0:
        print("sum_len is 0 → nothing was evaluated. Check paths.")
        summary = {
            "dataset": dataset,
            "instance_cd": None,
            "category_cd": None,
            "instance_iou": None,
            "category_iou": None,
            "instance_miss": None,
            "category_miss": None,
            "instance_red": None,
            "category_red": None,
            "instance_f1": None,
            "category_f1": None,
            "n": 0,
        }
        if os.environ.get("EVAL_JSON", "0") == "1":
            print("EVAL_SUMMARY_JSON=" + json.dumps(summary))
        return summary

    instance_cd = sum_cd / sum_len
    category_cd = float(np.array(cat_cds).mean())

    instance_iou = sum_iou / sum_len
    category_iou = float(np.array(cat_ious).mean())

    instance_miss = sum_miss / sum_len
    category_miss = float(np.array(cat_misses).mean())

    instance_red = sum_red / sum_len
    category_red = float(np.array(cat_reds).mean())

    instance_f1 = sum_f1 / sum_len
    category_f1 = float(np.array(cat_f1s).mean())

    print("instance_cd")
    print(instance_cd)
    print("category cd")
    print(category_cd)
    print("instance_iou")
    print(instance_iou)
    print("category iou")
    print(category_iou)
    print("instance_miss")
    print(instance_miss)
    print("category miss")
    print(category_miss)
    print("instance_red")
    print(instance_red)
    print("category red")
    print(category_red)
    print("instance_f1")
    print(instance_f1)
    print("category f1")
    print(category_f1)
    print(sum_len)

    all_entries_plain = [(cd, iou, miss, red, f1, name, cat, cat_name)
                         for cd, iou, miss, red, f1, name, cat, cat_name in all_entries]
    all_entries_sorted = sorted(all_entries_plain, key=lambda x: x[1], reverse=True)

    report_lines.append("=" * 100)
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("=" * 100)
    report_lines.append(f"Total evaluated: {sum_len}")
    report_lines.append(f"Instance CD   : {instance_cd:.6f}")
    report_lines.append(f"Category CD   : {category_cd:.6f}")
    report_lines.append(f"Instance IoU  : {instance_iou:.6f}")
    report_lines.append(f"Category IoU  : {category_iou:.6f}")
    report_lines.append(f"Instance Miss : {instance_miss:.6f}")
    report_lines.append(f"Category Miss : {category_miss:.6f}")
    report_lines.append(f"Instance Red  : {instance_red:.6f}")
    report_lines.append(f"Category Red  : {category_red:.6f}")
    report_lines.append(f"Instance F1   : {instance_f1:.6f}")
    report_lines.append(f"Category F1   : {category_f1:.6f}")
    report_lines.append("")

    report_lines.append("=" * 100)
    report_lines.append("ALL RESULTS OVERALL (sorted by IoU descending)")
    report_lines.append("=" * 100)
    report_lines.append("")

    for idx, entry in enumerate(all_entries_sorted, 1):
        cd, iou, miss, red, f1, name, cat, cat_name = entry
        report_lines.append(
            f"{idx}. {name}\n"
            f"    Category : {cat} ({cat_name})\n"
            f"    CD       : {cd:.6f}\n"
            f"    IoU      : {iou:.6f}\n"
            f"    Miss     : {miss:.6f}\n"
            f"    Red      : {red:.6f}\n"
            f"    F1       : {f1:.6f}\n"
        )

    summary = {
        "dataset": dataset,
        "instance_cd": float(instance_cd),
        "category_cd": float(category_cd),
        "instance_iou": float(instance_iou),
        "category_iou": float(category_iou),
        "instance_miss": float(instance_miss),
        "category_miss": float(category_miss),
        "instance_red": float(instance_red),
        "category_red": float(category_red),
        "instance_f1": float(instance_f1),
        "category_f1": float(category_f1),
        "n": int(sum_len),
        "per_category": per_category_summary,
        "all_results_sorted_by_iou": [
            {
                "name": e[5],
                "category_id": e[6],
                "category_name": e[7],
                "cd": safe_float(e[0]),
                "iou": safe_float(e[1]),
                "miss": safe_float(e[2]),
                "red": safe_float(e[3]),
                "f1": safe_float(e[4]),
            }
            for e in all_entries_sorted
        ],
    }

    out_dir = resolve_report_dir(dataset, pred_root)
    txt_path = os.path.join(out_dir, f"{dataset}_evaluation_report.txt")
    json_path = os.path.join(out_dir, f"{dataset}_evaluation_report.json")

    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved evaluation text report to: {txt_path}")
    print(f"Saved evaluation json report to: {json_path}")

    if os.environ.get("EVAL_JSON", "0") == "1":
        print("EVAL_SUMMARY_JSON=" + json.dumps(summary))

    return summary


def main():
    """
    Evaluate chamfer distance on prediction
    """
    args = parse_arguments().parse_args()
    print(args)
    evaluator = Evaluation(args)

    if args.dataset == "shapenet":
        eval_res = evaluator.evaluate_shapenet()
        print_results(eval_res, "shapenet", pred_root=evaluator._pred_path)

    if args.dataset == "scannet":
        eval_res = evaluator.evaluate_scannet()
        print_results(eval_res, "scannet", pred_root=evaluator._pred_path)


if __name__ == "__main__":
    main()