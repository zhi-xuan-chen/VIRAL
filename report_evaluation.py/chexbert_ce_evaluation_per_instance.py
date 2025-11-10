from pprint import pprint
import pandas as pd
import os
from chexbert import CheXbert
import numpy as np

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

CONDITIONS = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]

class CheXbertMetrics():
    def __init__(self, checkpoint_path, mbatch_size, device):
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.device = device
        self.chexbert = CheXbert(self.checkpoint_path, self.device,).to(self.device)

    def compute_instance(self, gt, res):
        gt_chexbert = np.array(self.chexbert([gt]).tolist())
        res_chexbert = np.array(self.chexbert([res]).tolist())

        res_chexbert_cvt = (res_chexbert == 1)
        gt_chexbert_cvt = (gt_chexbert == 1)

        tp = (res_chexbert_cvt * gt_chexbert_cvt).astype(float)
        fp = (res_chexbert_cvt * ~gt_chexbert_cvt).astype(float)
        fn = (~res_chexbert_cvt * gt_chexbert_cvt).astype(float)

        tp_cls = tp.sum(0)
        fp_cls = fp.sum(0)
        fn_cls = fn.sum(0)

        precision_class = np.nan_to_num(tp_cls / (tp_cls + fp_cls + 1e-6))
        recall_class = np.nan_to_num(tp_cls / (tp_cls + fn_cls + 1e-6))
        f1_class = np.nan_to_num(tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls) + 1e-6))

        scores_cvt = {
            'ce_precision_macro': precision_class.mean(),
            'ce_recall_macro': recall_class.mean(),
            'ce_f1_macro': f1_class.mean(),
            'ce_precision_micro': tp_cls.sum() / (tp_cls.sum() + fp_cls.sum() + 1e-6),
            'ce_recall_micro': tp_cls.sum() / (tp_cls.sum() + fn_cls.sum() + 1e-6),
            'ce_f1_micro': tp_cls.sum() / (tp_cls.sum() + 0.5 * (fp_cls.sum() + fn_cls.sum()) + 1e-6),
            'ce_num_examples': 1.0,  # Single instance
        }
        return scores_cvt


def main():
    chexbert_metrics = CheXbertMetrics('/home/chenzhixuan/Workspace/R2GenGPT/utils/chexbert.pth', 16, 'cuda:7')  # NOTE: Update with your CheXbert model path
    report_path = "/home/chenzhixuan/Workspace/R2GenGPT/results/tmpclip_b_iu/tmpclip_b_iu.csv"
    report_data = pd.read_csv(report_path)

    image_id_column = 'Image_ID'
    gt_report_column = 'GT_whole_report'
    pred_report_column = 'Pred_whole_report'

    gt_list = report_data[gt_report_column].tolist()
    res_list = report_data[pred_report_column].tolist()
    image_ids = report_data[image_id_column].tolist()

    all_instance_scores = []
    for img_id, gt, res in zip(image_ids, gt_list, res_list):
        scores = chexbert_metrics.compute_instance(gt, res)
        scores['Image_ID'] = img_id  # Use Image_ID as the identifier
        all_instance_scores.append(scores)

    # Save the results to a CSV file
    results_df = pd.DataFrame(all_instance_scores)
    save_dir = os.path.dirname(report_path)
    save_file = os.path.join(save_dir, "instance_ce_evaluation_results.csv")
    results_df.to_csv(save_file, index=False)
    print(f"Results saved to {save_file}")


if __name__ == '__main__':
    main()