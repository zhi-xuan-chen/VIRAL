from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import pandas as pd
import os
from tqdm import tqdm


def compute_instance_scores(gt, pred):
    """
    Computes evaluation metrics for a single instance.

    :param gt: List of gold captions for an instance
    :param pred: List of predicted captions for an instance
    :return: Dictionary containing metrics for the instance
    """
    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    
    eval_res = {}

    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score({0: gt}, {0: pred})
        except TypeError:
            score, _ = scorer.compute_score({0: gt}, {0: pred})
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


if __name__ == "__main__":
    results_path = "/home/chenzhixuan/Workspace/R2GenGPT/results/tmpclip_b_iu/tmpclip_b_iu.csv"

    gt_report_column = 'GT_whole_report'
    pred_report_column = 'Pred_whole_report'
    id_column = 'Image_ID'

    df = pd.read_csv(results_path)

    # Print number of samples
    print(f"Number of samples: {len(df)}")  

    # Initialize a list to store results for all instances
    all_instance_metrics = []

    # Iterate over each row in the DataFrame
    for _, row in tqdm(df.iterrows()):
        id_value = row[id_column]
        gt_value = [row[gt_report_column]]  # Wrap in a list to match format
        pred_value = [row[pred_report_column]]

        # Compute metrics for the current instance
        instance_metrics = compute_instance_scores(gt_value, pred_value)
        instance_metrics[id_column] = id_value  # Add instance ID to the metrics

        # Append the instance metrics to the list
        all_instance_metrics.append(instance_metrics)

    # Convert the list of metrics into a DataFrame
    instance_metrics_df = pd.DataFrame(all_instance_metrics)

    # Save the results to a CSV file
    save_dir = os.path.dirname(results_path)
    save_file = os.path.join(save_dir, "instance_nlg_evaluation_results.csv")
    instance_metrics_df.to_csv(save_file, index=False)

    print(f"Results saved to: {save_file}")