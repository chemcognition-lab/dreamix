import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append( str(base_dir / 'src/') )

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chemix.utils import evaluate

parser = ArgumentParser()
parser.add_argument("--dir", action="store", type=str, default='results/chemix_ensemble', help="Folder with ensemble of results.")
FLAGS = parser.parse_args()

def get_ensemble_predictions(prediction_set: str, directory: str, topk: int = 10):
    """
    Loop through topk models in the directory. Assuming the models are saved in 
    files named as `top{i}` for i an integer in [1,topk].
    Specify the prediction_set in ['leaderboard', 'test']
    Returns a dataframe with all results.
    """
    all_df = []
    for i in range(topk):
        i += 1
        df = pd.read_csv(f'{directory}/top{i}/{prediction_set}_predictions.csv')
        df['data_index'] = range(len(df))
        df['run'] = i
        all_df.append(df)
    return pd.concat(all_df)


if __name__ == '__main__':
    ##### LEADERBOARD SET #####
    # loop through top10 in ensemble
    all_df = get_ensemble_predictions('leaderboard', FLAGS.dir)

    # calculate averages
    average_pred = all_df.groupby('data_index')['Predicted_Experimental_Values'].mean().to_numpy()
    ground_truth = all_df.groupby('data_index')['Ground_Truth'].mean().to_numpy()

    # calculate a bunch of metrics on the results to compare
    leaderboard_metrics = evaluate(ground_truth, average_pred)
    leaderboard_metrics = pd.DataFrame(leaderboard_metrics, index=['metrics']).transpose()
    leaderboard_metrics.to_csv(f'{FLAGS.dir}/ensemble_leaderboard_metrics.csv')

    # plot the predictions with ground truth, include all metrics in legend
    ax = sns.lineplot(data=all_df, x='Ground_Truth', y='Predicted_Experimental_Values', marker='o', linestyle='', err_style='bars')
    ax.plot([0,1], [0,1], 'r--')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.annotate(''.join(f'{k}: {v['metrics']:.4f}\n' for k, v in leaderboard_metrics.iterrows()).strip(),
                xy=(0.05,0.7), xycoords='axes fraction',
                size=12,
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))
    plt.savefig(f'{FLAGS.dir}/ensemble_leaderboard_predictions.png', bbox_inches='tight')
    plt.close()

    # save the results
    save_pred = pd.DataFrame({
        'Predicted_Experimental_Values': average_pred, 
        'Ground_Truth': ground_truth,
    }, index=range(len(average_pred)))
    save_pred.to_csv(f'{FLAGS.dir}/ensemble_leaderboard_predictions.csv')


    ##### TEST SET #####
    all_df = get_ensemble_predictions('test', FLAGS.dir)
    average_pred = all_df.groupby('data_index')['Predicted_Experimental_Values'].mean().to_numpy()

    # save the results
    save_pred = pd.DataFrame({
        'Predicted_Experimental_Values': average_pred, 
    }, index=range(len(average_pred)))
    save_pred.to_csv(f'{FLAGS.dir}/ensemble_test_predictions.csv')



