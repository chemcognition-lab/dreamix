import numpy as np
import seaborn as sns
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchmetrics
from scipy.stats import wilcoxon

# make things nicer to look at 
sns.set_context('talk', font_scale=2)
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['errorbar.capsize'] = 10

def cast_float(x):
    return x if isinstance(x, float) else x.item()

def bootstrap_ci(true_values, predictions, metric_fn, num_samples=1000, alpha=0.05):
    """
    Calculates a bootstrap confidence interval for a given metric.

    Args:
        true_values: True values of the target variable.
        predictions: Predicted values.
        metric: A function that takes true_values and predictions as input and returns a scalar metric.
        num_samples: Number of bootstrap samples to generate.
        alpha: Significance level for the confidence interval.

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """

    n = len(true_values)
    values = []
    for _ in range(num_samples):
        indices = np.random.randint(0, n, n)
        bootstrap_true = true_values[indices]
        bootstrap_pred = predictions[indices]
        value = metric_fn(bootstrap_true, bootstrap_pred)
        values.append(cast_float(value))
    lower_bound = np.percentile(values, alpha / 2 * 100)
    upper_bound = np.percentile(values, (1 - alpha / 2) * 100)

    return lower_bound, upper_bound, values

class RootMeanSquaredError(torchmetrics.MeanSquaredError):
    def __init__(self):
        super().__init__(squared=False)

if __name__ == '__main__':
    metrics = torchmetrics.MetricCollection([
        torchmetrics.PearsonCorrCoef(),
        RootMeanSquaredError(),
        torchmetrics.KendallRankCorrCoef(),
        # torchmetrics.MeanAbsoluteError(),
        # torchmetrics.R2Score(),
    ])


    all_df = []
    samples_df = []
    model_order = ['XGBoost\nrdkit', 
        'XGBoost\nPOM embed',
        'CheMix\nPOM embed',
        'DreaMix\nTop model',
        'DreaMix\nEnsemble']

    for filename, tag in zip(
        ['../scripts_baseline/xgb_rdkit/leaderboard_predictions.csv',
        '../scripts_baseline/xgb_embed/leaderboard_predictions.csv',
        '../scripts_chemix/results/chemix_frozenpom/leaderboard_predictions.csv',
        '../scripts_chemix/results/chemix_ensemble/top1/leaderboard_predictions.csv',
        '../scripts_chemix/results/chemix_ensemble_comb/ensemble_leaderboard_predictions.csv'], 
        model_order
    ):
        df = pd.read_csv(filename)
        y_true = torch.from_numpy(df['Predicted_Experimental_Values'].to_numpy(np.float32))
        y_pred = torch.from_numpy(df['Ground_Truth'].to_numpy(np.float32))

        results = []
        samples = {}
        for key, metric_fn in metrics.items():
            low, up, sample = bootstrap_ci(y_pred, y_true, metric_fn, num_samples=1000, alpha=0.05)
            info = {'metric': key, 'low_ci': low, 'upper_ci': up,
                            'mean': metric_fn(y_pred, y_true).item()
                            }
            samples[key] = sample
            results.append(info)

        samples = pd.DataFrame(samples)
        samples['fname'] = tag
        samples_df.append(samples)

        results_df = pd.DataFrame(results) #.set_index('metric')
        results_df['fname'] = tag
        all_df.append(results_df)

    all_df = pd.concat(all_df)
    samples_df = pd.concat(samples_df)

    metrics = all_df['metric'].unique()

    # loop through and test significance
    comparisons = []
    for i, gdf in samples_df.groupby('fname'):
        for j, gdf2 in samples_df.groupby('fname'):
            if i == j:
                continue
            for m in metrics:
                t_ind, p_val = wilcoxon(gdf[m], gdf2[m])
                comparisons.append({'metric': m, 'A': i, 'B': j, 't_ind': t_ind, 'p_val': p_val})
            
    comparisons = pd.DataFrame(comparisons)
    print('The following are NOT statistically significantly different...')
    print(comparisons[comparisons['p_val'] > 0.05].head())

    print()
    print('Here are the statistics... ')
    print(all_df)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(30, 10), sharey=True)
    axs = axs.flatten()

    # List of metrics
    metric_name = {
        'PearsonCorrCoef':'Pearson (↑)',
        'KendallRankCorrCoef':'Kendall (↑)',
        'RootMeanSquaredError':'RMSE (↓)',
    }

    # Plot each metric
    for i, (metric, c) in enumerate(zip(metrics, ['Blues', "pink_r", 'Oranges'])):
        # Filter data for the current metric
        metric_data = all_df[all_df['metric'] == metric]
        palette = sns.color_palette(c, n_colors=len(model_order))

        # Plot the data
        sns.barplot(y='fname', x='mean', hue='fname', data=metric_data, ax=axs[i], order=model_order, orient='h', palette=palette, legend=False)
        (_, caps, _) = axs[i].errorbar(y=range(len(metric_data)), x=metric_data['mean'], 
                        xerr=[metric_data['mean'] - metric_data['low_ci'], 
                            metric_data['upper_ci'] - metric_data['mean']],
                        fmt='none', c='black')
        for cap in caps:
            cap.set_markeredgewidth(5)
        
        # axs[i].locator_params(axis='x', nbins=7)
        axs[i].set_ylabel('Model')
        axs[i].set_xlabel(f'{metric_name[metric]}')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig('baseline_models.svg', format='svg')
    plt.savefig('baseline_models.png')
