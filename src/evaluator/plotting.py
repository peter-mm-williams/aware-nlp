import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
from typing import List, Literal, Optional, Tuple
from src.util.string_formatting import wrap_text


def make_barplot_subplot(
    gs: gridspec.GridSpec,
    retrieval_size: int,
    similarity_metric: Literal['cosine', 'euclidian', 'dot'],
    evaluation_metric: Literal['f1', 'recall', 'precision'],
    summary_df: pd.DataFrame,
    color_pallette: sns.color_palette
):
    # Filter summary_df to make plot_df
    filter_logic = (
        (summary_df.retrieval_size == retrieval_size) &
        (summary_df.similarity == similarity_metric)
    )
    plot_df = summary_df[filter_logic].copy()
    ax0 = plt.subplot(gs[0])  # This will be the larger plot on the left
    sns.set_theme(style="whitegrid")
    sns.barplot(x="encoder", y=evaluation_metric, hue="consensus_threshold",
                data=plot_df, palette=color_pallette, ax=ax0)
    ax0.set_xlabel('Encoder Model', fontsize=30)
    ax0.set_ylabel(f'{evaluation_metric.capitalize()} Score', fontsize=30)
    ax0.set_title(f'{retrieval_size} Documents Retrieved', fontsize=30)
    ax0.tick_params(axis='x', labelsize=17)  # Adjust x-axis tick labels
    # Moving the legend of the first plot outside
    ax0.legend(title='Consensus Threshold',
               bbox_to_anchor=(1.05, 1), loc='upper left')


def make_swarmplot(
    fig: plt.figure,
    sample_df: pd.DataFrame,
    consensus_thresholds: List[int],
    color_pallete: sns.color_palette,
    axes_definition: List[float] = [0., 0., 1., 1.],
    display_questions: bool = True,
    marker_size: float = 6
):
    # We place it manually with respect to the first plot for more control
    # Adjust left, bottom, width, and height values as needed for positioning and size
    # x, y, width, height relative to figure size
    ax1 = fig.add_axes(axes_definition)

    sample_df['Question'] = sample_df['question'].apply(wrap_text)

    sns.swarmplot(x='Question', y='average_label', data=sample_df,
                  edgecolor='black', color='k', s=marker_size, ax=ax1)
    ax1.set_ylabel('Average Human Label', fontsize=20)
    # Assume at least one sample is uniformly accepted
    n_evaluators = sample_df.label_sum.max()
    for idx, threshold in enumerate(consensus_thresholds):
        ax1.axhline(y=threshold/n_evaluators,
                    color=color_pallete[idx], linewidth=2, linestyle='--')
    ax1.set_xlabel('Question', fontsize=20)

    if not display_questions:
        ax1.tick_params(labelbottom=False)  # Hide x-axis labels


def plot_performance(
    retrieval_size: int,
    similarity_metric: Literal['cosine', 'euclidian', 'dot'],
    evaluation_metric: Literal['f1', 'recall', 'precision'],
    summary_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (20, 6)
):
    # reset index on summary_df to filter across columns
    summary_df = summary_df.reset_index()

    # Setup the figure and gridspec
    fig = plt.figure(figsize=figsize)  # Adjust overall figure size as needed
    # Define a gridspec of 1 row and 2 columns
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    # Define color pallete
    color_palette = sns.color_palette("dark")
    color_palette.pop(1)
    color_palette.pop(1)

    consensus_thresholds = summary_df.consensus_threshold.unique()

    # Make subplots
    make_barplot_subplot(gs, retrieval_size, similarity_metric,
                         evaluation_metric, summary_df, color_palette)
    make_swarmplot(
        fig,
        sample_df,
        consensus_thresholds,
        color_palette,
        axes_definition=[0.7, 0.1, 0.3, 0.6],
        display_questions=False
    )

    # Save plot if path is provided
    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path)
