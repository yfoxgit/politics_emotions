import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


# grouped histogram for years, author types
def plot_years_authors(
        df: pd.DataFrame,
        year_col: str = 'year',
        author_col: str = 'author_type',
        save_plot: bool = False,
        file_path: Optional[str] = None
        ) -> None:
    '''
    Plots bar chart for author types, broken down by year.

    Parameters:
    df (pd.DataFrame):
        The DataFrame containing the data.
    year_col (str):
        Year column name, default 'year'.
    author_col (str):
        Author type column name, default 'author_type'.
    save_plot (bool, default=False):
        True saves the plot.
    file_path (str, Optional):
        Optional filepath to save the plot at.
    '''

    counts = df.groupby([year_col, author_col]).size().unstack(fill_value=0)
    years = sorted(counts.index)
    x = np.arange(len(counts.columns))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, year in enumerate(years):
        ax.bar(x + i * w, counts.loc[year], w, label=str(year))

    ax.set_xticks(x + w)
    ax.set_xticklabels(counts.columns)
    ax.set_xlabel('Author Type')
    ax.set_ylabel('Count')
    ax.set_title('Author Type Distribution by Year', fontweight='bold')
    ax.legend(title='Year')

    plt.tight_layout()

    if save_plot:
        if file_path is None:
            file_path = 'Grouped_Bar_Years_Author_Types.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f'Saved plot to {file_path}')

    plt.show()
