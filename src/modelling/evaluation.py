import pandas as pd
from scipy import stats
from scipy.stats import spearmanr


# t-test cs vs pol scores
def t_test_scores(scores: pd.DataFrame,
                  scores_col: str = 'score'
                  ) -> None:
    '''
    Performs t-test comparing emotionality scores of civil servants
    compared to politicians.
    Can be applied at the sentence or document level, but document
    level is more appropriate as sentences are likely not independent.

    Parameters:
    scores (pd.DataFrame): DataFrame containing emotionality scores.
    scores_col (str): tokens column name, default 'score'
    Returns: (None) prints results
    '''
    cs_scores = scores[scores['author_type'] == 'cs'][scores_col].dropna()
    pol_scores = scores[scores['author_type'] == 'pol'][scores_col].dropna()

    t_stat, p_value = stats.ttest_ind(cs_scores, pol_scores)

    print(f"T-test ({'Student'}):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4f}")
    print(f"  Significant at 5% level: {p_value < 0.05}")


# t-test political parties
def t_test_author_time(scores: pd.DataFrame,
                       author_type: str,
                       scores_col: str = 'score',
                       year_col: str = 'year'
                       ) -> None:
    '''
    Performs t-test comparing emotionality scores of author types
    between years.

    Parameters:
    scores (pd.DataFrame): DataFrame containing emotionality scores.
    scores_col (str): tokens column name, default 'score'
    year_col (str): years column name, default 'year'
    author_type (str): the author type to be tested
    Returns: (None) prints results
    '''
    df = scores[scores['author_type'] == author_type]
    years = sorted(df[year_col].unique())
    splits = {}
    for yr in years:
        splits[yr] = df[df[year_col] == yr]

    t_stat, p_value = stats.ttest_ind(
        splits[years[0]][scores_col].dropna(),
        splits[years[1]][scores_col].dropna())

    print(f"Author type: {author_type} — {years[0]} vs {years[1]}")
    print(f"T-test ({'Student'}):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4f}")
    print(f"  Significant at 5% level: {p_value < 0.05}")


# compare correlation between emotionality and Claude scores
def compare_scores(
        scores_df: pd.DataFrame,
        claude_df: pd.DataFrame
        ) -> None:
    '''
    Aggregates data from emotionality scores and Claude scores.
    Calculates the correlation between scores.
    Aggregates Claude ratings for mean within the same department,
    year and author_type when there are multiple entries.

    Parameters:
        scores_df (pd.DataFrame): document level emotionality scores
        claude_df (pd.DataFrame): document level Claude scores

    Returns: (None) prints results
    '''
    # pre-defined groups
    groups = ['department', 'year', 'author_type']

    # aggregate claude scores dataframe to match scores dataframe
    # use mean for multiple documents with same dept/year/author_type
    claude_agg = (
        claude_df
        .groupby(groups)['claude_score']
        .mean()
        .reset_index()
    )

    # preserve only documents present in both
    merged = scores_df.merge(
        claude_agg,
        on=groups,
        how='inner'
    )

    print(f'Documents in G&A Emotionality Scores:  {len(scores_df)}')
    print(f'Documents in Claude Scores:            {len(claude_agg)}')
    print(f'Documents in both (merged):            {len(merged)}')

    corr, p = spearmanr(merged['score'], merged['claude_score'])
    print(f"\nSpearman correlation: {corr:.4f}")
    print(f"p-value:              {p:.4f}")
    print(f"Significant at 5%:    {p < 0.05}")
