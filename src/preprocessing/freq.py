from collections import Counter
import pandas as pd


# count absolute frequencies of tokens
def count_words(df: pd.DataFrame,
                tokens_col: str = 'tokens'
                ) -> Counter[str]:
    '''
    Replicates Gennaro & Ash (2022) frequency calculation.
    Modified to my dataframe format.

    Parameters:
        df (pd.DataFrame): Input dataframe
        tokens_col (str): Column name for tokens, default 'tokens'
    Returns:
        Counter[str]: Token counts.
    '''
    all_tokens = []

    # flatten tokens into 1 list
    for tokens in df[tokens_col]:
        for t in tokens:
            all_tokens.append(t)

    # compute word counts
    counts = Counter(all_tokens)

    return counts


# calculate SIF frequencies of tokens
def calc_SIF(freqs: Counter[str],
             a: float = 0.001
             ) -> dict[str, float]:
    '''
    Replicates Gennaro & Ash (2022) SIF frequency calculation.
    Modified to the format of my data.

    Parameters:
        freqs (Counter): Token absolute frequencies
        a (float): Smoothing parameter, default 0.001 as in G&A(2022)
    Returns:
        dict[str, int]: SIF frequencies of tokens.
    '''
    total = sum(freqs.values())

    SIF_freqs = {}
    for key in freqs.keys():
        SIF_freqs[key] = a / (a + (freqs[key] / total))

    return SIF_freqs


# count absolute frequencies of affect, cognition words
def count_dict_words(
        counts: Counter[str],
        affect: list[str],
        cognition: list[str]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Replicates Gennaro & Ash (2022) dictionary counts.
    Collects absolute frequencies of affect (emotion)
    and cognition (reason) words in the corpus.
    Uses predefined affect and cognition word lists.

    Parameters:
        counts (Counter): token counts
        affect (list): pre-defined affect token list
        cognition (list): pre-defined cognition token list

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            affect_df: counts of affect words in corpus
            cog_df: counts of cognition words in corpus

    '''
    # create affect dataframe
    affect_df = (pd.DataFrame({
        'word': affect,
        'count': [counts[t] for t in affect]
    })).sort_values('count', ascending=False).reset_index(drop=True)

    # create cognition dataframe
    cog_df = (pd.DataFrame({
        'word': cognition,
        'count': [counts[t] for t in cognition]
    })).sort_values('count', ascending=False).reset_index(drop=True)

    return affect_df, cog_df
