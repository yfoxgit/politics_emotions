import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from typing import Optional, Any


# load Gennaro & Ash (2022) pre-trained model & assets
def load_assets(model_dir: str) -> tuple[
        Word2Vec,
        dict[str, float],
        np.ndarray,
        np.ndarray]:
    '''
    Load model, frequencies and centroids from Gennaro & Ash (2022).
    Parameters: model_dir (Path): Path to models directory
    Returns: tuple
    '''
    w2v = Word2Vec.load(f'{model_dir}/w2v-vectors_8_300.pkl', mmap='r')
    ga_freqs = joblib.load(f'{model_dir}/word_freqs.pkl')
    affect_centroid = joblib.load(
        f'{model_dir}/affect_centroid.pkl').reshape(-1)
    cog_centroid = joblib.load(f'{model_dir}/cog_centroid.pkl').reshape(-1)
    return w2v, ga_freqs, affect_centroid, cog_centroid


# vectorization
def vectorize(
        tokens: list[str],
        w2v: Word2Vec,
        freq: dict[str, float]
        ) -> Any:
    '''
    Computes SIF-weighted mean vector for list of tokens.
    Replicates Gennaro & Ash (2022) using their pre-trained Word2Vec model.
    Skips tokens not in vocabulary.

    Parameters:
        tokens (list[str]): list of tokens
        w2v: pre-trained word2vec model
        freq (dict[str, float]): SIF frequencies

    Returns:
        np.ndarray of shape (300, ) or None if no tokens in vocabulary
    '''
    vecs = [
        w2v.wv[w] * freq.get(w, 0)
        for w in tokens
        if w in w2v.wv and freq.get(w, 0) > 0
    ]
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    return v.reshape(-1)


# calculate emotionality score
def emotionalize(vector: Optional[np.ndarray],
                 affect_centroid: np.ndarray,
                 cog_centroid: np.ndarray
                 ) -> dict[str, float]:
    '''
    Computes emotionality score for a document/sentence vector.
    Replicates Gennaro & Ash (2022) formula:
        score = (1 + 1 - affect_distance) / (1 + 1 - cognition_distance)

    Parameters:
        vector (np.ndarray): SIF-weighted mean vector
        affect_centroid (np.ndarray): Pre-trained affect pole vector
        cog_centroid (np.ndarray): Pre-trained cognition pole vector
    Returns:
        dict[str, float]: with keys affect_d, cognition_d, score
    '''
    if vector is None:
        return {'affect_d': np.nan, 'cognition_d': np.nan, 'score': np.nan}

    a = cosine(vector, affect_centroid)
    c = cosine(vector, cog_centroid)
    score = (1 + 1 - a) / (1 + 1 - c)

    return {'affect_d': a, 'cognition_d': c, 'score': score}


# sentence level scoring
def score_sentences(df: pd.DataFrame,
                    w2v: Word2Vec,
                    freq: dict[str, float],
                    affect_centroid: np.ndarray,
                    cog_centroid: np.ndarray,
                    tokens_col: str = 'tokens'
                    ) -> pd.DataFrame:
    '''
    Calculates emotionality score at sentence level.
    Replicates Gennaro & Ash (2022) using their pre-trained Word2Vec model.
    Score every sentence in the dataframe individually.
    Each row (sentence) is assigned affect_d, cognition_d, and score.

    Parameters:
        df (pd.DataFrame): Dataframe with sentence level tokens column
        w2v (Word2Vec): Pre-trained word2vec model
        freq (dict[str, float]): SIF frequency weights
        affect_centroid (np.ndarray): Pre-trained affect pole vector
        cog_centroid (np.ndarray): Pre-trained cognition pole vector
        tokens_col (str): tokens column name, default 'tokens'
    Returns: pd.DataFrame:
        Input dataframe with affect_d, cognition_d, score appended
    '''
    df = df.copy()

    # vectorize and score each sentence
    vectors = df[tokens_col].apply(
        lambda t: vectorize(
            t, w2v, freq)
    )
    scores = vectors.apply(
        lambda v: emotionalize(
            v, affect_centroid, cog_centroid)
    )

    df = pd.concat([df, pd.DataFrame(scores.tolist())], axis=1)

    # print error if null vector produced
    n_null = df['score'].isna().sum()
    if n_null > 0:
        print(f"Warning: {n_null} sentences ({n_null/len(df):.1%}) "
              f"produced no vector - no tokens found in w2v vocabulary")

    return df


# document level scoring
def score_documents(df: pd.DataFrame,
                    w2v: Word2Vec,
                    freq: dict[str, float],
                    affect_centroid: np.ndarray,
                    cog_centroid: np.ndarray,
                    tokens_col: str = 'tokens'
                    ) -> pd.DataFrame:
    '''
    Calculates emotionality score at the document level.
    Aggregates sentence tokens to documents before calculating.
    Replicates Gennaro & Ash (2022) using their pre-trained Word2Vec model.
    Documents identified by department + year + author_type.

    Parameters:
        df (pd.DataFrame): Dataframe with sentence level tokens column
        w2v (Word2Vec): Pre-trained word2vec model
        freq (dict[str, float]): SIF frequency weights
        affect_centroid (np.ndarray): Pre-trained affect pole vector
        cog_centroid (np.ndarray): Pre-trained cognition pole vector
        tokens_col (str): tokens column name, default 'tokens'
    Returns: pd.DataFrame:
        document level dataframe with affect_d, cognition_d, score appended
    '''
    doc_keys = ['department', 'year', 'author_type']

    # aggregate sentences to document level
    # sum in line with bag-of-words approach in G&A(2022)
    doc_df = (
        df.groupby(doc_keys)[tokens_col]
        .sum()
        .reset_index()
    )

    # vectorize and score each document
    vectors = doc_df[tokens_col].apply(
        lambda t: vectorize(
            t, w2v, freq)
    )
    scores = vectors.apply(
        lambda v: emotionalize(
            v, affect_centroid, cog_centroid)
    )

    doc_df = pd.concat([doc_df, pd.DataFrame(scores.tolist())], axis=1)

    # drop aggregated tokens
    doc_df = doc_df.drop(columns=tokens_col)

    # print error if null vector produced
    n_null = doc_df['score'].isna().sum()
    if n_null > 0:
        print(f"Warning: {n_null} documents ({n_null/len(doc_df):.1%}) "
              f"produced no vector - no tokens found in w2v vocabulary")

    return doc_df
