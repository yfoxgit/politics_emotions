import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize


# NLTK model for sentence splitting
nltk.download('punkt_tab', quiet=True)


# function to merge author types
def merge_authors(
        df: pd.DataFrame,
        author_col: str = 'author_type'
        ) -> pd.DataFrame:
    '''
    Merges author types into 'pol' (politician) and 'cs' (civil servant).
    'pol' joins 'min' (minister) and 'sos' (Secretary of State).
    'cs' joins 'ps', 'ps1' and 'ps2', accounting for multiple statements
    by Permanent Secretaries in report.
    'joint' forewords (written by ministers and civil servants together)
    are dropped from the data.

    Parameters:
        df (pd.DataFrame): Input dataframe
        author_col (str): Column name for author type, default 'author_type'

    Returns:
        pd.DataFrame: DataFrame with merged author types.
    '''

    mapping = {
        'min': 'pol',
        'sos': 'pol',
        'ps': 'cs',
        'ps1': 'cs',
        'ps2': 'cs'
        # 'joint' is dropped
    }

    df = df[df[author_col] != 'joint'].copy()
    df[author_col] = df[author_col].map(mapping)

    return df


# function to drop years (2024)
def select_years(
        df: pd.DataFrame,
        years: list[int]
        ) -> pd.DataFrame:
    '''
    Reduces DataFrame to only select years.

    Parameters:
    df (pd.DataFrame): The DataFrame to be reduced.
    years (list[int]): The list of years to be kept.

    Returns:
    pd.DataFrame: The reduced DataFrame.
    '''
    return df[df['year'].isin(years)]


# function to detect noise in text
def is_noise(lines: list[str], i: int) -> bool:
    '''
    Returns True if a line is likely a header, footer, title,
    or page number rather than real sentence content.

    Parameters:
    line (str): line string to be analysed

    Returns:
    bool: True or false per line
    '''
    line = lines[i].strip()

    if not line:
        return True

    # Page numbers:
    # Lines containing only digits
    if re.fullmatch(r'\d+', line):
        return True
    # "Page X"
    if re.match(r'^(page\s*)?\d+\s*$', line, re.IGNORECASE):
        return True

    # Lines with pipe separators e.g. "6 | Title"
    if re.match(r'^\d+\s*\|', line):
        return True

    # Very short lines and no sentence-ending punctuation
    words = line.split()
    if len(words) <= 3 and not re.search(r'[.!?:;]$', line):
        return True

    # Lines that are ALL CAPS but greater than 1 word
    if line.isupper() and len(words) > 1:
        return True

    # Lines ending with a year or date eg "CO Annual report 2024-2025"
    if re.search(r'\b(19|20)\d{2}[-–]\d{2,4}\s*$', line):
        return True

    # Short lines surrounded by whitespace
    prev_blank = (i == 0) or not lines[i-1].strip()
    next_blank = (i == len(lines)-1) or not lines[i+1].strip()

    if len(words) <= 6 and prev_blank and next_blank:
        return True

    return False


# clean text
def clean_text(text: str) -> list[str]:
    '''
    Takes a raw text string, removes noise lines,
    cleans whitespace, and returns a list of sentences.

    Parameters:
    text (str): text to be analysed

    Returns:
    list[str]: list of clean sentences
    '''
    # remove BOM and other non-standard characters
    text = text.lstrip('\ufeff\x07')

    # split into lines, filter out noise
    lines = text.split('\n')
    clean_lines = [
        line.strip()
        for i, line in enumerate(lines)
        if not is_noise(lines, i)
    ]

    # rejoin to a single string
    merged = ' '.join(clean_lines)

    # remove duplicate spaces
    merged = re.sub(r' {2,}', ' ', merged).strip()

    # replace non-breaking spaces '\xa0'
    merged = merged.replace('\xa0', ' ')

    # tokenize using NLTK function
    sentences = sent_tokenize(merged)

    # filter out sentences <=3 words
    sentences = [s for s in sentences if len(s.split()) > 3]

    # strip white space
    sentences = [s.strip() for s in sentences if s.strip()]

    # remove the first 2 and last as buffer against failed cleaning
    sentences = sentences[2:-1]

    return list(sentences)


# process entire dataframe
def clean_df(df: pd.DataFrame,
             text_col: str = 'text') -> pd.DataFrame:
    '''
    Takes a dataframe with a text column,
    segments each text into sentences,
    and returns a new dataframe where each row is one sentence,
    with all original metadata columns preserved.

    Parameters:
        df (pd.DataFrame): Input dataframe
        text_col (str): text column name, default 'text'

    Returns:
        pd.DataFrame: DataFrame with cleaned sentences.
    '''
    records = []  # 1 dict per sentence
    for _, row in df.iterrows():
        sentences = clean_text(row[text_col])
        for i, sentence in enumerate(sentences):
            record = row.to_dict()
            record[text_col] = sentence  # replace raw text with the sentence
            record['sentence_index'] = i  # position within original document
            records.append(record)

    result_df = pd.DataFrame(records).reset_index(drop=True)
    return result_df


# rejoin sentences to document level
def agg_to_docs(df: pd.DataFrame,
                text_col: str = 'text',
                sentence_index_col: str = 'sentence_index'
                ) -> pd.DataFrame:
    '''
    Aggregates sentences back to document level to make
    DataFrame with 1 string per document.
    Uses department, year, author_type and sentence_index
    to identify individual documents.

    Parameters:
        df (pd.DataFrame): sentence level dataframe
        text_col (str): text column name, default 'text'
        sentence_index_col (str): sentenence index column name,
            default 'sentence_index'

    Returns:
        pd.DataFrame: document level
    '''

    groups = ['department', 'year', 'author_type']

    df = df.copy()

    # detect where index resets in each group
    df['_reset'] = (
        df.groupby(groups)[sentence_index_col]
        .transform(lambda x: (x <= x.shift(1).fillna(-1)).cumsum())
        )

    # identify document by department, year, author_type, _reset
    doc_cols = groups + ['_reset']
    df = df.sort_values(groups + [sentence_index_col])

    doc_df = (
        df.groupby(doc_cols, sort=False)
        .agg(
            text=(text_col, ' '.join))  # join sentences with ' '
        .reset_index()
        .drop(columns='_reset')
    )

    return doc_df
