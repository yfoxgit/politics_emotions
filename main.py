import joblib
import pandas as pd

from preprocessing.load_pdf import get_repo_root as grr
from preprocessing.load_pdf import extract_pdf_text as ept
from preprocessing.cleaning import (merge_authors,
                                    select_years,
                                    clean_df,
                                    agg_to_docs)
from preprocessing.sentence_cleaning import clean_sentence
from preprocessing.freq import (count_words,
                                calc_SIF,
                                count_dict_words)

from modelling.emotionality import (load_assets,
                                    score_sentences,
                                    score_documents)
# from modelling.sentiment import rate_df
from modelling.evaluation import (t_test_scores, t_test_author_time)
from modelling.evaluation import (compare_scores)


# get working directories and data folder paths
root_dir = grr()
raw_dir = root_dir / "data" / "raw"
interim_dir = root_dir / "data" / "interim"
model_dir = root_dir / "models"
print("Working directories defined.")

# initial pdf text extraction
reports = ept(str(raw_dir))
print('Reports extracted.')

# save as parquet as interim data
# uncomment line if want saved
# reports.to_parquet("data/interim/df.parquet")

# select authors and years
reports = merge_authors(reports)
print("Author types merged to 'pol' = politician and 'cs' = civil servant.")

YEARS = [2023, 2025]
reports = select_years(reports, YEARS)
print("Years selected: 2023 and 2025.")

# clean text
# create sentence and document level dataframes
sen_df = clean_df(reports)
doc_df = agg_to_docs(sen_df).reset_index()
print("Sentence and document level DataFrames created.")

# tokenize sentences
sen_df['tokens'] = sen_df['text'].apply(clean_sentence)

# drop rows where cleaning left an empty token list
sen_df = sen_df[
    sen_df['tokens'].map(len) > 1].reset_index(drop=True)
print("Sentences tokenized.")

# save tokenized sentences
# sen_df.to_parquet("data/processed/tokens.parquet")

# calculate my own frequencies
freqs = count_words(sen_df)
SIF_freqs = calc_SIF(freqs)
print("\nFrequencies from forewords calculated.")

# calculate affect, cognition counts
# uses Gennaro & Ash (2022) affect, cognition dictionaries
affect = joblib.load('models/dictionary_affect.pkl')
cognition = joblib.load('models/dictionary_cognition.pkl')
affect_df, cog_df = count_dict_words(freqs, affect, cognition)
print("Gennaro & Ash (2022) affect and cognition dictionaries loaded in.")

# check affect, cognition coverage
affect_found = (affect_df['count'] > 0).sum()
cog_found = (cog_df['count'] > 0).sum()
print(f"Affect words found in forewords:    {affect_found}/{len(affect)} "
      f"({affect_found/len(affect):.1%})")
print(f"Cognition words found in forewords: {cog_found}/{len(cognition)} "
      f"({cog_found/len(cognition):.1%})")

# modelling
w2v, ga_freqs, affect_centroid, cog_centroid = load_assets(str(model_dir))
print("\nGennaro & Ash (2022) model loaded in.")
print('''
      Note that the warning comes from the G&A model being trained
      on an older version of Numpy compared to the version in this
      project, but does not cause any issues for the analysis.
      ''')

# calculate emotionality scores at sentence and document level
# use G&A (2022) frequencies
sen_scores = score_sentences(sen_df,
                             w2v,
                             ga_freqs,
                             affect_centroid,
                             cog_centroid)
doc_scores = score_documents(sen_df,
                             w2v,
                             ga_freqs,
                             affect_centroid,
                             cog_centroid)
print("\n\nScores calculated using frequencies from Gennaro & Ash (2022).")

# document level t-test
print('''\nT-test politician vs civil servant emotionality scores
      at the Document Level using Gennaro & Ash (2022) frequencies.''')
t_test_scores(doc_scores)
print('''\nT-test politician vs civil servant emotionality scores
      at the Sentence Level using Gennaro & Ash (2022) frequencies.''')
t_test_scores(sen_scores)

# compare political parties
print('\nT-test politicians in 2023 vs 2025.')
t_test_author_time(doc_scores, 'pol')
print('\nT-test civil servants in 2023 vs 2025.')
t_test_author_time(doc_scores, 'cs')

# calculate emotionality scores at sentence and document level
# use my own frequencies
sen_scores_own_freqs = score_sentences(sen_df,
                                       w2v,
                                       SIF_freqs,
                                       affect_centroid,
                                       cog_centroid)
doc_scores_own_freqs = score_documents(sen_df,
                                       w2v,
                                       SIF_freqs,
                                       affect_centroid,
                                       cog_centroid)
print("\n\nScores calculated using frequencies from the forewords.")

# document level t-test
# using my own frequencies
print('''\nT-test politician vs civil servant emotionality scores
      at the Document Level using foreword frequencies.''')
t_test_scores(doc_scores_own_freqs)
print('''\nT-test politician vs civil servant emotionality scores
      at the Sentence Level using foreword frequencies.''')
t_test_scores(sen_scores_own_freqs)

# compare political parties
# using my own frequencies
print('\nT-test politicians in 2023 vs 2025 using foreword frequencies.')
t_test_author_time(doc_scores_own_freqs, 'pol')
print('\nT-test civil servants in 2023 vs 2025 using foreword frequencies.')
t_test_author_time(doc_scores_own_freqs, 'cs')

# save as files
# uncomment if want to save files
'''
doc_scores.to_parquet("results/document_scores.parquet")
doc_scores_own_freqs.to_parquet("results/document_scores_own_freqs.parquet")
sen_scores.to_parquet("results/sentence_scores.parquet")
sen_scores_own_freqs.to_parquet("results/sentence_scores_own_freqs.parquet")
'''

# run Claude sentiment analysis
# only runs if you have a credited ANTHROPIC_API_KEY
# otherwise use pre-saved ratings loaded in line 171
# uncomment package imports in line 18
# uncomment code by deleting ''' in lines 164 & 168
'''
doc_ratings = rate_df(doc_df)
print("Document ratings complete.")
doc_ratings.to_parquet("results/doc_ratings.parquet")
'''

# load in pre-saved ratings
doc_ratings = pd.read_parquet('results/doc_ratings.parquet',
                              engine='fastparquet')
print('\n\nClaude scores loaded in.')

# conduct tests
print('\nT-test politician vs civil servant Claude scores.')
t_test_scores(doc_ratings, scores_col='claude_score')
print('\nT-test Claude scores for politicians in 2023 vs 2025.')
t_test_author_time(doc_ratings,
                   author_type='pol',
                   scores_col='claude_score',
                   year_col='year')
print('\nT-test Claude scores for civil servants in 2023 vs 2025.')
t_test_author_time(doc_ratings,
                   author_type='cs',
                   scores_col='claude_score',
                   year_col='year')

# compare emotionality vs Claude scores
print('\nScore comparison between G&A (2022) and Claude.')
compare_scores(doc_scores, doc_ratings)

# analysis complete
print('\nAnalysis complete.')
