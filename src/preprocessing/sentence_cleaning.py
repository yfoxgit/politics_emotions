import nltk
import gensim
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as nltk_stopwords


# download required NLTK data (run once)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialise tools
tagger = nltk.PerceptronTagger()
stemmer = SnowballStemmer("english")
stopwords = set(nltk_stopwords.words('english'))


# sentence cleaning function
def clean_sentence(text: str) -> list[str]:
    '''
    Cleans sentences replicating steps in  Gennaro and Ash (2022).
    Uses NLTK list of stopwords instead of their custom-built list.

    Parameters:
    text (str): the sentence to be cleaned

    Returns:
    list[str]: list of processed words
    '''

    # tokenize, lowercase
    tokens = gensim.utils.simple_preprocess(text)

    # drop digits
    tokens = [t for t in tokens if not t.isdigit()]

    # drop short tokens (2 chars or fewer)
    tokens = [t for t in tokens if len(t) > 2]

    # tag POS, keep only Nouns (N), Verbs (V), Adjectives (J)
    tagged = tagger.tag(tokens)
    tokens = [word for word, pos in tagged if pos.startswith(('N', 'V', 'J'))]

    # stem
    tokens = [stemmer.stem(t) for t in tokens]

    # remove stopwords
    tokens = [t for t in tokens if t not in stopwords]

    return list(tokens)
