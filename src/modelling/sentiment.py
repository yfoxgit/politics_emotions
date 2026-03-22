import anthropic
import pandas as pd
from dotenv import load_dotenv
from typing import Any


# to load in .env into environment variables
load_dotenv()

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment


# use Claude LLM to rate emotionality of sentences
def rate_emotionality(input_text: str) -> dict[str, Any]:
    '''
    Uses Claude Sonnect 4.6 to rate the emotionality of strings.
    Asks to rate on a 1-10 scale, where 1 is purely factual
    and 10 is highly emotional, affective language.
    Requires ANTHROPIC_API_KEY for code to run.
    Rates a 0 when the text is not grammatically correct.

    Parameters:
    input_text (str): Text to rate.

    Returns:
    dict[str, Any]:
        the SCORE (a number 1-10),
        and a 1 sentence explanation for the rating
    '''

    prompt = f'''
    Rate the emotionality of the following text
    on a scale from 1 to 10, where:
    1 = purely factual, technical, or administrative
    language with no emotional content,
    5 = equally balanced in factual and emotional language, and
    10 = highly emotional, charged, or affective language.
    Rate the text a 0 if the text is
    not grammatically correct or is otherwise erroneous.

    Text: "{input_text}"

    Respond in this exact format:
    SCORE: [number 0-10]
    REASONING: [one sentence explanation]
    '''

    system_prompt = '''
        You are a neutral linguistic researcher rating
        the emotional content of text.

        Emotionality refers to the degree to which language
        appeals to affect and feeling rather than cognition and reason.
        A highly emotional text uses affective, value-laden,
        or evocative language. A cognitive text uses
        factual, technical, or analytical language.

        You rate text purely on its linguistic features, applying
        consistent and objective reasoning throughout and are not
        influenced by the content or ideology expressed in
        the text.'''

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    lines = text.strip().split('\n')

    score_line = next(line for line in lines if line.startswith('SCORE:'))
    reason_line = next(line for line in lines if line.startswith('REASONING:'))

    return {
        'claude_score': float(score_line.replace('SCORE:', '').strip()),
        'claude_reasoning': reason_line.replace('REASONING:', '').strip()
    }


# applies Claude to dataframe
def rate_df(df: pd.DataFrame,
            text_col: str = 'text'
            ) -> pd.DataFrame:
    '''
    Uses Claude to rate emotionality for text in a dataframe.
    Can apply to sentences or document level text.

    Parameters:
        df (pd.DataFrame): dataframe containing untokenized sentences
        text_col: column containing raw sentence text, default 'text'
    Returns:
        pd.DataFrame with claude_score and claude_reasoning appended
    '''
    df = df.copy()
    n = len(df)
    ratings = []

    for i, entry in enumerate(df[text_col], start=1):
        ratings.append(rate_emotionality(entry))
        print(f'\rRating entires: {i}/{n}', end='', flush=True)
    print()

    df = pd.concat([df, pd.DataFrame(ratings)], axis=1)

    return df
