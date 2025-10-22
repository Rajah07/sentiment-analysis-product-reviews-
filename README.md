# Sentiment Analysis of Amazon Product Reviews

This project conducts a comprehensive sentiment analysis on a large dataset of over 568K Amazon product reviews. The primary goal is to compare the effectiveness of different Natural Language Processing (NLP) techniques, from traditional lexicon-based methods to advanced pre-trained transformer models, in accurately classifying the sentiment of customer reviews.

Due to computational limitations, this notebook performs a detailed analysis on a sample of the first 500 reviews to demonstrate the methodology.

## Tech Stack

  * **Data Manipulation:** `pandas`, `numpy`
  * **Data Visualization:** `matplotlib`, `seaborn`
  * **NLP (Lexicon-based):** `nltk` (Natural Language Toolkit)
  * **NLP (Transformer-based):** `transformers` (by Hugging Face), `scipy`

-----

## Project Workflow

### 1\. Data Loading and Exploratory Data Analysis (EDA)

The project begins by loading the `Reviews.csv` dataset, which contains 568,454 reviews. A sample of 500 reviews is taken for this analysis.

An initial analysis of the sample shows the distribution of star ratings, which is heavily skewed towards 5-star reviews.

### 2\. NLTK Preprocessing

Basic NLP preprocessing steps are performed using NLTK on an example review, including:

  * **Tokenization:** Splitting the text into individual words (tokens).
  * **Part-of-Speech (POS) Tagging:** Identifying the grammatical parts of speech for each token (e.g., noun, verb, adjective).

### 3\. Model 1 - VADER Sentiment Scoring

The first model used is **VADER** (`Valence Aware Dictionary and sEntiment Reasoner`), a lexicon and rule-based sentiment analysis tool from NLTK.

  * VADER analyzes text and returns four scores: `positive`, `neutral`, `negative`, and a `compound` score (a normalized score from -1, most negative, to +1, most positive).

  * The compound score generally correlates with the star rating, as expected. 5-star reviews have a high average compound score, while 1-star reviews have a negative average.

  * Breaking this down further, 5-star reviews show a high positive score, while 1-star reviews show a notable negative score.

### 4\. Model 2 - RoBERTa (Transformer Model)

The second model is a pre-trained **RoBERTa** model from Hugging Face (`cardiffnlp/twitter-roberta-base-sentiment`). This is a powerful transformer model that has been fine-tuned for sentiment analysis and understands the context of words in a sentence.

  * The model returns scores for `positive`, `neutral`, and `negative` sentiments, which are then normalized using the `softmax` function.
  * This process was run on all 500 sample reviews.

### 5\. Model Comparison

A pairplot was generated to visually compare the sentiment scores from both VADER and RoBERTa, categorized by the actual star rating of the review.

This visualization reveals that while both models generally agree, RoBERTa provides a much clearer distinction between positive, neutral, and negative sentiments, especially when compared to VADER's tendency to score many reviews as neutral.

-----

## Key Findings & Challenges

A key part of the analysis was reviewing examples where the model scores and star ratings disagreed. This highlights the challenges of sarcasm, nuance, and mixed sentiments in text.

#### Positive 1-Star Review (Sarcasm/Context)

The models identified text with positive language in 1-star reviews.

  * **Example:** `"Product arrived labeled as Jumbo Salted Peanuts...the peanuts were...small...not salted...This is a confection that has been around a few centuries...I paid $3.99 for this drink. I could have just drunk a cup of coffee and saved my money."`
  * **Analysis:** Both models incorrectly flagged parts of this text as positive, failing to capture the user's overall disappointment and sarcasm.

#### Negative 5-Star Review (Mixed Sentiment)

The models found negative language in 5-star reviews, often indicating a mixed but ultimately positive opinion.

  * **Example:** `"this was sooooo deliscious but too bad i ate em too fast and gained 2 pds! my fault"`
  * **Analysis:** Both VADER and RoBERTa heavily flagged the negative words ("too bad," "fault"), resulting in a high negative score. They missed the context that the user loved the product, and the "negative" aspect was a humorous complaint about its own lack of self-control.

-----

## Extra: Transformers Pipeline

Finally, the notebook demonstrates a simple implementation of the Hugging Face `pipeline` for quick and easy sentiment analysis using the default `distilbert-base-uncased-finetuned-sst-2-english` model.

```python
from transformers import pipeline
sent_pipeline = pipeline("sentiment-analysis")
sent_pipeline('I love sentiment analysis!')
# Output: [{'label': 'POSITIVE', 'score': 0.99978...}]
```

    ```
4.  Ensure you have the `Reviews.csv` file in the same directory.
5.  Run the `Sentiment Analysis Python.ipynb` notebook in a Jupyter environment.
