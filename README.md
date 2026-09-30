# News Headline NLP & Sentiment Classification

Built during a Data Science / NLP internship (Mar–Jul 2023). An end-to-end NLP pipeline that turns a large, messy news-headline dataset into a working sentiment classifier.

## What this does

- Preprocesses a large news-headline dataset: deduplication, missing-value handling, source-domain extraction
- Runs tokenization, stopword removal, POS tagging, and named entity recognition (spaCy, NLTK)
- Labels sentiment using VADER (`SentimentIntensityAnalyzer`) and visualizes trends with word clouds
- Trains a multinomial Logistic Regression classifier to predict sentiment (negative / neutral / positive), tuned via `GridSearchCV`

## Results

~80% cross-validated accuracy across 3 sentiment classes.

## Tech stack

Python · spaCy · NLTK · scikit-learn · VADER · pandas

## Notes

Early iterations of this pipeline had a data-leakage bug — duplicate headlines republished across multiple source domains were landing in both the train and test splits, artificially inflating accuracy. Deduplicating by normalized headline text before splitting brought the number down to a more honest ~80%, which is the version reflected here.
