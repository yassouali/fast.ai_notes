### Structured and Time Series Data

There are two types of columns:

* Categorical — It has a number of “levels” e.g. StoreType, Assortment.
* Continuous — It has a number where differences or ratios of that numbers have some kind of meanings e.g. `CompetitionDistance`.

Numbers like `Year` , `Month`, although we could treat them as continuous, we do not have to. If we decide to make `Year` a categorical variable, we are telling our neural net that for every different “level” of `Year` (2000, 2001, 2002), you can treat it totally differently; where-else if we say it is continuous, it has to come up with some kind of smooth function to fit them. So often things that actually are continuous but do not have many distinct levels (e.g. Year, DayOfWeek), it often works better to treat them as categorical.

Choosing categorical vs. continuous variable is a modeling decision we get to make. In summary, if it is categorical in the data, it has to be categorical. If it is continuous in the data, we get to pick whether to make it continuous or categorical in the model. Generally, floating point numbers are hard to make categorical as there are many levels (Cardinality).

#### Joining the data
In the competition, the winners used additionnal data like state names, google trends and weather, so first we need to joint this dataframes with different sizes, this is done using the `merge` method. The `suffixes` argument describes the naming convention for duplicate fields. We've elected to leave the duplicate field names on the left untouched, and append a "\_y" to those on the right.

`join_df` is a function for joining tables on specific fields. By default, we'll be doing a left outer join of `right` on the `left` argument using the given fields for each table.

```python
def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))

weather = join_df(weather, state_names, "file", "StateName")
```

Here we join state_names to the right of weather, using the fields on the right of "StateName" in `state_names` and `weather`.

#### Preprocessing

* Turn all the continuous ones into 32bit floating point for pytorch.
* Pull out the dependent variable it into a separate variable, and deletes it from the original data frame. (the value to be predicted is deleted from `df`, and now we have a single column dataframe with the target value).
* Neural nets like to have the input data to all be somewhere around zero with a standard deviation of somewhere around 1. So we take our data, subtract the mean, and divide by the standard deviation to make that happen. It returns a special object which keeps track of what mean and standard deviation it used for that normalization so we can do the same to the test set later (mapper).
* Hnadling missing values — for categorical variable, it becomes ID: 0 and other categories become 1, 2, 3, and so on. For continuous variable, it replaces the missing value with the median and create a new boolean column that says whether it was missing or not.

Now we have a data frame which does not contain the dependent variable and where everything is a number. Now we're ready to deep learning.

#### The model, Embeddings and training:
As per usual, we will start by creating model data object which has a validation set, training set, and optional test set built into it. From that, we will get a learner, we will then optionally call lr_find, then call learn.fit and so forth.

The continus variables are directly fed in to the network, now for the categorical variables, instead of representing them as one hot vectors, we'll embedd into a lower space, using different emebedding matrices depending on the cardinality of the categories of each column, for example, the categories we have and their cardinality :

```python
[('Store', 1116),  ('DayOfWeek', 8),  ('Year', 4),  ('Month', 13),  ('Day', 32),  ('StateHoliday', 3),  ('CompetitionMonthsOpen', 26),  ('Promo2Weeks', 27),  ('StoreType', 5),  ('Assortment', 4),  ('PromoInterval', 4),  ('CompetitionOpenSinceYear', 24),  ('Promo2SinceYear', 9),  ('State', 13),  ('Week', 53),  ('Events', 22),  ('Promo_fw', 7),  ('Promo_bw', 7),  ('StateHoliday_fw', 4),  ('StateHoliday_bw', 4),  ('SchoolHoliday_fw', 9),  ('SchoolHoliday_bw', 9)]
```

Now depending on the cardinality of each one cat we'll chose an embedding size, with a max of 50:

```python
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]

[(1116, 50),  (8, 4),  (4, 2),  (13, 7),  (32, 16),  (3, 2),  (26, 13),  (27, 14),  (5, 3),  (4, 2),  (4, 2),  (24, 12),  (9, 5),  (13, 7),  (53, 27),  (22, 11),  (7, 4),  (7, 4),  (4, 2),  (4, 2),  (9, 5),  (9, 5)]
```

So now we have for each column a specific embedding matrice, we pass all the values of each column per their corresponding matrice, so our model we'll have nb_cat embeddings, each one with the specific size ((1116, 50) ... (9, 5)), so now, per example, all the categorical vars are transformed into vectors (50 ..... 5), the total size of all of them cancatenated is 185, now we add the numerical values of the continus variables which are 185, and the inputs of our model is a vector of size 201, and them we add linear layers (201x100 and then 1000x500), batch norm and dropout.

```python
MixedInputModel(
  (embs): ModuleList(
    (0): Embedding(1116, 50)
    (1): Embedding(8, 4)
    (2): Embedding(4, 2)
    (3): Embedding(13, 7)
    (4): Embedding(32, 16)
    (5): Embedding(3, 2)
    .....
    (17): Embedding(7, 4)
    (18): Embedding(4, 2)
    (19): Embedding(4, 2)
    (20): Embedding(9, 5)
    (21): Embedding(9, 5)
  )
  (lins): ModuleList(
    (0): Linear(in_features=201, out_features=1000, bias=True)
    (1): Linear(in_features=1000, out_features=500, bias=True)
  )
  (bns): ModuleList(
    (0): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(500, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (outp): Linear(in_features=500, out_features=1, bias=True)
  (emb_drop): Dropout(p=0.04)
  (drops): ModuleList(
    (0): Dropout(p=0.001)
    (1): Dropout(p=0.01)
  )
  (bn): BatchNorm1d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
```

<p align="center"> <img src="../figures/model_for_non_structured_data.png" width="800"> </p>)

### NLP

#### spaCy
spaCy is a relatively new package for “Industrial strength NLP in Python”. It is designed with the applied data scientist in mind, meaning it does not weigh the user down with decisions over what esoteric algorithms to use for common tasks and it’s fast. Incredibly fast (it’s implemented in Cython). If you are familiar with the Python data science stack, spaCy is your numpy for NLP – it’s reasonably low-level, but very intuitive and performant.

spacy provides a one-stop-shop for tasks commonly used in any NLP project, including:

* Tokenisation.
* Lemmatisation.
* Part-of-speech tagging.
* Entity recognition.
* Dependency parsing.
* Sentence recognition.
* Word-to-vector transformations.
* Many convenience methods for cleaning and normalising text.

Examples:
 * **Lemmatisation** is the process of reducing a word to its base form, its mother word
 * **Tokenising** text is the process of splitting a piece of text into words, symbols, punctuation, spaces and other elements
 * **Part-of-speech tagging** is the process of assigning grammatical properties (e.g. noun, verb, adverb, adjective etc.) to words
 * **Entity recognition** is the process of classifying named entities found in a text into pre-defined categories, such as persons, places, organizations, dates, etc. 

```python
import spacy
nlp = spacy.load("en")
doc = nlp("The big grey dog ate all of the chocolate, but fortunately he wasn't sick!")

# Tokenization
[token.orth_ for token in doc]
''' OUT -> ['The', 'big', 'grey', 'dog', 'ate', 'all', 'of', 'the', 'chocolate', ',', 'but', 'fortunately', 'he', 'was', "n't", ' ', 'sick', '!']'''

# Lemmatization
doc = nlp(""practice practiced practicing" ")
[word.lemma_ for word in doc] 
''' OUT -> [['practice', 'practice', 'practice']]'''

# POS Tagging
doc = nlp("Conor's dog's toy was hidden under the man's sofa in the woman's house")
pos_tags = [(i, i.tag_) for i in doc2]
''' OUT -> [(Conor, 'NNP'), ('s, 'POS'), (dog, 'NN'), ('s, 'POS'), (toy, 'NN'), (was, 'VBD'), (hidden, 'VBN'), (under, 'IN'), (the, 'DT'), (man, 'NN'), ('s, 'POS'), (sofa, 'NN'), (in, 'IN'), (the, 'DT'), (woman, 'NN'), ('s, 'POS'), (house, 'NN')]'''

# Entity recognition
doc = nlp("Barack Obama is an American politician")
[(i, i.label_, i.label) for i in nlp_obama.ents]
''' OUT -> [(Barack Obama, 'PERSON', 346), (American, 'NORP', 347)]'''
```



python -m spacy download en
language modeling  

! find command seems very interesting

tokenization spacy tok

text mapping


#### References:
* [FastAi lecture 3 notes](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-4-2048a26d58aa)
* [Spacy](https://towardsdatascience.com/a-short-introduction-to-nlp-in-python-with-spacy-d0aa819af3ad
)