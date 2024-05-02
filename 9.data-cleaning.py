import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation
from nltk.corpus import stopwords
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import seaborn as sns
from wordcloud import WordCloud

train_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/train.csv"
test_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

stop = set(stopwords.words('english'))
color = [sns.xkcd_rgb['medium blue'], sns.xkcd_rgb['pale red']]
# Replace NaNs with 'None'
train.keyword.fillna('None', inplace=True)

## Expand Contractions

# Function for expanding most common contractions
def decontraction(phrase):
  # specific
  phrase = re.sub(r"won\'t", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)
  
  # general
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)
  return phrase

train.text = [decontraction(var) for var in train.text]

## Rmove Emojis
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
  emoji_pattern = re.compile("["
                             u"\U0001F600-\U0001F64F" # emoticons
                             u"\U0001F300-\U0001F5FF" # symbols & pictorgraphs
                             u"\U0001F680-\U0001F6FF" # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r'', text)

def remove_url(txt):
  return " ".join(re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', txt).split())

print(remove_emoji("OMG there is a volcano eruption!!! 😭😇😷"))
train.text = train.text.apply(lambda x: remove_emoji(x))

# Remove URLs
train.text = train.text.apply(lambda x: remove_url(x))

## Rmove Punctuations except '!?'
def remove_punct(text):
  new_punct = re.sub(r'\ |\!|\?', '', punctuation)
  table = str.maketrans('', '', new_punct)
  return text.translate(table)

train.text = train.text.apply(lambda x: remove_punct(x))

## Replace amp
def replace_amp(text):
  text = re.sub(r" amp ", " and ", text)
  return text

train.text = train.text.apply(lambda x: replace_amp(x))

# word segmentation
from wordsegment import load, segment
load()
train.text = train.text.apply(lambda x: ' '.join(segment(x)))

# lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemma(text):
  words = word_tokenize(text)
  return ' '.join([lemmatizer.lemmatize(w.lower(), pos='v') for w in words])

train.text = train.text.apply(lambda x: lemma(x))

# Ngrams
from nltk.util import ngrams

def generate_ngrams(text, n):
  words = word_tokenize(text)
  return [' '.join(ngram) 
          for ngram in list(get_data(ngrams(words, n))) 
          if not all(w in stop for w in ngram)] # exclude if all are stopwords

# in newer versions of python, raising StopIteration exception to end a generator, which is used in ngram, is deprecated
def get_data(gen):
  try:
    for elem in gen:
      yield elem
  except (RuntimeError, StopIteration):
    return

# Bigrams
def show_bigrams():
  bigrams_disaster = train[train.target == 1].text.apply(lambda x: generate_ngrams(x, 2))
  bigrams_ndisaster = train[train.target == 0].text.apply(lambda x: generate_ngrams(x, 2))

  bigrams_d_dict = {}
  for bgs in bigrams_disaster:
    for bg in bgs:
      if bg in bigrams_d_dict:
        bigrams_d_dict[bg] += 1
      else:
        bigrams_d_dict[bg] = 1
  bigrams_d_df = pd.DataFrame(bigrams_d_dict.items(), columns=['Bigrams', 'Count'])

  bigrams_nd_dict = {}
  for bgs in bigrams_ndisaster:
    for bg in bgs:
      if bg in bigrams_nd_dict:
        bigrams_nd_dict[bg] += 1
      else:
        bigrams_nd_dict[bg] = 1
        
  bigrams_nd_df = pd.DataFrame(bigrams_nd_dict.items(), columns=['Bigrams', 'Count'])

  # Barplots for bigrams
  plt.figure(figsize=(15,10))
  plt.subplot(1,2,1)
  sns.barplot(x='Count', y='Bigrams', 
              data=bigrams_nd_df
              .sort_values('Count', ascending=False).head(40),
              color=color[0]).set_title('Most Common Bigrams for Non-Disasters')
  ax = plt.gca()
  ax.set_ylabel('')

  plt.subplot(1,2,2)
  sns.barplot(x='Count', y='Bigrams', 
              data=bigrams_d_df
              .sort_values('Count', ascending=False).head(40),
              color=color[0]).set_title('Most Common Bigrams for Non-Disasters')
  ax = plt.gca()
  ax.set_ylabel('')
  plt.tight_layout()
  # plt.show()

  # Woudcloud for bigrams
  plt.figure(figsize=(15,10))
  plt.subplot(1,2,1)
  my_cloud = WordCloud(background_color='white',
                      stopwords=stop).generate_from_frequencies(bigrams_nd_dict)
  plt.imshow(my_cloud, interpolation='bilinear')
  plt.axis('off')

  plt.subplot(1,2,2)
  my_cloud = WordCloud(background_color='white',
                      stopwords=stop).generate_from_frequencies(bigrams_d_dict)
  plt.imshow(my_cloud, interpolation='bilinear')
  plt.axis('off')

  plt.show()
  
# Trigrams
def show_trigrams():
  trigrams_disaster = train[train.target == 1].text.apply(lambda x: generate_ngrams(x, 3))
  trigrams_ndisaster = train[train.target == 0].text.apply(lambda x: generate_ngrams(x, 3))

  trigrams_d_dict = {}
  for bgs in trigrams_disaster:
    for bg in bgs:
      if bg in trigrams_d_dict:
        trigrams_d_dict[bg] += 1
      else:
        trigrams_d_dict[bg] = 1
  trigrams_d_df = pd.DataFrame(trigrams_d_dict.items(), columns=['Trigrams', 'Count'])

  trigrams_nd_dict = {}
  for bgs in trigrams_ndisaster:
    for bg in bgs:
      if bg in trigrams_nd_dict:
        trigrams_nd_dict[bg] += 1
      else:
        trigrams_nd_dict[bg] = 1
        
  trigrams_nd_df = pd.DataFrame(trigrams_nd_dict.items(), columns=['Trigrams', 'Count'])

  # Barplots for trigrams
  plt.figure(figsize=(15,10))
  plt.subplot(1,2,1)
  sns.barplot(x='Count', y='Trigrams', 
              data=trigrams_nd_df
              .sort_values('Count', ascending=False).head(40),
              color=color[0]).set_title('Most Common Trigrams for Non-Disasters')
  ax = plt.gca()
  ax.set_ylabel('')

  plt.subplot(1,2,2)
  sns.barplot(x='Count', y='Trigrams', 
              data=trigrams_d_df
              .sort_values('Count', ascending=False).head(40),
              color=color[0]).set_title('Most Common Trigrams for Non-Disasters')
  ax = plt.gca()
  ax.set_ylabel('')
  plt.tight_layout()

  # Woudcloud for trgrams
  plt.figure(figsize=(15,10))
  plt.subplot(1,2,1)
  my_cloud = WordCloud(background_color='white',
                      stopwords=stop).generate_from_frequencies(trigrams_nd_dict)
  plt.imshow(my_cloud, interpolation='bilinear')
  plt.axis('off')

  plt.subplot(1,2,2)
  my_cloud = WordCloud(background_color='white',
                      stopwords=stop).generate_from_frequencies(trigrams_d_dict)
  plt.imshow(my_cloud, interpolation='bilinear')
  plt.axis('off')

  plt.show()

# show_bigrams()
# show_trigrams()

## Remove Stopwords
def remove_stopwords(text):
  word_tokens = word_tokenize(text)
  return ' '.join([w.lower() for w in word_tokens if not w.lower() in stop])
train['text_notopwords'] = train.text.apply(lambda x: remove_stopwords(x))

import numpy as np
from PIL import Image
import requests

img_url = 'https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Pictures/Twitter-Logo.png'
img = Image.open(requests.get(img_url, stream=True).raw)
mask = np.array(img)
reverse = mask[...,::-1,:]

def wc_words(target, mask=mask):
  words = [word.lower() for tweet in train[train.target == target].text_notopwords for word in tweet.split()]
  words = list(filter(lambda w: w != 'like', words))
  words = list(filter(lambda w: w != 'new', words))
  words = list(filter(lambda w: w != 'people', words))
  dict = {}
  for w in words:
    if w in dict:
      dict[w] += 1
    else:
      dict[w] = 1
  # plot using frequencies
  my_cloud = WordCloud(background_color='white', stopwords=stop, mask=mask, random_state=0).generate_from_frequencies(dict)
  plt.subplot(1,2,target+1)
  plt.imshow(my_cloud, interpolation='bilinear')
  plt.axis('off')

def show_wordcloud_tweet():
  plt.figure(figsize=(15,10))
  wc_words(0)
  plt.title('Non-Disaster')
  wc_words(1, reverse)
  plt.title('Disaster')
  plt.show()

pd.options.display.max_colwidth = 200
# for t in train['text'].sample(n=20, random_state=0):
#   print(t)
pd.reset_option('max_colwidth')
pd.reset_option('max_colwidth')
train.drop('text_notopwords', axis=1, inplace=True)
train.head()

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
  train.drop(['id', 'keyword', 'location', 'target'], axis=1),
  train[['target']], 
  test_size=0.2,
  stratify=train[['target']],
  random_state=0)
X_train_text = X_train['text']
X_val_text = X_val['text']

print('X_train shape: ', X_train.shape)
print('X_val shape: ', X_val.shape)
print('y_train_shape: ', y_train.shape)
print('y_val_shape: ', y_val.shape)
print('Train Class Proportion:\n', y_train['target'].value_counts() / len(y_train)*100)
print('\nValidation Class Proportion:\n', y_val['target'].value_counts() / len(y_val)*100)