import pandas as pd
from textblob import TextBlob
import re

train_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/train.csv"
test_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

# polarity and subjectivity
train['polarity'] = [TextBlob(var).sentiment.polarity for var in train.text]
train['subjectivity'] = [TextBlob(var).sentiment.subjectivity for var in train.text]

#################################################################
# exclaimation and question marks
train['exclaimation_num'] = [var.count('!') for var in train.text]
train['questionmark_num'] = [var.count('?') for var in train.text]

#################################################################
# count number of hashtag and mentions
# Function for counting number of hashtags and mentions
def count_url_hashtag_mention(text):
  url_nums = len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_.&+]|[!*\(\),]|(?:%[0-9a-fA-F]))+', text))
  word_tokens = text.split()
  hash_num = len([word for word in word_tokens if word[0] == '#' and word.count('#') == 1]) # only appears once in front of word
  mention_num = len([word for word in word_tokens if word[0] == '@' and word.count('@') == 1]) # only appears once in front of word
  return url_nums, hash_num, mention_num

#################################################################
# count number of contractions
contractions = ["'t", "'re", "'s", "'d", "'ll", "'ve", "'m"]
train['contraction_num'] = [sum([var.count(cont) for cont in contractions]) for var in train.text]