import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

stop = set(stopwords.words('english'))

train_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/train.csv"
test_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

color = [sns.xkcd_rgb['medium blue'], sns.xkcd_rgb['pale red']]

# combine stop words from different sources
stop = ENGLISH_STOP_WORDS.union(stop)

#  function for removing url from text


def remove_url(txt):
  return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/S+)", "", txt).split())


# Get all the word tokens in dataframe for Disaster and Non-Disaster
# - remove url, tokenize tweet into words, lowercase words
corpus0 = []  # None Disaster
[corpus0.append(word.lower()) for var in train[train.target == 0].text for word in
 word_tokenize(remove_url(var))]
# use filter to unselect stopwords
corpus0 = list(filter(lambda x: x not in stop, corpus0))

corpus1 = []  # None Disaster
[corpus1.append(word.lower()) for var in train[train.target == 1].text for word in
 word_tokenize(remove_url(var))]
# use filter to unselect stopwords
corpus1 = list(filter(lambda x: x not in stop, corpus1))

a = Counter(corpus0).most_common()
df0 = pd.DataFrame(a, columns=['Word', 'Count'])

a = Counter(corpus1).most_common()
df1 = pd.DataFrame(a, columns=['Word', 'Count'])

# Plot for Disaster and Non-Disaster
plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.barplot(x='Word', y='Count', data=df0.head(10), color=color[1]).set_title('Most Common Words for Non-Disasters')
plt.xticks(rotation=45)
plt.subplot(1,2,2)
sns.barplot(x='Word', y='Count', data=df1.head(10), color=color[0]).set_title('Most Common Words for Disasters')
plt.xticks(rotation=45)
plt.show()

