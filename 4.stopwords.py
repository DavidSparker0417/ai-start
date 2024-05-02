import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize, sent_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

train_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/train.csv"
test_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)
color = [sns.xkcd_rgb['medium blue'], sns.xkcd_rgb['pale red']]

# Get all the word tokens in dataframe for Disaster and Non-Disaster
corpus0 = []
[corpus0.append(word.lower()) for var in train[train.target == 0].text for word in word_tokenize(var)]
corpus1 = []
[corpus1.append(word.lower()) for var in train[train.target == 1].text for word in word_tokenize(var)]

# Function for counting top stopwords in a corpus
def count_top_stopwords(corpus):
  stopwords_freq = {}
  for word in corpus:
    if word in stop:
      if word in stopwords_freq:
        stopwords_freq[word] += 1
      else:
        stopwords_freq[word] = 1
  topwords = sorted(stopwords_freq.items(), key=lambda item: item[1], reverse=True)[:10]
  x,y = zip(*topwords) # get key and values
  return x,y

x0,y0 = count_top_stopwords(corpus0)
x1,y1 = count_top_stopwords(corpus1)

# Plot bar plot of top stopwords for each class
plt.figure(figsize=[15,4])
plt.subplot(1,2,1)
plt.bar(x0, y0, color=color[0])
plt.title('Top stopwords for Non-Disaster Tweets')
plt.subplot(1,2,2)
plt.bar(x1,y1,color=color[1])
plt.title('Top Stopwords for Disaster Tweets')
plt.show()