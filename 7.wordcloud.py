import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation
from nltk.corpus import stopwords

train_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/train.csv"
test_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)
stop = set(stopwords.words('english'))

def clean(word):
  for p in punctuation: word = word.replace(p, '')
  return word

from wordcloud import WordCloud

def wc_hash(target):
  hashtag = [clean(w[1:].lower()) for var in train[train.target == target].text for w in var.split() if '#' in w and w[0] == '#']
  hashtag = ' '.join(hashtag)
  my_cloud = WordCloud(background_color='white', stopwords=stop).generate(hashtag)
  
  plt.subplot(1,2,target+1)
  plt.imshow(my_cloud, interpolation='bilinear')
  plt.axis("off")

plt.figure(figsize=(15,4))
wc_hash(0)
plt.title('Non-Disaster')
wc_hash(1)
plt.title('Disaster')
plt.show()