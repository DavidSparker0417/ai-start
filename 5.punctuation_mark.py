import nltk
import pandas as pd
from string import punctuation
import matplotlib.pyplot as plt
import seaborn as sns

train_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/train.csv"
test_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

color = [sns.xkcd_rgb['medium blue'], sns.xkcd_rgb['pale red']]

# Get all the punctuations in dataframe for Disaster and Non-Disaster
corpus0 = []
[corpus0.append(c) for var in train[train.target == 0].text for c in var]
corpus0 = list(filter(lambda x: x in punctuation, corpus0))
corpus1 = []
[corpus1.append(c) for var in train[train.target == 1].text for c in var]
corpus1 = list(filter(lambda x: x in punctuation, corpus1))

from collections import Counter
x0,y0 = zip(*Counter(corpus0).most_common())
x1,y1 = zip(*Counter(corpus1).most_common())

#  Plot bar plot of top punctuations for each class
plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.bar(x0,y0, color=color[0])
plt.title('Top punctuations for Non-Disaster Tweets')
plt.subplot(1,2,2)
plt.bar(x1,y1, color=color[1])
plt.title('Top punctuations for Disaster Tweets')
plt.show()
