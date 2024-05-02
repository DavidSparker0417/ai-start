import nltk
nltk.download('punkt')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import word_tokenize, sent_tokenize

train_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/train.csv"
test_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

# count number of characters in each tweet
train['char_len'] = train.text.str.len()

# count number of words in each tweet
word_tokens = [len(word_tokenize(var)) for var in train.text]
train['word_len'] = word_tokens

# count number of sentence in each tweet
sent_tokens = [len(sent_tokenize(var)) for var in train.text]
train['sent_len'] = sent_tokens

plot_cols = ['char_len', 'word_len', 'sent_len']
plot_titles = ['Character Length', 'Word Length', 'Sentence Length']
color = [sns.xkcd_rgb['medium blue'], sns.xkcd_rgb['pale red']]

plt.figure(figsize=(20, 4))
for counter, i in enumerate([0, 1, 2]):
  plt.subplot(1,3,counter+1)
  sns.distplot(train[train.target == 1][plot_cols[i]], label='Disaster',
    color=color[1]).set_title(plot_titles[i])
  sns.distplot(train[train.target == 0][plot_cols[i]], label = 'Non-Disaster', color=color[0])
  # plt.legend()
plt.show()
