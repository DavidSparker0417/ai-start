import pandas as pd
import matplotlib.pyplot as plt

train_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/train.csv"
test_url = "https://raw.githubusercontent.com/teyang-lau/Disaster_Tweet_Classification/main/Data/test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

nullhist = train.isnull().sum()

plt.bar(
  list(nullhist.keys()), 
  list(nullhist.values),
  color='maroon', width=0.4)
# plt.plot()

plt.show()
