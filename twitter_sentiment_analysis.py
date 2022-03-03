import tweepy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from nltk.corpus import stopwords
from textblob import Word, TextBlob

# Απαραίτητοι κωδικοί για να κάνουμε σύνδεση στο twitter API
consumerKey = ''
consumerSecret = ''
accessToken = ''
accessTokenSecret = ''

# Δημιουργούμε ένα αντικείμενο τύπου Authedicate
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

# Προσθέτουμε τους κωδικούς
authenticate.set_access_token(accessToken, accessTokenSecret)

# Δημιουργούμε ένα αντικείμενο API καθώς περνάμε τις πληροφορίες μας
api = tweepy.API(authenticate, wait_on_rate_limit = True)

# Κάνουμε εξόρυξη δεδομένων από τον χρήστη POTUS (President Biden), 
# link: https://twitter.com/POTUS
# "τραβάμε" τα 1000 τελευταία tweets του λογαριασμού του
posts = api.user_timeline(screen_name="POTUS", count=1000, lang="en", tweet_mode="extended")


# Τυπώνουμε ενδεικτικά 5 tweets από αυτά
i = 1
for tweet in posts[0:5]:
  print(f" {i}) {tweet.full_text} \n")
  i += 1

# Δημιουργούμε ένα dataframe με ένα column που το ονομάζουμε Tweets
# Κάθε γραμμή του column θα είναι και ενα tweet του χρήστη
df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])


# Συνάρτηση που καθαρίζει τα tweets
def cleanTxt(text):
  text = re.sub(r'@[A-Za-z0-9]+','', text) # Αφαιρούμε @mentions
  text = re.sub(r'#', '', text) # Αφαιρούμε το '#'
  text = re.sub(r'RT[\s]+', '', text) # Αφαιρούμε τα RT
  text = re.sub(r'https?:\/\/\S+', '', text) # Αφαιρούμε τα hyper link

  return text

# Εφαρμόζουμε τη συνάρτηση στα δεδομένα μας
df['Tweets'] = df['Tweets'].apply(cleanTxt)

# Εμφανίζουμε τα δεδομένα
print(df)

# Δημιουργούμε μια συνάρτηση που μας επιστρέφει την υποκειμενικότητα
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

# Δημιουργούμε μια συνάρτηση που μας επιστρέφει την πολικότητα
def getPolarity(text):
  return TextBlob(text).sentiment.polarity

# Δημιουργούμε δύο columns στο dataframe μας
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

# Εμφανίζουμε το καινούργιο dataframe μας 
print(df.head())

# Δημιουργούμε μια συνάρτηση η οποία μας επιστρέφει
# σύμφονα με τη πολικότητα αν το συναίσθημα είναι
# αρνητικό θετικό ή ουδέτερο
def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'

# εφαρμόζουμε τη συνάρτηση στο dataframe μας
df['Analysis'] = df['Polarity'].apply(getAnalysis)

# Εμφανίζουμε το dataframe μας
print(df)

# Το εμφανίζουμε με γραφικό τρόπο με τη βοήθεια του matplotlib 
plt.figure(figsize=(8,6)) 
for i in range(0, df.shape[0]):
  plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color='Blue') 
# θέτουμε τίτλο στο γράφημα μας   
plt.title('Sentiment Analysis') 
# θέτουμε ονομασία στον x άξονα
plt.xlabel('Polarity') 
# θέτουμε ονομασία στον y άξονα
plt.ylabel('Subjectivity') 
# εμφάνιση 
plt.show()

# Εμφανίζουμε το value_counts
df['Analysis'].value_counts()

# Εμφανίζουμε σε γράφημα σε μορφή μπαρών το μέγεθος
# των θετικών-αρνητικών-ουδέτερων συναισθημάτων
plt.title('Sentiment Analysis')
# ονομασία για τον x άξονα
plt.xlabel('Sentiment')
# ονομασία για τον y άξονα
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind = 'bar')
plt.show()