import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from string import punctuation
from nltk.corpus import stopwords
import gensim
from sklearn.manifold import TSNE

def w2varray_mean(model_w2v,tweets_list):
    w2v_array = np.empty([len(tweets_list),200])
    i=0

    for tweet in tweets_list:
        mylist = []
        for each in tweet.split():
            if each in model_w2v.wv.vocab:
                k = model_w2v[each]
                mylist.append(k)
        if not mylist:
            continue
        array = np.array(mylist)
        mean = np.mean(array,axis = 0)
        w2v_array[i] = mean.tolist()
        i+=1

    return w2v_array

def cleanup(tweets_list):
    clean_tweets = []
    for tweet in tweets_list:
        words = []
        for word in tweet.lower().split()[2:]:
            if "http" in word or "@" in word:
                continue
            word = word.translate(str.maketrans('', '', punctuation))
            if word == "":
                continue
            if not word in stopwords.words('english'):
                words.append(word)
        clean_tweet = " ".join(words)
        clean_tweets.append(clean_tweet)
    return clean_tweets

def show_WordCloud(text, str):
    wordcloud = WordCloud(background_color = "white").generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.title("Most frequent words in "+ str + " tweets")
    plt.show()

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
               textcoords='offset points',
                         ha='right',
                         va='bottom')
    plt.show()
