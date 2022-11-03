from django.shortcuts import render, redirect


# Create your views here.
from django.http import HttpResponse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from gnewsclient import gnewsclient
import os
current_path = os.path.abspath(os.getcwd())

def home(request):
    if request.method=='POST':
        news_detect = request.POST['news_input']
        print(news_detect)
        context = detect(news_detect)
        context["user_news"] = news_detect
        print(context)
        return render(request, 'fakestat/result.html', context)
    context = {}
    context["news_list"] = trending()
    return render(request, 'fakestat/home.html', context)

def detect(news):

    algo_accuracies = {}

    random = pickle.load(open(os.path.join(current_path, "static\sav", "random_forest_data.sav"), 'rb'))
    tfidf_vectorize = pickle.load(open(os.path.join(current_path, "static\sav", "vectorizer_data.sav"), 'rb'))
    vec_news = tfidf_vectorize.transform([news])
    
    rf_pred = random.predict(vec_news) # vectorizer input provided by user
    print(rf_pred)
    algo_accuracies["random_forest"] = rf_pred


    return algo_accuracies


def result(request):
    context = {}
    return render(request, 'fakestat/result.html', context)

def trending():
    client = gnewsclient.NewsClient(language='english', location='india', topic='nation', max_results=5)
    
    news_list = client.get_news()    
    return news_list

#Incredible 360 Video Shows What It’s Like To Be In A Coffin    real
#Football Fans Get Around Face Covering Ban by Wearing Islamic Niqabs   fake
#Fact Check: Trump Misleads About The Times’s Reporting on Surveillance - The New York Times    fake


