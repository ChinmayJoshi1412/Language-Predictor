import pandas as pd
import re
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
class langDetect:
    def __init__(self): #ML Model Constructor
        #ignoring warnings
        warnings.simplefilter("ignore")
        # Loading the dataset from root dir conating specimen languages
        data = pd.read_csv("Language Detection.csv")
        # value count for each language
        data["Language"].value_counts()
        # separating the independent and dependant features
        X = data["Text"]
        y = data["Language"]
        # converting categorical variables to numerical
        self.le = LabelEncoder()
        y = self.le.fit_transform(y)
        # creating a list for appending the preprocessed text
        data_list = []
        # iterating through all the text
        for text in X:
            # removing the symbols and numbers
            text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
            text = re.sub(r'[[]]', ' ', text)
            # converting the text to lower case
            text = text.lower()
            # appending to data_list
            data_list.append(text)
        # creating vectorised words
        self.cv = CountVectorizer()
        X = self.cv.fit_transform(data_list).toarray()
        #train test splitting - Supervised Learning Module
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
        #model creation and prediction
        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

    def predict(self,text):
        x = self.cv.transform([text]).toarray()
        lang = self.model.predict(x)
        lang = self.le.inverse_transform(lang)
        return lang

    def result(self, input1):
        return self.predict(input1)


