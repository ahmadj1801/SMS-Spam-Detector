import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Press button
def press(button):
    if button=='Exit':
        app.stop()
    elif button=='Clear':
        app.clearTextArea('sms')
    else:
        text = np.array([app.getTextArea('sms')])
        data_frame = pd.read_csv('Data/sms_spam_data.csv', encoding='windows-1252')
        data_frame.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

        print('\nData Frame after Mapping:')
        data_frame['v1'] = data_frame['v1'].map({'ham': 0, 'spam': 1})
        print(data_frame)

        x = data_frame['v2']
        y = data_frame['v1']

        count_vectorizer = CountVectorizer()
        x = count_vectorizer.fit_transform(x)
        text = count_vectorizer.transform(text)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        classifier = MultinomialNB(fit_prior=True)
        classifier.fit(x_train, y_train)
        classifier.score(x_test, y_test)
        predictions = classifier.predict(x_test)

        print('\nClassification Report:\n', classification_report(y_test, predictions))
        predictions = classifier.predict(text)
        if predictions==1:
            app.infoBox('Prediction', 'SPAM')
        else:
            app.infoBox('Prediction', 'NOT SPAM')

from appJar import gui

app = gui('Spam Detector', '500x350')
app.addLabel('title', 'SMS Spam Detector')
app.setLabelBg('title', 'cyan')
app.addLabel('break', '')
app.addLabel('input', 'Text Message')
app.addTextArea('sms')
app.addButtons(['Predict', 'Clear', 'Exit'], press)
app.go()