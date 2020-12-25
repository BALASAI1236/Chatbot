from flask import Flask, render_template, request, jsonify, json
from nltk.chat.util import Chat, reflections
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder as le

app = Flask(__name__)

df = pd.read_csv("C:/xampp/htdocs/mini_project/diseases_symptoms_train.csv", index_col=0)
disease_list = df['prognosis'].unique().reshape(1,41)
df['prognosis'] = le.fit_transform(df, df['prognosis'])
x_train = df.iloc[:,0:131]
y_train = df['prognosis']
dec = DecisionTreeClassifier()
dec.fit(x_train, y_train)

pairs = [
    ['I\'am (.*)', ['Hello %1']],
    ['(Hi|Hello|Hola|Hey|Namaste|Good Day)',['Hey there', 'Hello', 'Good day to you', 'Hola amigo']],
    ['How (.*) you?',['I\'m doing great, what about you?']],
    ['(Sup|What\'s up)',['Still stuck in this box of intelligence ;)']],
    ['(.*) weather (.*)', ['The weather\'s just fine, not too good not too bad']],
    ['(.*) (fine|good|great|nice) (.*)',['Glad to hear that']],
    ['(.*) are you?', ['I\'m DocBot, Your Personal Healthcare Assistant']],
    ['(.*) you', ['%1 you too']],
    ['What (.*) do?', ['Ask me anything related to your health']],
    ['What\'s your name?',['I\'am DocBot']],
    ['(.*) created you?', ['Manoj created me']],
    ['(.*) fever', ['Please, could you measure your body temperature and if it is greater than 98.4F consult a doctor and take loads of rest :)']],
    ['(.*) cold', ['Please, just take some rest and if you have headache along with it I suggest you to sleep in a peaceful place.']],
    ['(.*) cough', ['Take citrozen according to your dosage and take loads of rest :)']],
    ['(.*) headache', ['Sleep for 4 to 5 hours if it\'s day and if it\'s night I suggest you to sleep for 7 to 8 hours, you should be fine. If it still persists consult a doctor, don\'t worry :)']],
    ['Suggest (.*) medication', ['Paracetamol with correct dosage is suggested for fever, caugh, cold and headache, but it would be better for you to meet a doctor. If this is\'nt the answer your\'e looking for, try to enter the diseases one by one']],
    ['(.*) sick', ['Please be more specific or you can always check our disease predictor :)']],
    ['(.*) stomach ache', ['Please consider taking rest, if the pain persists for a long time consult a doctor']],
]

chat = Chat(pairs, reflections)

@app.route('/chatbot/<user_inp>', methods=['POST','GET'])
def get_mssg(user_inp):
    bot_reply = chat.respond(user_inp)
    return bot_reply
    
@app.route('/prediction', methods=['POST'])
def predict():
    symp = []
    files = request.get_json()
    for key, value in files.items() :
        symp.append(value)
    symp_new = []
    symp_new.append(symp)
    df_test = DataFrame(symp_new)
    pred = dec.predict(df_test)
    return disease_list[0][pred[0]]

if __name__ == '__main__':
    app.run(debug=True)