import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

## from keras.models import load model
from tensorflow.keras.models import load_model
model=load_model('chatbot_model.h5')
import json
import random
intents=json.loads(open('intents.json', encoding='utf8').read())
words=pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
  print('sentence words', sentence_words)
  return sentence_words

def bow(sentence, words, show_details=True):
  ## tokenize the pattern
  sentence_words = clean_up_sentence(sentence)

  ## Bag of words = matrix of hi words
  bag=[0]*len(words)

  for s in sentence_words:
    for i, w in enumerate(words):
        if(w==s):
          ## assign 1 if current word is in the vocabularly position
          bag[i]=1
          if show_details:
            print("found in bag: %s" % w)
  return(np.array(bag))

def predict_class(sentence, model):
  p=bow(sentence, words, show_details=False)
  print('p = ', p)
  res = model.predict(np.array([p]))[0]
  print('res = ', res)
  ERROR_THRESHOLD = 0.25
  results = [[i, r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
  print('result = ', results)
  ## Sort by strength of probability
  results.sort(key=lambda x: x[1], reverse=True)
  result_list=[]

  for r in results:
    result_list.append({'intent':classes[r[0]], "probability":str(r[1])})

  return result_list

def getResponse(ints, intents_json):
  tag=ints[0]['intent']
  print('tag', tag)
  list_of_intents = intents_json['intents']
  print('list of intents',list_of_intents )
  for i in list_of_intents:
    if(i['tag']==tag):
      result = random.choice(i['responses'])
      break
  return result

def chatbot_response(msg):
  ints=predict_class(msg,model)
  print('inte', ints)
  try:
      res = getResponse(ints, intents)
  except:
      print("An exception occured")
      res="I can't answer this query. I am sorry but this is ont of my limitation"
  print('chatbot response:', res)
  return res

#from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify

app = Flask(__name__)
#run_with_ngrok(app)

@app.route("/", methods=['GET'])
def hello():
    return jsonify({"health": "Hello, server is running successfully."})

def decrypt(msg):
    ## i/p: what is machine learning
    ## what is machine learning
    ## remove + and replace it with spaces
    string = msg
    new_string = string.replace("+", " ")
    return new_string

@app.route("/query/<sentence>")
def query_chatbot(sentence):

    ## decrypt message
    dec_msg = decrypt(sentence)

    response = chatbot_response(dec_msg)

    json_obj = jsonify({'top': {"res": response}})

    return json_obj
app.run()


