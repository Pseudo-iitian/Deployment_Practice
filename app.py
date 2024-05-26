from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle as pkl
import numpy as np
from textblob import TextBlob
import language_tool_python
import requests
from abydos.phonetic import Soundex, Metaphone, Caverphone, NYSIIS
import logging
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/api/greet')
def greet():
  return {"message": "Hello, welcome to the Flask app!"}


api_key_textcorrection = os.getenv('api_key_textcorrection')
endpoint_textcorrection = "https://api.bing.microsoft.com/"


def levenshtein(s1, s2):
  matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
  for i in range(len(s1) + 1):
    matrix[i][0] = i
  for j in range(len(s2) + 1):
    matrix[0][j] = j
  for i in range(1, len(s1) + 1):
    for j in range(1, len(s2) + 1):
      cost = 0 if s1[i - 1] == s2[j - 1] else 1
      matrix[i][j] = min(matrix[i - 1][j] + 1,
                          matrix[i][j - 1] + 1,
                          matrix[i - 1][j - 1] + cost)
  return matrix[len(s1)][len(s2)]

@app.route('/api/spelling_accuracy', methods=['POST'])
def spelling_accuracy():
  data = request.json
  extracted_text = data.get('text', '')
  
  if not extracted_text:
    return jsonify({"error": "No text provided"}), 400
  
  spell_corrected = str(TextBlob(extracted_text).correct())
  accuracy_score = ((len(extracted_text) - levenshtein(extracted_text, spell_corrected)) / (len(extracted_text) + 1)) * 100
  return jsonify({"spelling_accuracy": accuracy_score})




if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000)
