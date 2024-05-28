from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from textblob import TextBlob

from typing import List
from language_tool_python.server import LanguageTool
from language_tool_python.match import Match
from language_tool_python import LanguageTool
import requests
from abydos.phonetic import Soundex, Metaphone, Caverphone, NYSIIS
import logging
from dotenv import load_dotenv
import os
import joblib  
from flask_cors import cross_origin

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
CORS(app, origins=["*"])

@app.route('/',methods=['POST','GET'])
@cross_origin()
def home():
  return {"message": "Wakeup Call made!"}

quiz_model = None
loaded_model = None

try:
  quiz_model = joblib.load(r"Random_Forest_Model.sav")
  loaded_model = joblib.load(r"Decision_tree_model.sav")
except Exception as e:
  logging.error(f"Failed to load models: {e}")

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
def get_spelling_accuracy():
  try:
    # Get the text data from the request
    request_data = request.get_json()
    extracted_text = request_data.get('text')

    # Calculate the spelling accuracy
    accuracy = spelling_accuracy(extracted_text)

    # Prepare the response
    response = {
      "ok": True,
      "message": "Spelling accuracy calculated successfully",
      "accuracy": accuracy
    }

    # Return the response
    return jsonify(response), 200

  except Exception as e:
    # If an error occurs, return an error response
    response = {
      "ok": False,
      "message": f"An error occurred: {str(e)}"
    }
    return jsonify(response), 500


def spelling_accuracy(extracted_text):
  spell_corrected = TextBlob(extracted_text).correct()
  return ((len(extracted_text) - (levenshtein(extracted_text, spell_corrected)))/(len(extracted_text)+1))*100


@app.route('/api/grammatical_accuracy', methods=['POST'])
@cross_origin()
def get_grammatical_accuracy():
  try:
    # Get the text data from the request
    request_data = request.get_json()
    extracted_text = request_data.get('text')

    # Calculate grammatical accuracy
    accuracy_score = grammatical_accuracy(extracted_text)

    # Prepare the response
    response = {
      "ok": True,
      "message": "Grammatical accuracy calculated successfully",
      "accuracy_score": accuracy_score
    }

    # Return the response
    return jsonify(response), 200

  except Exception as e:
    # If an error occurs, return an error response
    response = {
        "ok": False,
        "message": f"An error occurred: {str(e)}"
    }
    return jsonify(response), 500


my_tool = LanguageTool('en-US')

def correct_it(text: str, matches: List[Match]) -> str:
  """Automatically apply suggestions to the text."""
  ltext = list(text)
  matches = [match for match in matches if match.replacements]
  errors = [ltext[match.offset:match.offset + match.errorLength]
            for match in matches]
  correct_offset = 0
  for n, match in enumerate(matches):
      frompos, topos = (correct_offset + match.offset,
                        correct_offset + match.offset + match.errorLength)
      if ltext[frompos:topos] != errors[n]:
          continue
      repl = match.replacements[0]
      ltext[frompos:topos] = list(repl)
      correct_offset += len(repl) - len(errors[n])
  return ''.join(ltext)

def grammatical_accuracy(extracted_text):
  try:
    # Initialize LanguageTool

    # Correct spelling
    spell_corrected = TextBlob(extracted_text).correct()
    print("Spell corrected:", spell_corrected)


    matches = my_tool.check(spell_corrected)  # Assuming lang_tool is an instance of LanguageTool

    # Correct grammar
    correct_text = correct_it(spell_corrected,matches)
    print("Corrected text:", correct_text)

    # Calculate accuracy
    extracted_text_set = set(spell_corrected.split(" "))
    correct_text_set = set(correct_text.split(" "))
    n = max(len(extracted_text_set - correct_text_set), len(correct_text_set - extracted_text_set))
    accuracy = ((len(spell_corrected) - n) / (len(spell_corrected) + 1)) * 100

    if(accuracy):
      return accuracy
    return 0

  except Exception as e:
    # Handle any errors gracefully
    print(f"An error occurred: {e}")
    return None

@app.route('/api/percentage_of_corrections', methods=['POST'])
@cross_origin()
def get_percentage_of_corrections():
  try:
    # Get the text data from the request
    request_data = request.get_json()
    extracted_text = request_data.get('text')

    # Calculate percentage of corrections
    correction_percentage = percentage_of_corrections(extracted_text)

    # Prepare the response
    response = {
        "ok": True,
        "message": "Percentage of corrections calculated successfully",
        "correction_percentage": correction_percentage
    }

    # Return the response
    return jsonify(response), 200

  except Exception as e:
    # If an error occurs, return an error response
    response = {
        "ok": False,
        "message": f"An error occurred: {str(e)}"
    }
    return jsonify(response), 500


def percentage_of_corrections(extracted_text):
  data = {'text': extracted_text}
  params = {
    'mkt': 'en-us',
    'mode': 'proof'
  }
  headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Ocp-Apim-Subscription-Key': api_key_textcorrection,
  }
  response = requests.post(endpoint_textcorrection, headers=headers, params=params, data=data)
  json_response = response.json()
  flagged_tokens_count = len(json_response.get('flaggedTokens', []))
  extracted_word_count = len(extracted_text.split(" "))
  if extracted_word_count > 0:
    percentage_corrected = (flagged_tokens_count / extracted_word_count) * 100
  else:
    percentage_corrected = 0
  return percentage_corrected



@app.route('/api/percentage_of_phonetic_accuraccy', methods=['POST'])
@cross_origin()
def get_percentage_of_phonetic_accuraccy():
  try:
    # Get the text data from the request
    request_data = request.get_json()
    extracted_text = request_data.get('text')

    # Calculate percentage of phonetic accuracy
    phonetic_accuracy_percentage = percentage_of_phonetic_accuraccy(extracted_text)

    # Prepare the response
    response = {
        "ok": True,
        "message": "Percentage of phonetic accuracy calculated successfully",
        "phonetic_accuracy_percentage": phonetic_accuracy_percentage
    }

    # Return the response
    return jsonify(response), 200

  except Exception as e:
    # If an error occurs, return an error response
    response = {
        "ok": False,
        "message": f"An error occurred: {str(e)}"
    }
    return jsonify(response), 500

def percentage_of_phonetic_accuraccy(extracted_text: str):
  soundex = Soundex()
  metaphone = Metaphone()
  caverphone = Caverphone()
  nysiis = NYSIIS()
  spell_corrected = TextBlob(extracted_text).correct()

  extracted_text_list = extracted_text.split(" ")
  extracted_phonetics_soundex = [soundex.encode(string) for string in extracted_text_list]
  extracted_phonetics_metaphone = [metaphone.encode(string) for string in extracted_text_list]
  extracted_phonetics_caverphone = [caverphone.encode(string) for string in extracted_text_list]
  extracted_phonetics_nysiis = [nysiis.encode(string) for string in extracted_text_list]

  extracted_soundex_string = " ".join(extracted_phonetics_soundex)
  extracted_metaphone_string = " ".join(extracted_phonetics_metaphone)
  extracted_caverphone_string = " ".join(extracted_phonetics_caverphone)
  extracted_nysiis_string = " ".join(extracted_phonetics_nysiis)

  spell_corrected_list = spell_corrected.split(" ")
  spell_corrected_phonetics_soundex = [soundex.encode(string) for string in spell_corrected_list]
  spell_corrected_phonetics_metaphone = [metaphone.encode(string) for string in spell_corrected_list]
  spell_corrected_phonetics_caverphone = [caverphone.encode(string) for string in spell_corrected_list]
  spell_corrected_phonetics_nysiis = [nysiis.encode(string) for string in spell_corrected_list]

  spell_corrected_soundex_string = " ".join(spell_corrected_phonetics_soundex)
  spell_corrected_metaphone_string = " ".join(spell_corrected_phonetics_metaphone)
  spell_corrected_caverphone_string = " ".join(spell_corrected_phonetics_caverphone)
  spell_corrected_nysiis_string = " ".join(spell_corrected_phonetics_nysiis)

  soundex_score = (len(extracted_soundex_string)-(levenshtein(extracted_soundex_string,spell_corrected_soundex_string)))/(len(extracted_soundex_string)+1)
  metaphone_score = (len(extracted_metaphone_string)-(levenshtein(extracted_metaphone_string,spell_corrected_metaphone_string)))/(len(extracted_metaphone_string)+1)
  caverphone_score = (len(extracted_caverphone_string)-(levenshtein(extracted_caverphone_string,spell_corrected_caverphone_string)))/(len(extracted_caverphone_string)+1)
  nysiis_score = (len(extracted_nysiis_string)-(levenshtein(extracted_nysiis_string,spell_corrected_nysiis_string)))/(len(extracted_nysiis_string)+1)
  return ((0.5*caverphone_score + 0.2*soundex_score + 0.2*metaphone_score + 0.1 * nysiis_score))*100

@app.route('/api/feature_array', methods=['POST'])
@cross_origin()
def display_feature_array():
  try:
    # Get the text data from the request
    request_data = request.get_json()
    extracted_text = request_data.get('text')

    # Calculate the feature array
    feature_array = get_feature_array(extracted_text)

    # Prepare the response
    response = {
      "ok": True,
      "message": "Feature array calculated successfully",
      "feature_array": feature_array
    }

    # Return the response
    return jsonify(response), 200

  except Exception as e:
    # If an error occurs, return an error response
    response = {
      "ok": False,
      "message": f"An error occurred: {str(e)}"
    }
    return jsonify(response), 500

def get_feature_array(extracted_text):
  feature_array = []
  feature_array.append(spelling_accuracy(extracted_text))
  feature_array.append(grammatical_accuracy(extracted_text))
  # feature_array.append(98)
  feature_array.append(percentage_of_corrections(extracted_text))
  feature_array.append(percentage_of_phonetic_accuraccy(extracted_text))
  return feature_array

@app.route('/api/submit_text', methods=['POST'])
@cross_origin()
def submit_text():
  try:
    # Check if the request method is POST
    if request.method != 'POST':
      return jsonify({"ok": False, "message": "Method not allowed"}), 405  # 405 for Method Not Allowed
    
    # Get the JSON data from the request
    request_data = request.get_json()
    
    # Check if the JSON data and 'text' field are present
    if not request_data or 'text' not in request_data:
      return jsonify({"ok": False, "message": "Invalid input"}), 400  # 400 for Bad Request
    
    # Extract the text from the request data
    extracted_text = request_data['text']
    
    # Log the received text
    logging.debug(f"Received text: {extracted_text}")

    # Generate features from the extracted text
    features = get_feature_array(extracted_text)
    features_array = np.array([features])
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(features_array)
    
    # Determine the result based on the prediction
    result = "There's a high chance that this person is suffering from dyslexia or dysgraphia" if prediction[0] == 1 else "There's a very slim chance that this person is suffering from dyslexia or dysgraphia"

    # Log the prediction result
    logging.debug(f"Prediction result: {result}")

    # Construct the response
    response = {
        "ok": True,
        "message": "Score Available",
        "result": result,
    }
    
    # Return the response with status code 200
    return jsonify(response), 200  # 200 for OK

  except Exception as e:
    # Log the error
    logging.error(f"An error occurred: {e}")

    # Return an error response with status code 500
    return jsonify({"ok": False, "message": "Internal Server Error"}), 500  # 500 for Internal Server Error


# for quiz here below
def get_result(lang_vocab, memory, speed, visual, audio, survey):
  array = np.array([[lang_vocab, memory, speed, visual, audio, survey]])
  label = int(quiz_model.predict(array))
  if label == 0:
    output = "There is a high chance of the applicant to have dyslexia."
  elif label == 1:
    output = "There is a moderate chance of the applicant to have dyslexia."
  else:
    output = "There is a low chance of the applicant to have dyslexia."
  return output


@app.route('/api/submit_quiz', methods=['POST'])
def submit_quiz():
  try:
    data = request.json  
    extracted_object = data.get('quiz')
    time_value = data.get('time')

    lang_vocab = (extracted_object['q1'] + extracted_object['q2'] + extracted_object['q3'] + extracted_object['q4'] + extracted_object['q5'] + extracted_object['q6'] + extracted_object['q8'])/28
    memory = (extracted_object['q2']+ extracted_object['q9'])/8
    speed = 1 - (time_value / 60000)
    visual = (extracted_object['q1'] + extracted_object['q3'] + extracted_object['q4'] + extracted_object['q6'])/16
    audio = (extracted_object['q7']+extracted_object['q10'])/8
    survey = (lang_vocab + memory + speed + visual + audio)/80
    
    result = get_result(lang_vocab, memory, speed, visual, audio, survey)

    response = {
        "ok": True,
        "message": "Score Available",
        "result": result
    }
    return jsonify(response)
  except Exception as e:
    logging.error(f"An error occurred: {e}")
    return jsonify({"ok": False, "message": "Internal Server Error"}), 500


if __name__ == "__main__":
  app.run(debug=True,host='0.0.0.0', port=5000)
