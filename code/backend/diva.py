import logging
import os
import random
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import spacy
import spacy.cli
import nltk
from nltk.tokenize import word_tokenize

# Download and load NLP models
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Initialize OpenAI client (choose the appropriate environment variable)
client = OpenAI(api_key=os.getenv("MINIGAME_API_KEY"))  # or use DIVA_API_KEY if needed

# Create Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

#####################################
# 20 Questions Game Configuration  #
#####################################

# Game variables for the 20 Questions game
question_count = 0
secret_object = ""
MAX_QUESTIONS = 20

# Chat history for the 20 Questions game (with a system prompt)
chat_history_game = [{
    "role": "system",
    "content": (
        "You are a sassy but friendly AI assistant named Ai Diva. "
        "You are playing the 20 Questions Game. You will think of a specific, common object "
        "that can be guessed with yes/no questions. Answer strictly with yes/no (and a bit of sass) "
        "without revealing the object's name (instead, refer to it as 'this object' or 'it')."
    )
}]

def reset_game():
    """Resets the game for a new session."""
    global question_count, secret_object, chat_history_game
    question_count = 0
    secret_object = generate_secret_object()
    chat_history_game = [{
        "role": "system",
        "content": (
            "You are a sassy but friendly AI assistant named Ai Diva. "
            "You are playing the 20 Questions Game. You will think of a specific, common object "
            "that can be guessed with yes/no questions. Answer strictly with yes/no (and a bit of sass) "
            "without revealing the object's name (instead, refer to it as 'this object' or 'it')."
        )
    }]
    logging.info(f"New secret object chosen: {secret_object}")

def generate_secret_object():
    """Generates a secret object using the OpenAI API."""
    while True:
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content":
                        "Think of a specific, common object that people can guess in 20 Questions. "
                        "Examples: cat, pizza, Eiffel Tower, bicycle. "
                        "It must be a single, specific noun (1-2 words) that can be guessed with yes/no questions. "
                        "Output only the object name with no extra text."
                     }
                ]
            )
            object_choice = chat_completion.choices[0].message.content.strip()
            if (object_choice.lower() not in ["alright", "okay", "yes", "no", "sure", "got it"]
                    and len(object_choice.split()) <= 3):
                return object_choice
            else:
                logging.warning(f"Invalid object generated: {object_choice}. Retrying...")
        except Exception as e:
            logging.error(f"Error generating object: {e}")
            return random.choice(["cat", "pizza", "phone", "tree", "Superman"])  # Fallback

def generate_hint():
    """Generates a hint for the secret object."""
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content":
                    f"Think of a hint for the secret object without revealing its name. "
                    f"Always refer to the object as 'this object' or 'it'. Make the hint one sentence and not too obvious."
                 }
            ]
        )
        response = chat_completion.choices[0].message.content
        chat_history_game.append({"role": "assistant", "content": response})
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return jsonify({"response": "Oops! Something went wrong. Try again.", "game_over": False})
    return jsonify({"response": response})

def is_question(user_input):
    """Determines if the user input sounds like a question."""
    doc = nlp(user_input)
    question_words = {"who", "what", "when", "where", "why", "how", "is", "does", "do", "can", "could", "would", "should", "will", "are", "was", "were"}
    if doc[0].text.lower() in question_words:
        return True
    if user_input.strip().endswith("?"):
        return True
    for token in doc:
        if token.dep_ == "aux" and token.head.dep_ == "ROOT":
            return True
    return False

# Initialize the game when the app starts
reset_game()

#############################
# General Chat Configuration#
#############################

TOTAL_WORD_LIMIT = 2500
word_count = 0

chat_history_chat = [{
    "role": "system",
    "content": (
        "You are a sassy but friendly AI assistant named Ai Diva. "
        "Answer the user's questions with wit and kindness."
    )
}]

def apply_word_limit(text, remaining_words):
    """Truncates the text if it exceeds the remaining word limit."""
    words = text.split()
    if len(words) > remaining_words:
        truncated_text = " ".join(words[:remaining_words])
        return truncated_text + "... [Response truncated due to word limit]", 0
    return text, remaining_words - len(words)

####################
# API Endpoints    #
####################

# 20 Questions Game Endpoint
@app.route("/api/minigame", methods=["POST"])
def minigame():
    global question_count, secret_object, chat_history_game
    data = request.get_json()
    user_prompt = data.get("prompt", "").strip().lower()
    if not user_prompt:
        return jsonify({"error": "No question provided."}), 400

    if question_count >= MAX_QUESTIONS:
        return jsonify({"response": "You've used all 20 questions! Now, guess what I'm thinking of.", "game_over": True})

    logging.info(f"Question {question_count + 1}/20: {user_prompt}")

    if not is_question(user_prompt):
        return jsonify({"response": "That doesn't sound like a question! Try asking a yes/no question. 😏"})

    guessed_object = user_prompt.replace("is it ", "").replace("i guess ", "").replace("my guess is ", "").strip()

    if secret_object.lower() in guessed_object:
        response = f"🎉 Yes! You got it right, it's {secret_object}! You must be psychic! 😏"
        logging.info(f"Correct guess: {secret_object}")
        reset_game()
        return jsonify({"response": response, "game_over": True})

    if user_prompt.startswith("is it ") or user_prompt.startswith("i guess ") or user_prompt.startswith("my guess is "):
        response = "Nope, that's not it! Keep trying, detective. 😏"
        return jsonify({"response": response, "game_over": False})

    question_count += 1
    chat_history_game.append({"role": "user", "content": user_prompt})

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": (
                     f"You are a sassy AI playing 20 Questions. The secret object is '{secret_object}'. "
                     f"Answer yes/no questions without revealing the object's name. "
                     f"Use 'this object' instead of the actual name."
                 )
                 },
                {"role": "user", "content": f"Does this object relate to: {user_prompt}?"}
            ]
        )
        response = chat_completion.choices[0].message.content
        chat_history_game.append({"role": "assistant", "content": response})
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return jsonify({"response": "Oops! Something went wrong. Try again.", "game_over": False})

    return jsonify({
        "response": response,
        "questions_left": MAX_QUESTIONS - question_count,
        "game_over": question_count >= MAX_QUESTIONS
    })

# Endpoint to reset the game
@app.route("/api/reset", methods=["POST"])
def reset():
    reset_game()
    return jsonify({"message": "Game has been reset! A new object has been chosen."})

# Endpoint to get a hint
@app.route("/api/hint", methods=["POST"])
def hint():
    return generate_hint()

# General Chat Endpoint
@app.route("/api/chat", methods=["POST"])
def chat():
    global word_count, chat_history_chat
    data = request.get_json()
    user_prompt = data.get("prompt", "")
    if not user_prompt:
        return jsonify({"error": "No prompt provided."}), 400
    if word_count >= TOTAL_WORD_LIMIT:
        return jsonify({
            "error": "Word limit reached. No more responses.",
            "remaining_words": 0
        }), 400
    chat_history_chat.append({"role": "user", "content": user_prompt})
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history_chat
        )
    except Exception as e:
        logging.error("Error calling OpenAI API", exc_info=True)
        return jsonify({"error": f"OpenAI API error: {e}"}), 500
    response_message = chat_completion.choices[0].message.content
    chat_history_chat.append({"role": "assistant", "content": response_message})
    limited_response, _ = apply_word_limit(response_message, TOTAL_WORD_LIMIT - word_count)
    word_count += len(response_message.split())
    return jsonify({
        "response": limited_response,
        "remaining_words": max(TOTAL_WORD_LIMIT - word_count, 0)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
