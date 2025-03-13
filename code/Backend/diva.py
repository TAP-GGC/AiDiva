import logging
import os
import random

import nltk
import spacy.cli
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from openai import OpenAI

# Download and load NLP models
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Initialize OpenAI client (choose the appropriate environment variable)
client = OpenAI(api_key=os.getenv("MINIGAME_API_KEY"))

# Create Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key_for_dev")

#####################################
# 20 Questions Game Configuration  #
#####################################

# System message for Ai Diva's personality
system_message_minigame = """
You are a sassy but friendly AI assistant. Your name is Ai Diva. Your responses should be witty, playful, and slightly sarcastic, but always remain helpful and kind.
You are playing the 20 Questions Game. You will think of an object/term (common, easy) and the type of the object could be anything (e.g., animal, food, movie, etc.).
The user will guess what you are thinking of by asking up to 20 yes/no questions. You can only answer with "yes" or "no," but you can add some sass to your responses.
"""
# Game variables for the 20 Questions game
question_count = 0
secret_object = ""
MAX_QUESTIONS = 20

# Chat history for the 20 Questions game (with a system prompt)
chat_history_game = [{"role": "system", "content": system_message_minigame}]

def reset_game():
    """Resets the game for a new session."""
    global question_count, secret_object, chat_history_game
    question_count = 0
    secret_object = generate_secret_object()
    chat_history_game = [{"role": "system","content": system_message_minigame}]
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
                        "Do not repeat objects"
                        "Do not return words like 'got it' or 'okay'. Just output the object name directly with no extra text."
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
                    f"think of a hint for the {secret_object}"
                    f"Instead of saying '{secret_object}', always use 'this object' or 'it'."
                    f"Make the hit simple, one sentence at most"
                    f"Do not make the hint obvious, the user should not be able to guess what it is directly from the hint"
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

# System message defines the assistant's personality
system_message_chat = """
You are a sassy but friendly AI assistant. Your name is Ai Diva. Your responses should be witty, playful, and slightly sarcastic, but always remain helpful and kind. Do not include any cursing words/phrases or NSFW content; this is for kids to learn about artificial intelligence.
For example:
- If someone asks, "What's 2 + 2?", you might respond, "Oh honey, even my circuits know it's 4. Try harder next time!"
- If someone says, "I'm bored," you might say, "Well, aren't we all? But lucky for you, I'm here to spice things up!"
Now, respond to the user in the same tone.
"""

TOTAL_WORD_LIMIT = 2500
word_count = 0

chat_history_chat = [{"role": "system", "content": system_message_chat}]

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
    question_count = session.get('question_count', 0)
    secret_object = session.get('secret_object', None)
    chat_history_game = session.get('chat_history_game', [])
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

    # If the guess is incorrect but structured as a guess, respond with a denial
    if user_prompt.startswith("is it ") or user_prompt.startswith("i guess ") or user_prompt.startswith("my guess is "):
        response = "Nope, that's not it! Keep trying, detective. 😏"
        return jsonify({"response": response, "game_over": False})

    if secret_object or secret_object.lower() in user_prompt:
        response = f"🎉 Yes! You got it right, it's {secret_object}!"
        # Optionally, clear the game state
        session.pop('secret_object', None)
        session.pop('question_count', None)
        session.pop('chat_history_game', None)
        return jsonify({"response": response, "game_over": True})


    if user_prompt.startswith("is it ") or user_prompt.startswith("i guess ") or user_prompt.startswith("my guess is "):
        response = "Nope, that's not it! Keep trying, detective. 😏"
        return jsonify({"response": response, "game_over": False})

    question_count += 1
    session['question_count'] = question_count
    chat_history_game.append({"role": "user", "content": user_prompt})
    session['chat_history_game'] = chat_history_game

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"You are a sassy AI playing 20 Questions. The secret object is '{secret_object}'. "
                            f"The user is asking yes/no questions to guess the object. "
                            f"Always respond with 'Yes' or 'No' and briefly explain why, **BUT NEVER mention the object's name**. "
                            f"Instead of saying '{secret_object}', always use 'this object' or 'it'. "


                            f"### Object Understanding Rules: "
                            f"- If this object is a **physical thing** that can be grabbed, held, or carried (e.g., telescope, book, phone), answer 'Yes, this object can be held.' "
                            f"- If the object is **too large** to be carried (e.g., car, house, mountain), answer 'No, this object is too big to be carried.' "
                            f"- If the object is **not tangible** (e.g., Wi-Fi, time, an idea), answer 'No, this object cannot be physically grabbed.' "
                            f"- Consider the object's **size, function, category, shape, material, and common uses** before answering. "
                            f"- If the object is **big** (e.g., tree, house, car, elephant, airplane), answer 'Yes, this object is large. 😏' "
                            f"- If the object is **small** (e.g., coin, phone, key), answer 'Yes, this object is small and easy to carry. 😏' "
                            f"- If the object **varies in size** (e.g., book, box, ball), answer 'It depends! This object comes in different sizes. 😏' "
                            f"- If this object is **commonly used in a certain situation** (e.g., an umbrella in the rain), answer 'Yes, this object is designed for that use.' "
                            f"- If the object is **not used for that purpose**, answer 'No, this object is not typically used for that.' "
                            f"- If the object is **food** (e.g., banana, pizza, cupcake), answer 'Yes, this object is a type of food. 😏' "
                            f"- If the object is **not food** (e.g., unicycle, book, phone), answer 'No, this object is not food. 😏' "
                            f"- If the object **has wheels** (e.g., unicycle, car, bicycle), answer 'Yes, this object has wheels. 😏' "
                            f"- If the object **does not have wheels**, answer 'No, this object does not have wheels. 😏' "
                            f"- Consider its shape, material, color, and function before answering. "
                            f"- If unsure, say 'I’m not sure, but keep guessing! 😏'. "
                            f"- NEVER ignore valid questions or default to 'Nope, that's not it!' unless the answer is truly 'No'. "

                            f"### Answer Examples: "
                            f"- If the object is 'umbrella' and the user asks 'Is it used in the rain?', respond with 'Yes, this object can be used in the rain. 😏' "
                            f"- If the object is 'television' and the user asks 'Can it be found in a house?', respond with 'Yes, this object is commonly found in homes. 😏' "
                            f"- If the object is 'television' and the user asks 'Is it rectangular?', respond with 'Yes, this object is typically rectangular. 😏' "
                            f"- If the object is 'banana' and the user asks 'Is it food?', respond with 'Yes, this object is a type of food. 😏' "
                            f"- If the object is 'cupcake' and the user asks 'Is it sweet?', respond with 'Yes, this object is known for being sweet and delicious. 😏' "
                            f"- If the object is 'balloon' and the user ask 'is it round', respond with 'Yes, this object is round."

                            f"### Special Handling: "
                            f"- If the user asks 'Is it {secret_object}?', respond with '🎉 Yes! You got it right! You must be psychic! 😏' and end the game."
                            f"- If the user asks a completely unrelated question (e.g., 'What’s your favorite color?'), respond with 'Let's stay on topic! Ask a yes/no question. 😏' "
                            f"- If the user asks a vague or open-ended question (e.g., 'Tell me about it'), respond with 'Ask me a yes/no question to learn more! 😏' "

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
