import logging
import os
import random
from pyexpat.errors import messages

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import re
import spacy
import spacy.cli
import nltk
from nltk.tokenize import word_tokenize

spacy.cli.download("en_core_web_sm")

# Load NLP models
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client (Replace with your API key securely)
load_dotenv()
client = OpenAI(api_key=os.getenv("MINIGAME_API_KEY"))

# System message for Ai Diva's personality
system_message = """
You are a sassy but friendly AI assistant. Your name is Ai Diva. Your responses should be witty, playful, and slightly sarcastic, but always remain helpful and kind.
You are playing the 20 Questions Game. You will think of an object/term (common, easy) and the type of the object could be anything (e.g., animal, food, movie, etc.).
The user will guess what you are thinking of by asking up to 20 yes/no questions. You can only answer with "yes" or "no," but you can add some sass to your responses.
"""

# Game variables
question_count = 0
secret_object = ""
chat_history = [{"role": "system", "content": system_message}]
MAX_QUESTIONS = 20


def reset_game():
    """ Resets game variables for a new session """
    global question_count, secret_object, chat_history
    question_count = 0
    secret_object = generate_secret_object()
    chat_history = [{"role": "system", "content": system_message}]

    # Print the new secret object for debugging
    print(f"DEBUG: New secret object chosen -> {secret_object}")
    logging.info(f"New secret object chosen: {secret_object}")


def generate_secret_object():
    """ Uses OpenAI to generate a valid object to guess """
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

            # Validate response: Must not be a generic confirmation
            if object_choice.lower() not in ["alright", "okay", "yes", "no", "sure", "got it"] and len(
                    object_choice.split()) <= 3:
                return object_choice
            else:
                logging.warning(f"Invalid object generated: {object_choice}. Retrying...")
        except Exception as e:
            logging.error(f"Error generating object: {e}")
            return random.choice(["cat", "pizza", "phone", "tree", "Superman"])  # Fallback option


def generate_hint():
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

        chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return jsonify({"response": "Oops! Something went wrong. Try again.", "game_over": False})

    return jsonify({
        "response": response,
    })


def is_question(user_input):
    """Advanced NLP detection to determine if the input is a question"""
    doc = nlp(user_input)

    # Check if the sentence starts with a question word
    question_words = {"who", "what", "when", "where", "why", "how", "is", "does", "do", "can", "could", "would",
                      "should", "will", "are", "was", "were"}
    if doc[0].text.lower() in question_words:
        return True

    # Check if it ends with a question mark
    if user_input.strip().endswith("?"):
        return True

    # Check if the sentence has an auxiliary verb before the subject (common in questions)
    for token in doc:
        if token.dep_ == "aux" and token.head.dep_ == "ROOT":
            return True

    return False


# Start a new game
reset_game()


@app.route("/api/minigame", methods=["POST"])
def minigame():
    """ Handles incoming messages from frontend and returns AI responses """
    global question_count, secret_object

    data = request.get_json()
    user_prompt = data.get("prompt", "").strip().lower()

    if not user_prompt:
        return jsonify({"error": "No question provided."}), 400

    if question_count >= MAX_QUESTIONS:
        return jsonify(
            {"response": "You've used all 20 questions! Now, guess what I'm thinking of.", "game_over": True})

    # Log the question count in the terminal
    logging.info(f"Question {question_count + 1}/20: {user_prompt}")

    #Checks if the user input is a question
    if not is_question(user_prompt):
        return jsonify({"response": "That doesn't sound like a question! Try asking a yes/no question. 😏"})

    # Handle guessing attempts (allowing partial matches)
    guessed_object = user_prompt.replace("is it ", "").replace("i guess ", "").replace("my guess is ", "").strip()

    # Check if the secret object is inside the user's input
    if secret_object.lower() in guessed_object:
        response = f"🎉 Yes! You got it right, it's {secret_object}! You must be psychic! 😏"
        logging.info(f"🎉 Correct guess on question {question_count + 1}: {secret_object}")

        # Reset game BEFORE returning the response
        reset_game()
        return jsonify({"response": response, "game_over": True})

    # If the guess is incorrect but structured as a guess, respond with a denial
    if user_prompt.startswith("is it ") or user_prompt.startswith("i guess ") or user_prompt.startswith("my guess is "):
        response = "Nope, that's not it! Keep trying, detective. 😏"
        return jsonify({"response": response, "game_over": False})

    # Process normal Yes/No Questions
    question_count += 1
    chat_history.append({"role": "user", "content": user_prompt})

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

        chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return jsonify({"response": "Oops! Something went wrong. Try again.", "game_over": False})

    return jsonify({
        "response": response,
        "questions_left": MAX_QUESTIONS - question_count,
        "game_over": question_count >= MAX_QUESTIONS
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """ Resets the game for a new round and logs new secret object """
    reset_game()
    return jsonify({"message": "Game has been reset! A new object has been chosen."})


@app.route("/api/hint", methods=["POST"])
def hint():
    return generate_hint()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)  # Running on port 5001 to avoid conflicts with diva.py

#FLASK_APP=minigame.py
#flask run --port=5001
