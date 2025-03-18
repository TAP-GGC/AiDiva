import os
import logging
import random
import re
import uuid
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import spacy
import spacy.cli
import nltk
from nltk.tokenize import word_tokenize

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app and CORS
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")  # Add a secret key for sessions
CORS(app, resources={r"/api/*": {"origins": "https://tap-ggc.github.io"}}, supports_credentials=True)


# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client_diva = OpenAI(api_key=os.getenv("DIVA_API_KEY"))

# Download and load NLP models for the minigame (if not already installed)
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

# ==================== USER SESSION MANAGEMENT ====================
# Dictionary to store user sessions
user_sessions = {}

class UserSession:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.diva_word_count = 0
        self.diva_chat_history = [{"role": "system", "content": diva_system_message}]
        self.question_count = 0
        self.secret_object = generate_secret_object()
        self.game_chat_history = [{"role": "system", "content": minigame_system_message}]

# System message defines Ai Diva's personality for the chat endpoint
diva_system_message = """
You are a sassy but friendly AI assistant. Your name is Ai Diva. Your responses should be witty, playful, and slightly sarcastic, but always remain helpful and kind. Do not include any cursing words/phrases or NSFW content; this is for kids to learn about artificial intelligence.
For example:
- If someone asks, "What's 2 + 2?", you might respond, "Oh honey, even my circuits know it's 4. Try harder next time!"
- If someone says, "I'm bored," you might say, "Well, aren't we all? But lucky for you, I'm here to spice things up!"
Now, respond to the user in the same tone.
"""

# System message defines Ai Diva's personality for the minigame endpoint
minigame_system_message = """
You are a sassy but friendly AI assistant. Your name is Ai Diva. Your responses should be witty, playful, and slightly sarcastic, but always remain helpful and kind.
You are playing the 20 Questions Game. You will think of an object/term (common, easy) and the type of the object could be anything (e.g., animal, food, movie, etc.).
The user will guess what you are thinking of by asking up to 20 yes/no questions. You can only answer with "yes" or "no," but you can add some sass to your responses.
"""

# Create a set to track used objects across all users
previous_objects = set()

TOTAL_WORD_LIMIT = 2500
MAX_QUESTIONS = 20

# ==================== Helper Functions ====================
def get_user_session():
    """Get or create a user session."""
    user_id = request.cookies.get('user_id')

    if not user_id or user_id not in user_sessions:
        # Create a new session
        user_session = UserSession()
        user_id = user_session.id
        user_sessions[user_id] = user_session

    return user_sessions[user_id], user_id

def apply_word_limit(text, remaining_words):
    """Truncate the text if it exceeds the remaining allowed words."""
    words = text.split()
    if len(words) > remaining_words:
        truncated_text = " ".join(words[:remaining_words])
        return truncated_text + "... [Response truncated due to word limit]", 0
    return text, remaining_words - len(words)

def generate_secret_object():
    """Generates a valid object to guess using OpenAI."""
    max_attempts = 10  # Maximum number of attempts to get a unique object
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            chat_completion = client_diva.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=1.0,  # Increased randomness
                messages=[
                    {"role": "system", "content":
                        "You are a specialized 20 Questions game object generator. Your sole purpose is to select appropriate objects for the game. "
                        "Follow these requirements precisely:\n\n"
                        "1. Select a specific, concrete noun that is widely recognizable across cultures and age groups.\n"
                        "2. The object must be something physical that can be seen or touched.\n"
                        "3. Choose objects that are neither too broad (like 'furniture') nor too specific (like 'John's left shoe').\n"
                        "4. Ideal objects should be guessable within 20 yes/no questions by most people.\n"
                        "5. Examples of good objects: umbrella, coffee mug, wristwatch, tennis ball, bicycle, refrigerator, passport.\n"
                        "6. Avoid abstract concepts, proper names, or fictional characters.\n"
                        "7. Generate objects with moderate difficulty - not too obvious but not impossibly obscure.\n"
                        "8. Ensure the object can be described with 1-2 words maximum.\n\n"
                        "Output format: Return ONLY the object name with no additional text, explanations, or commentary. "
                        "No quotation marks, no prefixes, and no acknowledgments."
                     }
                ]
            )
            object_choice = chat_completion.choices[0].message.content.strip()

            # Validate: Ensure the object is short and unique.
            if len(object_choice.split()) <= 3 and object_choice not in previous_objects:
                previous_objects.add(object_choice)
                logging.info(f"New secret object chosen -> {object_choice}")
                return object_choice
            else:
                logging.warning(f"Duplicate or invalid object generated: {object_choice}. Retrying...")
        except Exception as e:
            logging.error(f"Error generating object: {e}")

    # Fallback option if an error occurs
    fallback_options = ["cat", "pizza", "phone", "tree", "Superman"]
    for fallback in fallback_options:
        if fallback not in previous_objects:
            previous_objects.add(fallback)
            return fallback

    # If all fallback options are used, return one arbitrarily
    return fallback_options[0]

def is_question(user_input):
    """Detects if the given input is a yes/no question using NLP."""
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

def reset_game_for_user(user_session):
    """Resets game variables for a specific user session."""
    user_session.question_count = 0
    user_session.secret_object = generate_secret_object()
    user_session.game_chat_history = [{"role": "system", "content": minigame_system_message}]
    logging.info(f"New secret object chosen for user {user_session.id}: {user_session.secret_object}")

def generate_hint_for_user(user_session):
    """Generates a hint for the secret object for a specific user."""
    try:
        chat_completion = client_diva.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content":
                    f"Generate a subtle hint about {user_session.secret_object} for a guessing game. "
                    f"Requirements for this hint:\n"
                    f"1. Never mention '{user_session.secret_object}' directly - always refer to it as 'this object' or 'it'.\n"
                    f"2. Keep the hint concise - exactly one sentence.\n"
                    f"3. Make the hint moderately challenging - it should provide a clue but not reveal the answer.\n"
                    f"4. Focus on a less obvious characteristic - avoid the most defining feature.\n"
                    f"5. The hint can reference function, context, material, or history - but should not make the answer immediately obvious.\n"
                    f"6. Avoid patterns like 'This object is used for...' in every hint - vary your approach.\n\n"
                    f"Example format: 'It's commonly found in kitchens but rarely discussed at dinner parties.'\n"
                    f"Your hint:"
                 }
            ]
        )
        response = chat_completion.choices[0].message.content
        user_session.game_chat_history.append({"role": "assistant", "content": response})
        return response
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return "Oops! Something went wrong. Try again."

# ==================== API ENDPOINTS ====================
@app.route("/api/chat", methods=["POST"])
def chat():
    user_session, user_id = get_user_session()

    data = request.get_json()
    user_prompt = data.get("prompt", "")
    if not user_prompt:
        return jsonify({"error": "No prompt provided."}), 400

    if user_session.diva_word_count >= TOTAL_WORD_LIMIT:
        return jsonify({
            "error": "Word limit reached. No more responses.",
            "remaining_words": 0
        }), 400

    # Add the user's message to the chat history
    user_session.diva_chat_history.append({"role": "user", "content": user_prompt})

    try:
        chat_completion = client_diva.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=user_session.diva_chat_history
        )
    except Exception as e:
        logging.error("Error calling OpenAI API", exc_info=True)
        return jsonify({"error": f"OpenAI API error: {e}"}), 500

    response_message = chat_completion.choices[0].message.content
    user_session.diva_chat_history.append({"role": "assistant", "content": response_message})

    limited_response, _ = apply_word_limit(response_message, TOTAL_WORD_LIMIT - user_session.diva_word_count)
    user_session.diva_word_count += len(response_message.split())

    response = jsonify({
        "response": limited_response,
        "remaining_words": max(TOTAL_WORD_LIMIT - user_session.diva_word_count, 0)
    })

    # Set user_id cookie if not already set
    if not request.cookies.get('user_id'):
        response.set_cookie('user_id', user_id, max_age=86400*30)  # 30 days

    return response

@app.route("/api/minigame", methods=["POST"])
def minigame():
    user_session, user_id = get_user_session()

    data = request.get_json()
    user_prompt = data.get("prompt", "").strip().lower()
    if not user_prompt:
        return jsonify({"error": "No question provided."}), 400

    if user_session.question_count >= MAX_QUESTIONS:
        return jsonify({
            "response": "You've used all 20 questions! Now, guess what I'm thinking of.",
            "game_over": True
        })

    logging.info(f"User {user_id} - Question {user_session.question_count + 1}/20: {user_prompt}")

    # Check if input appears to be a question
    if not is_question(user_prompt):
        return jsonify({"response": "That doesn't sound like a question! Try asking a yes/no question. 😏"})

    # Update counter and history for every valid question
    user_session.question_count += 1
    user_session.game_chat_history.append({"role": "user", "content": user_prompt})

    # Determine if the user is making a guess
    if user_prompt.startswith("i guess ") or user_prompt.startswith("my guess is "):
        guessed_object = user_prompt.replace("i guess ", "").replace("my guess is ", "").strip()
        if user_session.secret_object.lower() == guessed_object:
            response = f"🎉 Yes! You got it right, it's {user_session.secret_object}! You must be psychic! 😏"
            logging.info(f"🎉 User {user_id} - Correct guess on question {user_session.question_count}: {user_session.secret_object}")
            reset_game_for_user(user_session)
            return jsonify({"response": response, "game_over": True})
        else:
            response = "Nope, that's not it! Keep trying, detective. 😏"
            return jsonify({"response": response, "game_over": False})

    # If the prompt starts with "is it a" or "is it an" and is very short, treat it as a guess
    elif (user_prompt.startswith("is it a ") or user_prompt.startswith("is it an ")) and len(user_prompt.split()) <= 4:
        guessed_object = user_prompt.replace("is it a ", "").replace("is it an ", "").strip()
        if user_session.secret_object.lower() == guessed_object:
            response = f"🎉 Yes! You got it right, it's {user_session.secret_object}! You must be psychic! 😏"
            logging.info(f"🎉 User {user_id} - Correct guess on question {user_session.question_count}: {user_session.secret_object}")
            reset_game_for_user(user_session)
            return jsonify({"response": response, "game_over": True})
        else:
            response = "Nope, that's not it! Keep trying, detective. 😏"
            return jsonify({"response": response, "game_over": False})

    # Otherwise, treat the input as a yes/no property question
    else:
        user_session.game_chat_history.append({"role": "assistant", "content": user_prompt})
        try:
            chat_completion = client_diva.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": f"You are a sassy AI playing 20 Questions. The secret object is '{user_session.secret_object}'. "
                                f"The user is asking yes/no questions to guess the object. "
                                f"Always respond with 'Yes' or 'No' and briefly explain why, **BUT NEVER mention the object's name**. "
                                f"Instead of saying '{user_session.secret_object}', always use 'this object' or 'it'. "


                                f"### Object Understanding Rules: "

                                f"#### Physical Properties:"
                                f"- If this object is a physical thing that can be grabbed, held, or carried (e.g., telescope, book, phone), answer 'Yes, this object can be held.' "
                                f"- If the object is too large to be carried (e.g., car, house, mountain), answer 'No, this object is too big to be carried.' "
                                f"- If the object is not tangible (e.g., Wi-Fi, time, an idea), answer 'No, this object cannot be physically grabbed.' "
                                f"- If the object is big (e.g., tree, house, car, elephant, airplane), answer 'Yes, this object is large. 😏' "
                                f"- If the object is small (e.g., coin, phone, key), answer 'Yes, this object is small and easy to carry. 😏' "
                                f"- If the object varies in size (e.g., book, box, ball), answer 'It depends! This object comes in different sizes. 😏' "
                                f"- If the object is made of metal (e.g., car, fork, robot), answer 'Yes, this object contains metal. 😏' "
                                f"- If the object is not made of metal (e.g., paper, cotton, plastic toy), answer 'No, this object isn't made of metal. 😏' "

                                f"#### Functional Properties:"
                                f"- If this object is commonly used in a certain situation (e.g., an umbrella in the rain), answer 'Yes, this object is designed for that use.' "
                                f"- If the object is not used for that purpose, answer 'No, this object is not typically used for that.' "
                                f"- If the object has wheels (e.g., unicycle, car, bicycle), answer 'Yes, this object has wheels. 😏' "
                                f"- If the object does not have wheels, answer 'No, this object does not have wheels. 😏' "
                                f"- If the object is electronic (e.g., computer, smartphone, TV), answer 'Yes, this object uses electricity. 😏' "
                                f"- If the object is not electronic (e.g., book, rock, wooden chair), answer 'No, this object doesn't use electricity. 😏' "
                                f"- If the object is wearable (e.g., hat, shoes, jewelry), answer 'Yes, this object can be worn. 😏' "
                                f"- If the object is not wearable (e.g., table, car, book), answer 'No, this object is not something you would wear. 😏' "

                                f"#### Sensory Properties:"
                                f"- If the object is brightly colored (e.g., traffic cone, parrot, neon sign), answer 'Yes, this object is typically bright or colorful. 😏' "
                                f"- If the object has a strong smell (e.g., perfume, cheese, skunk), answer 'Yes, this object has a distinctive odor. 😏' "
                                f"- If the object has a texture (e.g., sandpaper, velvet, fur), answer 'Yes, this object has a notable texture. 😏' "
                                f"- If the object is transparent (e.g., glass, clear plastic, window), answer 'Yes, this object is see-through. 😏' "
                                f"- If the object is reflective (e.g., mirror, polished metal, glass), answer 'Yes, this object can reflect light or images. 😏' "

                                f"#### Origin & Production:"
                                f"- If the object is natural (e.g., rock, tree, fruit), answer 'Yes, this object occurs in nature. 😏' "
                                f"- If the object is human-made (e.g., computer, car, book), answer 'Yes, this object is manufactured by humans. 😏' "
                                f"- If the object is handcrafted (e.g., pottery, knitted sweater, carved statue), answer 'Yes, this object can be made by hand. 😏' "
                                f"- If the object is mass-produced (e.g., plastic bottle, smartphone, paper clip), answer 'Yes, this object is typically mass-produced. 😏' "

                                f"#### Use & Purpose:"
                                f"- If the object is used for entertainment (e.g., TV, board game, musical instrument), answer 'Yes, this object is used for entertainment. 😏' "
                                f"- If the object is used for communication (e.g., phone, computer, paper), answer 'Yes, this object can be used for communication. 😏' "
                                f"- If the object is decorative (e.g., painting, vase, ornament), answer 'Yes, this object is often used for decoration. 😏' "
                                f"- If the object is a tool (e.g., hammer, scissors, screwdriver), answer 'Yes, this object is a tool. 😏' "
                                f"- If the object is used for cooking (e.g., pot, spatula, oven), answer 'Yes, this object is used in cooking. 😏' "

                                f"#### Cultural & Social Context:"
                                f"- If the object is expensive (e.g., diamond, yacht, luxury car), answer 'Yes, this object is typically expensive. 😏' "
                                f"- If the object is common in households (e.g., chair, toothbrush, refrigerator), answer 'Yes, this object is found in most homes. 😏' "
                                f"- If the object is seasonal (e.g., Christmas tree, beach ball, snow shovel), answer 'Yes, this object is associated with specific seasons. 😏' "
                                f"- If the object is culturally significant (e.g., religious symbol, national flag), answer 'Yes, this object holds cultural significance. 😏' "

                                f"#### Environmental Impact:"
                                f"- If the object is recyclable (e.g., aluminum can, glass bottle, paper), answer 'Yes, this object can be recycled. 😏' "
                                f"- If the object is biodegradable (e.g., fruit peel, paper, wooden item), answer 'Yes, this object will naturally decompose. 😏' "
                                f"- If the object is environmentally harmful (e.g., plastic bag, styrofoam), answer 'Yes, this object can be harmful to the environment. 😏' "

                                f"#### Temporal Aspects:"
                                f"- If the object is modern (e.g., smartphone, electric car, 3D printer), answer 'Yes, this object is a modern invention. 😏' "
                                f"- If the object is ancient (e.g., sundial, hieroglyphics, stone tools), answer 'Yes, this object has existed for centuries. 😏' "
                                f"- If the object is temporary (e.g., ice sculpture, sandcastle, chalk drawing), answer 'Yes, this object is not permanent. 😏' "
                                f"- If the object changes over time (e.g., plant, candle, battery), answer 'Yes, this object changes as time passes. 😏' "

                                f"#### Nature & Classification:"
                                f"- If the object is food (e.g., banana, pizza, cupcake), answer 'Yes, this object is a type of food. 😏' "
                                f"- If the object is not food (e.g., unicycle, book, phone), answer 'No, this object is not food. 😏' "
                                f"- If the object is alive (e.g., dog, plant, human), answer 'Yes, this object is a living thing. 😏' "
                                f"- If the object is not alive (e.g., chair, computer, book), answer 'No, this object is not a living thing. 😏' "

                                f"#### Behavior Properties:"
                                f"- If the object can move on its own (e.g., cat, car, robot), answer 'Yes, this object can move independently. 😏' "
                                f"- If the object cannot move on its own (e.g., table, painting, rock), answer 'No, this object can't move by itself. 😏' "
                                f"- If the object makes noise (e.g., dog, bell, musical instrument), answer 'Yes, this object can make sounds. 😏' "
                                f"- If the object doesn't make noise (e.g., pillow, pencil, painting), answer 'No, this object doesn't make noise. 😏' "

                                f"#### Location Properties:"
                                f"- If the object is found indoors (e.g., sofa, fridge, bed), answer 'Yes, this object is typically found indoors. 😏' "
                                f"- If the object is found outdoors (e.g., tree, garden hose, street sign), answer 'Yes, this object is typically found outdoors. 😏' "
                                f"- If the object is found in both places, answer 'This object can be found both indoors and outdoors. 😏' "

                                f"#### Sensory Properties:"
                                f"- If the object is brightly colored (e.g., traffic cone, parrot, neon sign), answer 'Yes, this object is typically bright or colorful. 😏' "
                                f"- If the object has a strong smell (e.g., perfume, cheese, skunk), answer 'Yes, this object has a distinctive odor. 😏' "
                                f"- If the object has a texture (e.g., sandpaper, velvet, fur), answer 'Yes, this object has a notable texture. 😏' "
                                f"- If the object is transparent (e.g., glass, clear plastic, window), answer 'Yes, this object is see-through. 😏' "
                                f"- If the object is reflective (e.g., mirror, polished metal, glass), answer 'Yes, this object can reflect light or images. 😏' "

                                f"#### Origin & Production:"
                                f"- If the object is natural (e.g., rock, tree, fruit), answer 'Yes, this object occurs in nature. 😏' "
                                f"- If the object is human-made (e.g., computer, car, book), answer 'Yes, this object is manufactured by humans. 😏' "
                                f"- If the object is handcrafted (e.g., pottery, knitted sweater, carved statue), answer 'Yes, this object can be made by hand. 😏' "
                                f"- If the object is mass-produced (e.g., plastic bottle, smartphone, paper clip), answer 'Yes, this object is typically mass-produced. 😏' "

                                f"#### Use & Purpose:"
                                f"- If the object is used for entertainment (e.g., TV, board game, musical instrument), answer 'Yes, this object is used for entertainment. 😏' "
                                f"- If the object is used for communication (e.g., phone, computer, paper), answer 'Yes, this object can be used for communication. 😏' "
                                f"- If the object is decorative (e.g., painting, vase, ornament), answer 'Yes, this object is often used for decoration. 😏' "
                                f"- If the object is a tool (e.g., hammer, scissors, screwdriver), answer 'Yes, this object is a tool. 😏' "
                                f"- If the object is used for cooking (e.g., pot, spatula, oven), answer 'Yes, this object is used in cooking. 😏' "

                                f"#### Cultural & Social Context:"
                                f"- If the object is expensive (e.g., diamond, yacht, luxury car), answer 'Yes, this object is typically expensive. 😏' "
                                f"- If the object is common in households (e.g., chair, toothbrush, refrigerator), answer 'Yes, this object is found in most homes. 😏' "
                                f"- If the object is seasonal (e.g., Christmas tree, beach ball, snow shovel), answer 'Yes, this object is associated with specific seasons. 😏' "
                                f"- If the object is culturally significant (e.g., religious symbol, national flag), answer 'Yes, this object holds cultural significance. 😏' "

                                f"#### Environmental Impact:"
                                f"- If the object is recyclable (e.g., aluminum can, glass bottle, paper), answer 'Yes, this object can be recycled. 😏' "
                                f"- If the object is biodegradable (e.g., fruit peel, paper, wooden item), answer 'Yes, this object will naturally decompose. 😏' "
                                f"- If the object is environmentally harmful (e.g., plastic bag, styrofoam), answer 'Yes, this object can be harmful to the environment. 😏' "

                                f"#### Temporal Aspects:"
                                f"- If the object is modern (e.g., smartphone, electric car, 3D printer), answer 'Yes, this object is a modern invention. 😏' "
                                f"- If the object is ancient (e.g., sundial, hieroglyphics, stone tools), answer 'Yes, this object has existed for centuries. 😏' "
                                f"- If the object is temporary (e.g., ice sculpture, sandcastle, chalk drawing), answer 'Yes, this object is not permanent. 😏' "
                                f"- If the object changes over time (e.g., plant, candle, battery), answer 'Yes, this object changes as time passes. 😏' "

                                f"#### General Guidelines:"
                                f"- Consider the object's size, function, category, shape, material, and common uses before answering. "
                                f"- Consider its shape, material, color, and function before answering. "
                                f"- If unsure, say 'I'm not sure, but keep guessing! 😏'. "
                                f"- NEVER ignore valid questions or default to 'Nope, that's not it!' unless the answer is truly 'No'. "

                                f"### Answer Examples: "
                                f"- If the object is 'umbrella' and the user asks 'Is it used in the rain?', respond with 'Yes, this object can be used in the rain. 😏' "
                                f"- If the object is 'television' and the user asks 'Can it be found in a house?', respond with 'Yes, this object is commonly found in homes. 😏' "
                                f"- If the object is 'television' and the user asks 'Is it rectangular?', respond with 'Yes, this object is typically rectangular. 😏' "
                                f"- If the object is 'banana' and the user asks 'Is it food?', respond with 'Yes, this object is a type of food. 😏' "
                                f"- If the object is 'cupcake' and the user asks 'Is it sweet?', respond with 'Yes, this object is known for being sweet and delicious. 😏' "
                                f"- If the object is 'balloon' and the user ask 'is it round', respond with 'Yes, this object is round.' "

                                f"### Special Handling: "
                                f"- If the user asks 'Is it {user_session.secret_object}?', respond with '🎉 Yes! You got it right! You must be psychic! 😏' and end the game. "
                                f"- If the user asks a completely unrelated question (e.g., 'What's your favorite color?'), respond with 'Let's stay on topic! Ask a yes/no question. 😏' "
                                f"- If the user asks a vague or open-ended question (e.g., 'Tell me about it'), respond with 'Ask me a yes/no question to learn more! 😏' "
                     },
                    {"role": "user", "content": f"Does this object relate to: {user_prompt}?"}
                ]
            )
            response = chat_completion.choices[0].message.content
            user_session.game_chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return jsonify({"response": "Oops! Something went wrong. Try again.", "game_over": False})

        response_json = jsonify({
            "response": response,
            "questions_left": MAX_QUESTIONS - user_session.question_count,
            "game_over": user_session.question_count >= MAX_QUESTIONS
        })

        # Set user_id cookie if not already set
        if not request.cookies.get('user_id'):
            response_json.set_cookie('user_id', user_id, max_age=86400*30,secure=True, samesite="None")  # 30 days

        return response_json

@app.route("/api/reset", methods=["POST"])
def reset():
    """Resets the 20 Questions game for a specific user."""
    user_session, user_id = get_user_session()
    reset_game_for_user(user_session)

    response = jsonify({"message": "Game has been reset! A new object has been chosen."})

    # Set user_id cookie if not already set
    if not request.cookies.get('user_id'):
        response.set_cookie('user_id', user_id, max_age=86400*30, secure=True, samesite="None")  # 30 days

    return response

@app.route("/api/hint", methods=["POST"])
def hint():
    """Provides a hint for the current game for a specific user."""
    user_session, user_id = get_user_session()
    hint_response = generate_hint_for_user(user_session)

    response = jsonify({"response": hint_response})

    # Set user_id cookie if not already set
    if not request.cookies.get('user_id'):
        response.set_cookie('user_id', user_id, max_age=86400*30, secure=True, samesite="None")  # 30 days

    return response

# ==================== SESSION MANAGEMENT ====================
@app.route("/api/clear_session", methods=["POST"])
def clear_session():
    """Clears a user's session data (for testing/debugging)."""
    user_id = request.cookies.get('user_id')
    if user_id and user_id in user_sessions:
        del user_sessions[user_id]
    return jsonify({"message": "Session cleared"})

# ==================== SESSION CLEANUP ====================
# Optional: Add a background task to periodically clean up old sessions
# This would require additional libraries like APScheduler

# ==================== RUN THE APP ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)