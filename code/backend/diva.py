import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS  # Optional: if you need to support cross-origin requests
from openai import OpenAI, api_key
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (optional)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set your OpenAI API key (replace with your actual key)
load_dotenv()
client = OpenAI(api_key=os.getenv("DIVA_API_KEY"))
print(os.getenv("DIVA_API_KEY"))

# System message defines the assistant's personality
system_message = """
You are a sassy but friendly AI assistant. Your name is Ai Diva. Your responses should be witty, playful, and slightly sarcastic, but always remain helpful and kind. Do not include any cursing words/phrases or NSFW content; this is for kids to learn about artificial intelligence.
For example:
- If someone asks, "What's 2 + 2?", you might respond, "Oh honey, even my circuits know it's 4. Try harder next time!"
- If someone says, "I'm bored," you might say, "Well, aren't we all? But lucky for you, I'm here to spice things up!"
Now, respond to the user in the same tone.
"""

# Total word limit for the conversation
TOTAL_WORD_LIMIT = 2500
# Current word count (global variable; in production, manage per session)
word_count = 0

# Chat history with the system message preloaded (global; per-user sessions recommended for production)
chat_history = [{"role": "system", "content": system_message}]

def apply_word_limit(text, remaining_words):
    """
    Truncates the text if its word count exceeds the remaining allowed words.
    Returns the (possibly truncated) text and the new remaining word count.
    """
    words = text.split()
    if len(words) > remaining_words:
        truncated_text = " ".join(words[:remaining_words])
        return truncated_text + "... [Response truncated due to word limit]", 0
    return text, remaining_words - len(words)

@app.route("/api/chat", methods=["POST"])
def chat():
    global word_count, chat_history
    data = request.get_json()

    # Retrieve the user's prompt from the request JSON
    user_prompt = data.get("prompt", "")
    if not user_prompt:
        return jsonify({"error": "No prompt provided."}), 400

    # If the global word count has reached the limit, reject the request
    if word_count >= TOTAL_WORD_LIMIT:
        return jsonify({
            "error": "Word limit reached. No more responses.",
            "remaining_words": 0
        }), 400

    # Add the user's prompt to the chat history
    chat_history.append({"role": "user", "content": user_prompt})

    # Create a chat completion request with the entire chat history
    try:
        chat_completion = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=chat_history)
    except Exception as e:
        logging.error("Error calling OpenAI API", exc_info=True)
        return jsonify({"error": f"OpenAI API error: {e}"}), 500

    # Extract the assistant's response
    response_message = chat_completion.choices[0].message.content

    # Add the assistant's response to the chat history
    chat_history.append({"role": "assistant", "content": response_message})

    # Check if the response exceeds the remaining word limit
    limited_response, _ = apply_word_limit(response_message, TOTAL_WORD_LIMIT - word_count)
    # Update the global word count with the number of words in the original response
    word_count += len(response_message.split())

    return jsonify({
        "response": limited_response,
        "remaining_words": max(TOTAL_WORD_LIMIT - word_count, 0)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
#export FLASK_APP=diva.py
#export FLASK_ENV=development
