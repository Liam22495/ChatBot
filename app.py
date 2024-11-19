from dotenv import load_dotenv
import os
import openai
from flask import Flask, request, jsonify
import time

# Load environment variables from the .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Global variable to store conversation history
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

@app.route("/chat/standard", methods=["POST"])
def standard_chat():
    global conversation_history

    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Add the user's message to the conversation history
        conversation_history.append({"role": "user", "content": user_message})

        start_time = time.time()

        # Generate a response using OpenAI
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=conversation_history
        )

        end_time = time.time()

        # Get the chatbot's response
        chatbot_response = completion['choices'][0]['message']['content'].strip()

        # Add the chatbot's response to the conversation history
        conversation_history.append({"role": "assistant", "content": chatbot_response})

        # Limit history size to 20 messages
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        return jsonify({"response": chatbot_response, "time_taken": end_time - start_time})

    except openai.error.AuthenticationError:
        return jsonify({"error": "Invalid API key. Please check your configuration."}), 401

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/chat/clear", methods=["POST"])
def clear_conversation():
    global conversation_history
    conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
    return jsonify({"message": "Conversation history cleared."})

if __name__ == "__main__":
    app.run(debug=True)
