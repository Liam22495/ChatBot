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

# Flask endpoint remains the same
@app.route("/chat/standard", methods=["POST"])
def standard_chat():
    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        start_time = time.time()

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        )

        end_time = time.time()

        chatbot_response = completion['choices'][0]['message']['content'].strip()

        return jsonify({"response": chatbot_response, "time_taken": end_time - start_time})

    except openai.error.AuthenticationError:
        return jsonify({"error": "Invalid API key. Please check your configuration."}), 401

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
