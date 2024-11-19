from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify, render_template
import json

# Load environment variables from the .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Check for OpenAI API key in environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable is not set.")
else:
    print("OpenAI API key successfully loaded.")

# Example global settings
user_preferences = {
    "tone": "friendly",
    "detail": "concise",
}
conversation_history = [
    {"role": "system", "content": (
        "You are an expert UI/UX design assistant. "
        "Provide concise and professional advice on UI/UX design, Figma, Adobe XD, Sketch, and web development. "
        "Use British English in responses, and focus on directly addressing the user's query."
    )}
]

# Placeholder for external resources
def load_design_library():
    try:
        with open("design_library/components.json") as f:
            components = json.load(f)
        with open("design_library/tutorials.json") as f:
            tutorials = json.load(f)
        with open("design_library/accessibility_guidelines.json") as f:
            accessibility = json.load(f)
        with open("design_library/analytics_insights.json") as f:
            analytics = json.load(f)
        return {
            "components": components,
            "tutorials": tutorials,
            "accessibility": accessibility,
            "analytics": analytics,
        }
    except Exception as e:
        print(f"Error loading design library: {e}")
        return {
            "components": [],
            "tutorials": [],
            "accessibility": [],
            "analytics": [],
        }

design_library = load_design_library()

# Default route to serve the frontend
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Chat endpoint
@app.route("/chat/uiux", methods=["POST"])
def uiux_chat():
    global conversation_history

    try:
        user_message = request.json.get("message", "").strip()
        tone = user_preferences.get("tone", "friendly")
        detail = user_preferences.get("detail", "concise")

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Add the user's message to the conversation history
        conversation_history.append({"role": "user", "content": user_message})

        # Customize system message
        system_message = (
            "You are an expert UI/UX assistant. "
            f"Respond with a {tone} tone and {detail} level of detail. "
            "Provide actionable advice and include relevant examples where necessary."
        )

        # Generate response from model (adjust for OpenAI API or local model)
        try:
            import openai
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use appropriate model
                messages=[{"role": "system", "content": system_message}] + conversation_history
            )
            chatbot_response = response['choices'][0]['message']['content'].strip()
        except ImportError:
            chatbot_response = "Error: OpenAI module not available. Please configure your environment correctly."

        # Append response to history
        conversation_history.append({"role": "assistant", "content": chatbot_response})

        # Limit history
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        return jsonify({"response": chatbot_response})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Preferences endpoint
@app.route("/chat/preferences", methods=["POST"])
def set_preferences():
    global user_preferences
    tone = request.json.get("tone", user_preferences["tone"])
    detail = request.json.get("detail", user_preferences["detail"])

    user_preferences["tone"] = tone
    user_preferences["detail"] = detail

    return jsonify({"message": "Preferences updated.", "preferences": user_preferences})

# Clear conversation history
@app.route("/chat/clear", methods=["POST"])
def clear_conversation():
    global conversation_history
    conversation_history = [
        {"role": "system", "content": (
            "You are an expert UI/UX design assistant. "
            "Provide concise and professional advice on UI/UX design, Figma, Adobe XD, Sketch, and web development."
        )}
    ]
    return jsonify({"message": "Conversation history cleared."})

# Resource endpoints
@app.route("/design/components", methods=["GET"])
def get_components():
    return jsonify({"components": design_library["components"]})

@app.route("/design/tutorials", methods=["GET"])
def get_tutorials():
    return jsonify({"tutorials": design_library["tutorials"]})

@app.route("/design/accessibility", methods=["GET"])
def get_accessibility_guidelines():
    return jsonify({"accessibility_guidelines": design_library["accessibility"]})

@app.route("/design/analytics", methods=["GET"])
def get_analytics_insights():
    return jsonify({"analytics_insights": design_library["analytics"]})

@app.route("/chat/info", methods=["GET"])
def bot_info():
    return jsonify({
        "creator": "Liam Bonello from 6.2A Multi-Media",
        "purpose": "To improve the quality and efficiency of UI/UX and development."
    })

if __name__ == "__main__":
    app.run(debug=True)
