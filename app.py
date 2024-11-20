from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import logging
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load Hugging Face token from environment
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")

# Load Llama 2 model and tokenizer
logging.info("Loading Llama 2 model...")
model_name = "meta-llama/Llama-2-13b-hf"  # Use "meta-llama/Llama-2-7b-hf" for the smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    offload_folder="./offload",  # Offload layers to CPU if needed
    token=hf_token
)

logging.info("Llama 2 model loaded successfully!")

# Global variables
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

# Load external resources (design library, tutorials, etc.)
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
        logging.error(f"Error loading design library: {e}")
        return {
            "components": [],
            "tutorials": [],
            "accessibility": [],
            "analytics": [],
        }

design_library = load_design_library()

# Truncate input to avoid exceeding model limits
def truncate_input(prompt, max_length=1024):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if tokens.size(1) > max_length:
        return tokenizer.decode(tokens[0, -max_length:], skip_special_tokens=True)
    return prompt

# Generate response using Llama 2
def generate_response(prompt):
    try:
        prompt = truncate_input(prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(inputs["input_ids"], max_length=150, do_sample=True, temperature=0.7)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError:
        return "Error: The model ran out of GPU memory. Try reducing the input size or using a smaller model."
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Default route to serve the frontend
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Chat endpoint for general UI/UX queries
@app.route("/chat/uiux", methods=["POST"])
def uiux_chat():
    global conversation_history

    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        tone = user_preferences.get("tone", "friendly")
        detail = user_preferences.get("detail", "concise")

        # Add user message to history
        conversation_history.append({"role": "user", "content": user_message})

        # Format system message and prompt
        system_message = (
            "You are an expert UI/UX assistant. "
            f"Respond with a {tone} tone and {detail} level of detail. "
            "Provide actionable advice and include relevant examples where necessary."
        )
        prompt = f"{system_message}\n" + "\n".join([f"{h['role']}: {h['content']}" for h in conversation_history]) + "\nAssistant:"

        # Generate chatbot response
        chatbot_response = generate_response(prompt).strip()

        # Add chatbot response to history
        conversation_history.append({"role": "assistant", "content": chatbot_response})
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        return jsonify({"response": chatbot_response})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Endpoint for setting user preferences
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

# Test endpoint for model
@app.route("/chat/test", methods=["GET"])
def test_chat():
    try:
        test_response = generate_response("Test message: What are the best practices in UI/UX design?")
        return jsonify({"response": test_response})
    except Exception as e:
        return jsonify({"error": f"Test failed: {str(e)}"}), 500

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
    # Use PORT from environment or default to 10000
    port = int(os.getenv("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
