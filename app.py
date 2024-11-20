from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import logging
import os
import requests

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Hugging Face token and environment setup
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")

# GitHub repository details
MODEL_REPO = "https://raw.githubusercontent.com/Liam22495/model_weights/main"
MODEL_DIR = "./model_weights"
MODEL_FILES = {
    "config": f"{MODEL_REPO}/config.json",
    "tokenizer": f"{MODEL_REPO}/tokenizer_config.json",
    "pytorch_model": f"{MODEL_REPO}/pytorch_model.bin",
}

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Function to download files from GitHub
def download_file(url, file_path):
    if not os.path.exists(file_path):
        logging.info(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            logging.info(f"Downloaded {file_path}")
        else:
            raise ValueError(f"Failed to download {url}. HTTP Status: {response.status_code}")

# Download model files if they do not exist
def ensure_model_files():
    for file_name, url in MODEL_FILES.items():
        local_path = os.path.join(MODEL_DIR, file_name.split("/")[-1])
        download_file(url, local_path)

# Ensure the model files are downloaded
ensure_model_files()

# Load Llama 2 model and tokenizer
logging.info("Loading Llama 2 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",
    device_map="auto",
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

# Define Flask routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat/uiux", methods=["POST"])
def uiux_chat():
    global conversation_history
    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_message})
        prompt = "Assistant: " + user_message
        response = model.generate(torch.tensor(tokenizer(prompt)["input_ids"]).to(device))
        assistant_message = tokenizer.decode(response[0], skip_special_tokens=True)

        # Append the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": assistant_message})
        return jsonify({"response": assistant_message})

    except Exception as e:
        logging.error(f"Error during chat: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logging.info(f"Starting server on port {port}...")
    app.run(debug=True, host="0.0.0.0", port=port)
