from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from dotenv import load_dotenv
import torch
import logging
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Load Hugging Face token from environment
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    logging.error("HUGGINGFACE_TOKEN environment variable is not set.")
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")

# Load Llama 2 model and tokenizer
try:
    logging.info("Loading Llama 2 model and tokenizer...")
    model_name = "meta-llama/Llama-2-13b-hf"  # Adjust the path to your pre-downloaded model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

    # Automatic device allocation
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="float16" if torch.cuda.is_available() else "float32",  # Use half-precision on GPU
        device_map="auto",  # Automatically split layers between GPU and CPU
        offload_folder="./offload",  # Disk storage for unused layers
        use_auth_token=hf_token
    )

    # Debug: Check how layers are allocated
    logging.info(f"Device map: {model.hf_device_map}")
    logging.info("Llama 2 model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the Llama 2 model or tokenizer.")

# Global variables
conversation_history = [
    {"role": "system", "content": (
        "You are an expert UI/UX design assistant. "
        "Provide concise and professional advice on UI/UX design, Figma, Adobe XD, Sketch, and web development. "
        "Use British English in responses, and focus on directly addressing the user's query."
    )}
]

# Truncate input to avoid exceeding model limits
def truncate_input(prompt, max_length=512):
    try:
        logging.info(f"Truncating input if it exceeds {max_length} tokens...")
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
        if tokens.size(1) > max_length:
            truncated = tokenizer.decode(tokens[0, -max_length:], skip_special_tokens=True)
            logging.warning(f"Input truncated. Original: {prompt} | Truncated: {truncated}")
            return truncated
        logging.info("Input does not require truncation.")
        return prompt
    except Exception as e:
        logging.error(f"Error truncating input: {e}")
        raise

# Generate response using Llama 2
def generate_response(prompt):
    try:
        logging.info("Generating response...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Set generation configuration
        generation_config = GenerationConfig(
            max_new_tokens=150,  # Limit response length
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

        outputs = model.generate(
            inputs["input_ids"],
            generation_config=generation_config
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Response generated: {response}")
        return response
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA Out of Memory error.")
        return "Error: The model ran out of GPU memory. Try reducing the input size or using a smaller model."
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

# Default route to serve the frontend
@app.route("/", methods=["GET"])
def home():
    logging.info("Serving index.html to the client.")
    return render_template("index.html")

# Chat endpoint for general UI/UX queries
@app.route("/chat/uiux", methods=["POST"])
def uiux_chat():
    global conversation_history

    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            logging.warning("Empty message received")
            return jsonify({"error": "Message cannot be empty"}), 400

        logging.info(f"User message received: {user_message}")
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_message})

        # Format system message and prompt
        system_message = (
            "You are an expert UI/UX assistant. "
            "Provide actionable advice and include relevant examples where necessary."
        )
        prompt = f"{system_message}\n" + "\n".join(
            [f"{h['role']}: {h['content']}" for h in conversation_history]
        ) + "\nAssistant:"

        logging.info("Formatted prompt for model:")
        logging.info(prompt)

        # Generate chatbot response
        chatbot_response = generate_response(prompt).strip()

        # Add chatbot response to history
        conversation_history.append({"role": "assistant", "content": chatbot_response})
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        
        logging.info(f"Chatbot response sent to client: {chatbot_response}")
        return jsonify({"response": chatbot_response})

    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    port = 8000  # Fixed port for local usage
    logging.info(f"Starting Flask app on port {port}...")
    app.run(debug=True, host="127.0.0.1", port=port, use_reloader=False)
