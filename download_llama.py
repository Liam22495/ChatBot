from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-13b-hf"

# Hugging Face API will use your token automatically
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
print("Download complete!")
