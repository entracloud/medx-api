from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

MODEL_NAME = "talhabangyal/mistral-medx"

# Load the model in 4-bit to fit in GPU memory
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True,
    trust_remote_code=True
)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prompt = data.get("prompt", "")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"response": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
