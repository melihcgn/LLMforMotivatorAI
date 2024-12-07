from flask import Flask, request, jsonify
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer using pipeline
model_path = "./model_files"  # Path to your model files
generator = pipeline("text-generation", 
                     model=GPT2LMHeadModel.from_pretrained(model_path), 
                     tokenizer=GPT2Tokenizer.from_pretrained(model_path))

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    mood = data.get('query')  # Get the mood input

    if not mood:
        return jsonify({"error": "Query is required"}), 400

    print(f"Received query: {mood}")

    # Generate motivational quote
    prompt = f"Motivational quote for someone feeling {mood}:"
    result = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]

    print(f"Generated response: {result}")
    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(debug=True)
