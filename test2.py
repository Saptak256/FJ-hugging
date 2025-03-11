from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS  # Enable CORS for React Native

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load model
pipe = pipeline("text-generation", model="cognitivecomputations/Dolphin3.0-R1-Mistral-24B")

@app.route("/chat", methods=["POST"])  # ✅ Match React Native's API endpoint
def chat():
    data = request.json
    user_message = data.get("message", "")  # ✅ Extract user input
    
    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    # Generate AI response
    response = pipe([{"role": "user", "content": user_message}], max_length=100)
    
    # Extract generated text
    ai_reply = response[0]["generated_text"]

    return jsonify({"reply": ai_reply})  # ✅ Match React Native's expected format

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
