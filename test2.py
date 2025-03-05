from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

# Load the Hugging Face model
text_generator = pipeline("text-generation", model="cognitivecomputations/Dolphin3.0-R1-Mistral-24B")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Generate a response
        response = text_generator(user_message, max_length=200, do_sample=True)

        bot_reply = response[0]['generated_text']

        return jsonify({"reply": bot_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets the PORT
    app.run(host="0.0.0.0", port=port, debug=True)
