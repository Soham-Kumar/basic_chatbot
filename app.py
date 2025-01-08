from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests for local development

# Load API key from environment variable (recommended)
API_KEY = os.environ.get("NVIDIA_API_KEY")
if not API_KEY:
    print("Warning: NVIDIA_API_KEY not found in environment variables. API calls might fail.")

client = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=API_KEY,
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])

    def generate():
        for chunk in client.stream(messages):
            yield chunk.content

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, port=5000)