from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Ensure the backend can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_chatbot import RAGChatbot
from chatbot import ChutesChatbot
import asyncio

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Load both chatbot systems once
rag_bot = RAGChatbot()
llm_bot = ChutesChatbot()

@app.route('/chat/rag', methods=['POST'])
def chat_rag():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    response = asyncio.run(rag_bot.get_response(user_message))
    return jsonify({'response': response})

@app.route('/chat/llm', methods=['POST'])
def chat_llm():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    response = asyncio.run(llm_bot.get_response(user_message))
    return jsonify({'response': response})

@app.route('/chat/blended', methods=['POST'])
def chat_blended():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    # Retrieve RAG context
    relevant_docs = rag_bot.retriever.retrieve_documents(user_message)
    context = rag_bot._format_context(relevant_docs)
    # Compose blended prompt
    blended_prompt = (
        "You are a helpful assistant. Use the following context if relevant, but also use your own knowledge if needed.\n\n" +
        context +
        f"User: {user_message}"
    )
    # Use LLM to answer
    response = asyncio.run(llm_bot.get_response(blended_prompt))
    return jsonify({'response': response})

@app.route('/files', methods=['GET'])
def list_txt_files():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    return jsonify({'files': files})

@app.route('/files/upload', methods=['POST'])
def upload_txt_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.txt'):
        return jsonify({'error': 'Invalid file name'}), 400
    save_path = os.path.join(DATA_DIR, file.filename)
    print(f"Saving uploaded file to: {save_path}")
    file.save(save_path)
    # Optionally, trigger ingestion here
    os.system(f'python {os.path.join(os.path.dirname(__file__), "ingest.py")}')
    rag_bot.retriever.load_artifacts()  # Reload vector store after ingestion
    return jsonify({'success': True, 'filename': file.filename})

@app.route('/files/append', methods=['POST'])
def append_to_txt_file():
    data = request.json
    filename = data.get('filename')
    content = data.get('content')
    if not filename or not content:
        return jsonify({'error': 'Missing filename or content'}), 400
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File does not exist'}), 404
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write('\n' + content)
    # Optionally, trigger ingestion here
    os.system(f'python {os.path.join(os.path.dirname(__file__), "ingest.py")}')
    return jsonify({'success': True, 'filename': filename})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
