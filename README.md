# Chutes.ai Chatbot

A simple Python-based chatbot using the Chutes.ai API with the Llama-4-Maverick-17B model.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Chutes.ai API token:
```
CHUTES_API_TOKEN=your_api_token_here
```

## Usage

Run the chatbot:
```bash
python chatbot.py
```

- Type your messages and press Enter to chat with the AI
- Type 'quit' to exit the conversation

## Features

- Real-time streaming responses
- Conversation history maintained throughout the session
- Simple command-line interface
- Error handling for API responses

## Requirements

- Python 3.7+
- aiohttp
- asyncio
- python-dotenv 