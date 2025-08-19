# Fashion Chatbot Backend

A Flask-based backend for a fashion chatbot with MongoDB integration and multi-chat functionality.

## Features

- **AI-Powered Chat**: Integration with Google Gemini AI for intelligent fashion recommendations
- **Multi-Chat Support**: Users can create, manage, and switch between multiple chat sessions
- **MongoDB Integration**: Persistent storage for user sessions and chat history
- **Product Catalog**: CSV-based product database with real-time pricing
- **Image Analysis**: Vision AI capabilities for analyzing fashion images
- **Cost Tracking**: Token usage and cost calculation for AI interactions

## Prerequisites

- Python 3.8+
- MongoDB (local or cloud instance)
- Google Gemini API key
- Exchange Rate API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fashion_Chatbot
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MongoDB**
   - Install MongoDB locally or use MongoDB Atlas
   - Create a database named `fashion_bot`
   - Ensure MongoDB is running on `mongodb://localhost:27017/`

4. **Configure environment variables**
   - Copy `config.json` and update with your API keys
   - Or set environment variables:
     ```bash
     export GOOGLE_API_KEY="your_gemini_api_key"
     export EXCHANGE_RATE_API_KEY="your_exchange_rate_api_key"
     export MONGODB_URI="mongodb://localhost:27017/"
     ```

## Configuration

Update `config.json` with your API keys:

```json
{
  "EXCHANGE_RATE_API_KEY": "your_exchange_rate_api_key",
  "GOOGLE_API_KEY": "your_gemini_api_key",
  "MONGODB_URI": "mongodb://localhost:27017/",
  "MONGODB_DATABASE": "fashion_bot"
}
```

## Usage

1. **Start the backend server**
   ```bash
   python FinetunnedBackendCode.py
   ```

2. **Server will run on** `http://localhost:5000`

## API Endpoints

### Chat Endpoints

- `POST /api/chat` - Send a message and get AI response
- `POST /api/history` - Get user chat history

### Multi-Chat Endpoints

- `POST /api/chats` - Create a new chat session
- `GET /api/chats/<user_id>` - Get all chats for a user
- `GET /api/chats/<chat_id>/messages` - Get messages for a specific chat
- `PUT /api/chats/<chat_id>/rename` - Rename a chat session
- `DELETE /api/chats/<chat_id>` - Delete a chat session

### Product Endpoints

- `GET /api/product-details/<product_id>` - Get product details
- `POST /api/product-analysis` - Analyze a product with AI

## Multi-Chat Features

The backend now supports multiple chat sessions per user:

- **Chat Management**: Create, rename, and delete chat sessions
- **Session Isolation**: Each chat maintains its own conversation history
- **User Identification**: Unique user IDs for session management
- **Persistent Storage**: All chat data stored in MongoDB

## Database Schema

### User Sessions Collection (`user_sessions`)
```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "history": [],
  "total_cost_usd": 0.0,
  "total_cost_pkr": 0.0,
  "total_input_tokens": 0,
  "total_output_tokens": 0,
  "total_image_displays": 0,
  "created_at": "datetime",
  "last_updated": "datetime"
}
```

### User Chats Collection (`user_chats`)
```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "chat_name": "string",
  "created_at": "datetime",
  "last_updated": "datetime",
  "messages": []
}
```

## Frontend Integration

The backend is designed to work with the existing React frontend. Update your frontend to:

1. Generate unique user IDs for each session
2. Use the new multi-chat API endpoints
3. Handle chat session management
4. Display chat history and allow switching between chats

## Troubleshooting

- **MongoDB Connection**: Ensure MongoDB is running and accessible
- **API Keys**: Verify your Google Gemini and Exchange Rate API keys are valid
- **Port Conflicts**: Change the port in `FinetunnedBackendCode.py` if 5000 is occupied

## License

This project is licensed under the MIT License.
