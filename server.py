from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import joblib
import uvicorn
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load('models/disease_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

class Symptoms(BaseModel):
    symptoms: str

# HTML Interface
HTML_CONTENT = '''<!DOCTYPE html>
<html>
<head>
    <title>MediBot - AI Medical Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chat-container {
            width: 90%;
            max-width: 500px;
            height: 90vh;
            background: #efeae2;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .chat-header {
            background: #075e54;
            color: white;
            padding: 15px 20px;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .message {
            max-width: 70%;
            padding: 10px 15px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        .user-message {
            background: #dcf8c5;
            align-self: flex-end;
        }
        .bot-message {
            background: white;
            align-self: flex-start;
        }
        .message-time {
            font-size: 10px;
            color: #666;
            margin-top: 5px;
        }
        .input-area {
            background: #f0f0f0;
            padding: 15px;
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 25px;
            outline: none;
        }
        .input-area button {
            background: #075e54;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
        }
        .disclaimer {
            font-size: 9px;
            text-align: center;
            padding: 8px;
            background: #f0f0f0;
            color: #888;
        }
        .typing {
            background: white;
            padding: 10px 15px;
            border-radius: 18px;
            align-self: flex-start;
            display: none;
        }
        .typing span {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #777;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-10px); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h2>🏥 MediBot AI</h2>
            <p>Medical Assistant <span style="color:#25d366;">● Online</span></p>
        </div>
        <div class="messages" id="messages">
            <div class="message bot-message">
                👋 Hello! I'm MediBot, your AI medical assistant.<br><br>
                Tell me your symptoms, and I'll help predict possible conditions.<br><br>
                <strong>Example:</strong> "fever, headache, body ache"<br><br>
                ⚠️ <em>AI prediction only — Consult a real doctor!</em>
                <div class="message-time">Just now</div>
            </div>
        </div>
        <div class="typing" id="typing">
            <span></span><span></span><span></span>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Type your symptoms..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send ➤</button>
        </div>
        <div class="disclaimer">⚠️ AI Prediction only — Not a substitute for professional medical advice</div>
    </div>
    <script>
        function handleKeyPress(event) {
            if (event.key === 'Enter') sendMessage();
        }
        function addMessage(text, isUser) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
            const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            messageDiv.innerHTML = text + '<div class="message-time">' + time + '</div>';
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        function showTyping() { document.getElementById('typing').style.display = 'block'; }
        function hideTyping() { document.getElementById('typing').style.display = 'none'; }
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const symptoms = input.value.trim();
            if (!symptoms) return;
            addMessage(symptoms, true);
            input.value = '';
            showTyping();
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symptoms: symptoms })
                });
                const data = await response.json();
                hideTyping();
                let botResponse = '🤖 <strong>Predicted:</strong> ' + data.predicted_disease + '<br><br>';
                botResponse += '📊 <strong>Confidence:</strong> ' + data.confidence + '<br><br>';
                botResponse += '💊 <strong>Advice:</strong> ' + data.advice + '<br><br>';
                botResponse += '⚠️ ' + data.disclaimer;
                addMessage(botResponse, false);
            } catch (error) {
                hideTyping();
                addMessage('❌ Server error. Please make sure backend is running.', false);
            }
        }
    </script>
</body>
</html>'''

@app.get("/")
def home():
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(symptoms: Symptoms):
    vec = vectorizer.transform([symptoms.symptoms.lower()])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    confidence = max(probs) * 100
    return {
        "symptoms": symptoms.symptoms,
        "predicted_disease": pred,
        "confidence": f"{confidence:.2f}%",
        "advice": "Please consult a doctor for proper diagnosis.",
        "disclaimer": "AI prediction only — consult real doctor"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    