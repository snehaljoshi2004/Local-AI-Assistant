from flask import Flask, render_template_string, request, jsonify, Response
import requests
import json
from datetime import datetime
import time

app = Flask(__name__)

class LLMWebUI:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.current_model = "phi3:mini"  # phi3 is more stable
        self.conversation_history = []
        
    def get_available_models(self):
        """Get list of installed models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
        except:
            pass
        return ["phi3:mini", "llama3.2:3b", "mistral"]
    
    def switch_model(self, model_name):
        """Switch to different model"""
        self.current_model = model_name
        return f"Switched to model: {model_name}"
    
    def generate_response_stream(self, message, temperature=0.7, max_tokens=500):
        """Generate streaming response from model with anti-repetition settings"""
        
        # Build conversation context
        prompt = ""
        for user_msg, assistant_msg in self.conversation_history[-5:]:
            prompt += f"User: {user_msg}\n"
            if assistant_msg:
                prompt += f"Assistant: {assistant_msg}\n"
        prompt += f"User: {message}\nAssistant: "
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.current_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "repeat_penalty": 1.1,  # Prevents repetition
                        "repeat_last_n": 64,     # Look back for repeats
                        "top_k": 40,             # Limit token choices
                        "top_p": 0.9,            # Nucleus sampling
                        "stop": ["User:", "\nUser"]  # Stop at new user message
                    }
                },
                stream=True,
                timeout=120
            )
            
            full_response = ""
            last_response = ""
            repeat_count = 0
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        chunk = data["response"]
                        
                        # Simple repetition detection
                        if chunk == last_response:
                            repeat_count += 1
                            if repeat_count > 3:  # Stop if repeating same character
                                break
                        else:
                            repeat_count = 0
                            last_response = chunk
                        
                        full_response += chunk
                        yield full_response
                        
            # Add to history after complete response
            if full_response and not full_response.isspace():
                self.conversation_history.append([message, full_response])
                # Keep only last 10 exchanges
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
            
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation cleared"
    
    def benchmark_model(self, model_name):
        """Run quick benchmark on selected model"""
        test_prompt = "What is artificial intelligence? Answer in 2-3 sentences."
        
        try:
            start_time = time.time()
            first_token_time = None
            token_count = 0
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": test_prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0,
                        "repeat_penalty": 1.1
                    }
                },
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    data = json.loads(line)
                    if "response" in data:
                        token_count += 1
            
            total_time = time.time() - start_time
            tokens_per_sec = token_count / total_time if total_time > 0 else 0
            
            return {
                "tokens_per_second": round(tokens_per_sec, 2),
                "ttft_ms": round(first_token_time * 1000, 2) if first_token_time else 0,
                "total_time": round(total_time, 2),
                "tokens": token_count
            }
        except Exception as e:
            return {"error": str(e)}

# Initialize the LLM interface
llm = LLMWebUI()

# HTML Template with better UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: #0f3460;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #e94560 0%, #533483 100%);
            color: white;
            padding: 20px 30px;
        }
        
        .header h1 {
            font-size: 28px;
            margin-bottom: 5px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .controls {
            padding: 20px 30px;
            background: #1a1a2e;
            border-bottom: 1px solid #16213e;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .control-group label {
            font-weight: 600;
            color: #e0e0e0;
        }
        
        select, input {
            padding: 8px 12px;
            border: 1px solid #533483;
            border-radius: 8px;
            font-size: 14px;
            background: #16213e;
            color: white;
        }
        
        button {
            padding: 8px 16px;
            background: #e94560;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        button:hover {
            background: #c73e56;
            transform: translateY(-2px);
        }
        
        button.danger {
            background: #dc2626;
        }
        
        button.danger:hover {
            background: #b91c1c;
        }
        
        .chat-container {
            height: 500px;
            overflow-y: auto;
            padding: 20px 30px;
            background: #16213e;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .user .message-content {
            background: #e94560;
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .assistant .message-content {
            background: #0f3460;
            color: #e0e0e0;
            border: 1px solid #533483;
            border-bottom-left-radius: 4px;
        }
        
        .input-area {
            padding: 20px 30px;
            background: #1a1a2e;
            border-top: 1px solid #16213e;
            display: flex;
            gap: 10px;
        }
        
        #message-input {
            flex: 1;
            padding: 12px;
            border: 1px solid #533483;
            border-radius: 10px;
            font-size: 14px;
            resize: none;
            font-family: inherit;
            background: #16213e;
            color: white;
        }
        
        #message-input::placeholder {
            color: #888;
        }
        
        #send-btn {
            padding: 12px 24px;
            font-size: 16px;
        }
        
        .status {
            padding: 10px 30px;
            background: #0f3460;
            color: #e94560;
            font-size: 12px;
            border-top: 1px solid #16213e;
        }
        
        .typing-indicator {
            display: inline-block;
            padding: 12px 18px;
            background: #0f3460;
            border: 1px solid #533483;
            border-radius: 18px;
            border-bottom-left-radius: 4px;
        }
        
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #e94560;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }
        
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        .info-box {
            background: #0f3460;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            font-size: 12px;
            color: #aaa;
            border-left: 3px solid #e94560;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Local AI Assistant</h1>
            <p>Fully Offline • Multiple Models • Real-time Streaming</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>Model:</label>
                <select id="model-select">
                    {% for model in models %}
                    <option value="{{ model }}" {% if model == current_model %}selected{% endif %}>{{ model }}</option>
                    {% endfor %}
                </select>
            </div>
            
            <div class="control-group">
                <label>Temperature:</label>
                <input type="range" id="temperature" min="0" max="1" step="0.05" value="0.7">
                <span id="temp-value" style="color:white;">0.70</span>
            </div>
            
            <div class="control-group">
                <label>Max Tokens:</label>
                <input type="number" id="max-tokens" min="50" max="1000" value="300" step="50">
            </div>
            
            <button id="benchmark-btn">Benchmark</button>
            <button id="clear-btn" class="danger"> Clear</button>
        </div>
        
        <div class="chat-container" id="chat-container">
            <div class="message assistant">
                <div class="message-content">
                    👋 Hello! I'm your local AI assistant.<br><br>
                    <strong>Note about weather questions:</strong> I run entirely offline, so I don't have access to real-time weather data. I can help with general information, coding, creative writing, reasoning, and more!<br><br>
                    Try asking me about:
                    <ul style="margin-top: 8px;">
                        <li>Science facts (speed of light, gravity, etc.)</li>
                        <li>Code examples</li>
                        <li>Creative writing</li>
                        <li>Problem solving</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="input-area">
            <textarea id="message-input" rows="2" placeholder="Type your message here..."></textarea>
            <button id="send-btn">Send</button>
        </div>
        
        <div class="status" id="status">
             Ready | Model: {{ current_model }}
        </div>
    </div>
    
    <script>
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');
        const clearBtn = document.getElementById('clear-btn');
        const modelSelect = document.getElementById('model-select');
        const temperatureSlider = document.getElementById('temperature');
        const tempValue = document.getElementById('temp-value');
        const maxTokensInput = document.getElementById('max-tokens');
        const benchmarkBtn = document.getElementById('benchmark-btn');
        const statusDiv = document.getElementById('status');
        
        temperatureSlider.addEventListener('input', function() {
            tempValue.textContent = this.value;
        });
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            messageInput.value = '';
            
            // Add typing indicator
            const typingId = addTypingIndicator();
            
            try {
                // Get streaming response
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: message,
                        temperature: parseFloat(temperatureSlider.value),
                        max_tokens: parseInt(maxTokensInput.value)
                    })
                });
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantMessage = '';
                let messageDiv = null;
                
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') continue;
                            
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.response) {
                                    assistantMessage = parsed.response;
                                    
                                    if (!messageDiv) {
                                        removeTypingIndicator(typingId);
                                        messageDiv = addMessage(assistantMessage, 'assistant');
                                    } else {
                                        updateMessage(messageDiv, assistantMessage);
                                    }
                                }
                            } catch (e) {}
                        }
                    }
                }
                
                if (!messageDiv) {
                    removeTypingIndicator(typingId);
                }
            } catch (error) {
                removeTypingIndicator(typingId);
                addMessage('Error: Could not connect to server', 'assistant');
            }
            
            scrollToBottom();
        }
        
        function addMessage(text, role) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = text;
            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);
            scrollToBottom();
            return contentDiv;
        }
        
        function updateMessage(element, text) {
            element.textContent = text;
            scrollToBottom();
        }
        
        function addTypingIndicator() {
            const id = 'typing-' + Date.now();
            const typingDiv = document.createElement('div');
            typingDiv.id = id;
            typingDiv.className = 'message assistant';
            typingDiv.innerHTML = `
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            `;
            chatContainer.appendChild(typingDiv);
            scrollToBottom();
            return id;
        }
        
        function removeTypingIndicator(id) {
            const element = document.getElementById(id);
            if (element) element.remove();
        }
        
        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        async function clearChat() {
            await fetch('/clear', {method: 'POST'});
            chatContainer.innerHTML = '';
            addMessage('Conversation cleared. How can I help you?', 'assistant');
            statusDiv.textContent = ' Conversation cleared';
            setTimeout(() => {
                statusDiv.textContent = ' Ready | Model: ' + modelSelect.value;
            }, 2000);
        }
        
        async function switchModel() {
            const model = modelSelect.value;
            const response = await fetch('/switch_model', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: model})
            });
            const data = await response.json();
            statusDiv.textContent = ` ${data.status}`;
            setTimeout(() => {
                statusDiv.textContent = ' Ready | Model: ' + model;
            }, 2000);
        }
        
        async function runBenchmark() {
            const model = modelSelect.value;
            statusDiv.textContent = ' Running benchmark...';
            benchmarkBtn.disabled = true;
            
            const response = await fetch('/benchmark', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: model})
            });
            const data = await response.json();
            
            if (data.error) {
                statusDiv.textContent = `Benchmark failed: ${data.error}`;
            } else {
                statusDiv.innerHTML = ` Benchmark: ${data.tokens_per_second} tok/s | TTFT: ${data.ttft_ms}ms | Total: ${data.total_time}s`;
                addMessage(`**Benchmark Results for ${model}:**\n- Tokens/Second: ${data.tokens_per_second}\n- Time to First Token: ${data.ttft_ms}ms\n- Total Time: ${data.total_time}s\n- Tokens Generated: ${data.tokens}`, 'assistant');
            }
            
            benchmarkBtn.disabled = false;
            setTimeout(() => {
                statusDiv.textContent = ' Ready | Model: ' + modelSelect.value;
            }, 5000);
        }
        
        sendBtn.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        clearBtn.addEventListener('click', clearChat);
        modelSelect.addEventListener('change', switchModel);
        benchmarkBtn.addEventListener('click', runBenchmark);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        models=llm.get_available_models(),
        current_model=llm.current_model
    )

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 300)
    
    def generate():
        for response in llm.generate_response_stream(message, temperature, max_tokens):
            yield f"data: {json.dumps({'response': response})}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/clear', methods=['POST'])
def clear():
    llm.clear_history()
    return jsonify({"status": "cleared"})

@app.route('/switch_model', methods=['POST'])
def switch_model():
    data = request.json
    model = data.get('model')
    status = llm.switch_model(model)
    return jsonify({"status": status})

@app.route('/benchmark', methods=['POST'])
def benchmark():
    data = request.json
    model = data.get('model')
    results = llm.benchmark_model(model)
    return jsonify(results)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING WEB INTERFACE (Fixed Version)")
    print("="*60)
    print("\nServer running at: http://127.0.0.1:5000")
    print("Open this URL in your browser")
    print("\n Tips:")
    print("   - Use phi3:mini for best results (most stable)")
    print("   - Lower temperature (0.3-0.5) for factual answers")
    print("   - Increase temperature (0.7-0.9) for creative responses")
    print("\n  Note: The model cannot access real-time data like weather")
    print("   since it runs completely offline.")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)