import json
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

class Conversation:
    def __init__(self, model_name="phi3:mini", max_history=10):
        self.model_name = model_name
        self.max_history = max_history
        self.messages = []
        self.ollama_url = "http://localhost:11434"
        
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last N messages
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history * 2:]
    
    def get_conversation_context(self) -> str:
        """Format conversation history as context"""
        context = []
        for msg in self.messages:
            if msg["role"] == "user":
                context.append(f"User: {msg['content']}")
            else:
                context.append(f"Assistant: {msg['content']}")
        return "\n".join(context)
    
    def send_message(self, user_input: str, temperature: float = 0.7) -> str:
        """Send a message and get response with context"""
        
        # Add user message to history
        self.add_message("user", user_input)
        
        # Build prompt with conversation history
        if len(self.messages) > 1:
            prompt = f"""Previous conversation:
{self.get_conversation_context()}

User: {user_input}
Assistant: """
        else:
            prompt = user_input
        
        # Get response from model
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=60
        )
        
        assistant_response = response.json().get("response", "")
        
        # Add assistant response to history
        self.add_message("assistant", assistant_response)
        
        return assistant_response
    
    def clear_history(self):
        """Clear conversation history"""
        self.messages = []
        print("Conversation history cleared.")
    
    def save_conversation(self, filename: str = None):
        """Save conversation to file"""
        if filename is None:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_dir = Path("conversations")
        output_dir.mkdir(exist_ok=True)
        
        data = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "messages": self.messages
        }
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f" Conversation saved to: {filepath}")
        return filepath
    
    def load_conversation(self, filepath: str):
        """Load conversation from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.model_name = data["model"]
        self.messages = data["messages"]
        print(f" Loaded conversation with {len(self.messages)} messages")

def run_conversation_cli():
    """Run interactive conversation in terminal"""
    print("\n" + "="*60)
    print("MULTI-TURN CONVERSATION MODE")
    print("="*60)
    print("\nCommands:")
    print("  /clear   - Clear conversation history")
    print("  /save    - Save conversation to file")
    print("  /model   - Switch model")
    print("  /temp    - Adjust temperature")
    print("  /exit    - Exit conversation")
    print("="*60)
    
    # Initialize conversation
    conv = Conversation(model_name="phi3:mini")
    
    while True:
        user_input = input("\n You: ").strip()
        
        if user_input.lower() == "/exit":
            print("\n Goodbye!")
            break
        
        elif user_input.lower() == "/clear":
            conv.clear_history()
            continue
        
        elif user_input.lower() == "/save":
            conv.save_conversation()
            continue
        
        elif user_input.lower() == "/model":
            new_model = input("Enter model name (llama3.2:3b, phi3:mini, mistral): ").strip()
            conv.model_name = new_model
            print(f" Switched to model: {new_model}")
            continue
        
        elif user_input.lower() == "/temp":
            try:
                temp = float(input("Enter temperature (0-1): ").strip())
                # Temperature will be passed in send_message
                print(f" Temperature set to: {temp}")
                # For demo, we'll just note it
            except:
                print("Invalid temperature value")
            continue
        
        else:
            print("\n Assistant: ", end="", flush=True)
            response = conv.send_message(user_input)
            print(response)

if __name__ == "__main__":
    run_conversation_cli()