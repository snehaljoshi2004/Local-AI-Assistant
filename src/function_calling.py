import json
import requests
from datetime import datetime
from typing import Dict, Any, List, Callable

class FunctionCallingAssistant:
    def __init__(self, model_name="phi3:mini"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434"
        self.functions = {}
        
    def register_function(self, name: str, func: Callable, description: str, parameters: Dict):
        """Register a function that the assistant can call"""
        self.functions[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
    
    def get_functions_description(self) -> str:
        """Get description of all available functions"""
        desc = "Available functions:\n"
        for name, info in self.functions.items():
            desc += f"- {name}: {info['description']}\n"
            desc += f"  Parameters: {json.dumps(info['parameters'])}\n"
        return desc
    
    def call_function(self, function_name: str, arguments: Dict) -> str:
        """Execute a registered function"""
        if function_name in self.functions:
            try:
                result = self.functions[function_name]["function"](**arguments)
                return json.dumps({"success": True, "result": result})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})
        else:
            return json.dumps({"success": False, "error": f"Function {function_name} not found"})
    
    def process_query(self, user_query: str) -> str:
        """Process query with potential function calling"""
        
        # First, determine if we need to call a function
        prompt = f"""You are an AI assistant with function calling capability.
{self.get_functions_description()}

User query: {user_query}

If the user wants to perform an action that matches one of the available functions, 
respond with a JSON in this format:
{{"function": "function_name", "arguments": {{"arg1": "value1"}}}}

If no function is needed, just respond with a normal answer."""
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=60
        )
        
        response_text = response.json().get("response", "")
        
        # Try to parse as JSON (function call)
        try:
            # Find JSON in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                function_call = json.loads(response_text[json_start:json_end])
                if "function" in function_call:
                    # Execute the function
                    result = self.call_function(
                        function_call["function"],
                        function_call.get("arguments", {})
                    )
                    
                    # Get final response with function result
                    final_prompt = f"""User asked: {user_query}
Function called: {function_call['function']}
Function result: {result}

Now provide a natural response to the user based on this result."""
                    
                    final_response = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": final_prompt,
                            "stream": False
                        },
                        timeout=60
                    )
                    
                    return final_response.json().get("response", "")
        except:
            pass
        
        return response_text

# Example functions
def get_current_time():
    """Get current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str):
    """Calculate mathematical expression"""
    try:
        return eval(expression)
    except:
        return "Error in calculation"

def get_weather(city: str):
    """Get weather for a city (mock)"""
    # This would be replaced with actual API call
    return f"Weather in {city}: Sunny, 22°C"

def run_function_calling_demo():
    """Demo of function calling"""
    print("\n" + "="*60)
    print("FUNCTION CALLING ASSISTANT DEMO")
    print("="*60)
    
    assistant = FunctionCallingAssistant(model_name="phi3:mini")
    
    # Register functions
    assistant.register_function(
        "get_time",
        get_current_time,
        "Get current date and time",
        {}
    )
    
    assistant.register_function(
        "calculate",
        calculate,
        "Perform mathematical calculation",
        {"expression": {"type": "string", "description": "Math expression to calculate"}}
    )
    
    assistant.register_function(
        "weather",
        get_weather,
        "Get weather for a city",
        {"city": {"type": "string", "description": "City name"}}
    )
    
    # Test queries
    test_queries = [
        "What time is it right now?",
        "Calculate 25 * 4 + 10",
        "What's the weather in London?"
    ]
    
    for query in test_queries:
        print(f"\n User: {query}")
        response = assistant.process_query(query)
        print(f" Assistant: {response}")

if __name__ == "__main__":
    run_function_calling_demo()