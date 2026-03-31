import json
import time
import requests
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, ValidationError, create_model
from typing import List, Optional, Any, Dict
import statistics

# Define JSON schemas for validation
class FactResponse(BaseModel):
    answer: str
    confidence: float
    source: Optional[str] = None

class CodeResponse(BaseModel):
    function_name: str
    code: str
    explanation: str

class CreativeResponse(BaseModel):
    title: str
    content: str
    style: str

class JSONValidator:
    """Handles JSON validation and retry logic"""
    
    def __init__(self, model_name="llama3.2:3b", max_retries=1):
        self.model_name = model_name
        self.max_retries = max_retries
        self.ollama_url = "http://localhost:11434"
        
    def validate_and_retry(self, prompt, expected_schema, temperature=0):
        """Send prompt, validate JSON, retry if needed"""
        
        for attempt in range(self.max_retries + 1):
            # Add JSON instruction to prompt
            json_prompt = f"""{prompt}
            
IMPORTANT: Return your response as a valid JSON object matching this schema:
{json.dumps(expected_schema, indent=2)}

Response must be ONLY valid JSON, no other text."""
            
            # Get response
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": json_prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=60
            )
            
            response_text = response.json().get("response", "")
            
            # Try to extract JSON
            try:
                # Find JSON in response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    parsed = json.loads(json_str)
                    
                    # Validate against schema
                    # For now, just check required fields
                    valid = all(field in parsed for field in expected_schema.keys())
                    
                    if valid:
                        return {
                            "success": True,
                            "data": parsed,
                            "attempt": attempt + 1,
                            "raw_response": response_text[:200]
                        }
                else:
                    # Try parsing entire response
                    parsed = json.loads(response_text)
                    return {
                        "success": True,
                        "data": parsed,
                        "attempt": attempt + 1,
                        "raw_response": response_text[:200]
                    }
                    
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == self.max_retries:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempt": attempt + 1,
                        "raw_response": response_text[:200]
                    }
                # Retry with stronger prompting
                continue
        
        return {"success": False, "error": "Max retries exceeded"}

class TemperatureExperiment:
    """Run prompts at different temperatures to measure variance"""
    
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434"
        
    def run_prompt_multiple_times(self, prompt, temperature, num_runs=5):
        """Run same prompt multiple times at given temperature"""
        
        results = []
        
        for run in range(num_runs):
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                }
            )
            
            output = response.json().get("response", "")
            results.append({
                "run": run + 1,
                "output": output,
                "length": len(output)
            })
        
        return results
    
    def analyze_variance(self, results):
        """Analyze variance in outputs"""
        
        lengths = [r["length"] for r in results]
        
        # Calculate similarity between runs (simplified)
        outputs = [r["output"] for r in results]
        
        # Simple uniqueness check
        unique_outputs = len(set(outputs))
        
        return {
            "length_variance": statistics.variance(lengths) if len(lengths) > 1 else 0,
            "length_std": statistics.stdev(lengths) if len(lengths) > 1 else 0,
            "unique_outputs": unique_outputs,
            "total_runs": len(results),
            "diversity_ratio": unique_outputs / len(results)
        }
    
    def compare_temperatures(self, prompts, temp_low=0, temp_high=0.7, num_runs=5):
        """Compare outputs at low vs high temperature"""
        
        results = {
            "temperature_0": {},
            "temperature_0.7": {}
        }
        
        for prompt_data in prompts:
            prompt_id = prompt_data["id"]
            prompt_text = prompt_data["prompt"]
            
            print(f"\nTesting: {prompt_id}")
            
            # Run at T=0
            print(f"  Temperature {temp_low}...")
            t0_results = self.run_prompt_multiple_times(prompt_text, temp_low, num_runs)
            t0_analysis = self.analyze_variance(t0_results)
            
            # Run at T=0.7
            print(f"  Temperature {temp_high}...")
            t07_results = self.run_prompt_multiple_times(prompt_text, temp_high, num_runs)
            t07_analysis = self.analyze_variance(t07_results)
            
            results[f"temperature_{temp_low}"][prompt_id] = {
                "analysis": t0_analysis,
                "sample_outputs": [r["output"][:150] for r in t0_results[:2]]
            }
            
            results[f"temperature_{temp_high}"][prompt_id] = {
                "analysis": t07_analysis,
                "sample_outputs": [r["output"][:150] for r in t07_results[:2]]
            }
        
        return results

def run_phase2():
    """Main Phase 2 execution"""
    
    print("="*70)
    print("PHASE 2: JSON Schema Enforcement & Temperature Experiments")
    print("="*70)
    
    # Part 1: JSON Schema Validation with Retry
    print("\n PART 1: JSON Schema Validation")
    print("-"*50)
    
    validator = JSONValidator(model_name="llama3.2:3b", max_retries=1)
    
    # Test with JSON schema
    test_schema = {
        "answer": "string",
        "confidence": "number",
        "source": "string"
    }
    
    test_prompt = "What is the capital of France? Provide answer with confidence level."
    
    print(f"Prompt: {test_prompt}")
    print(f"Schema: {test_schema}")
    
    result = validator.validate_and_retry(test_prompt, test_schema, temperature=0)
    
    if result["success"]:
        print(f" Valid JSON received (attempt {result['attempt']})")
        print(f"Data: {json.dumps(result['data'], indent=2)}")
    else:
        print(f" Failed: {result['error']}")
        print(f"Raw: {result['raw_response']}")
    
    # Part 2: Temperature Comparison
    print("\n\n PART 2: Temperature Comparison (T=0 vs T=0.7)")
    print("-"*50)
    
    # Select diverse prompts for temperature test
    test_prompts = [
        {"id": "creative", "prompt": "Write a one-sentence creative story about a robot."},
        {"id": "factual", "prompt": "What is the capital of Japan?"},
        {"id": "code", "prompt": "Write a Python function to add two numbers."}
    ]
    
    experiment = TemperatureExperiment(model_name="llama3.2:3b")
    temp_results = experiment.compare_temperatures(test_prompts, num_runs=3)
    
    # Save temperature experiment results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_file = output_dir / f"phase2_temperature_{timestamp}.json"
    with open(temp_file, 'w') as f:
        json.dump(temp_results, f, indent=2)
    
    print(f"\n Temperature results saved to: {temp_file}")
    
    # Display summary
    print("\n TEMPERATURE COMPARISON SUMMARY")
    print("="*50)
    
    for prompt_id in temp_results["temperature_0"]:
        print(f"\nPrompt: {prompt_id}")
        t0 = temp_results["temperature_0"][prompt_id]["analysis"]
        t07 = temp_results["temperature_0.7"][prompt_id]["analysis"]
        
        print(f"  T=0    : {t0['unique_outputs']}/{t0['total_runs']} unique outputs")
        print(f"  T=0.7  : {t07['unique_outputs']}/{t07['total_runs']} unique outputs")
        print(f"  Diversity increase: {t07['diversity_ratio'] - t0['diversity_ratio']:.2%}")
    
    return temp_results

if __name__ == "__main__":
    results = run_phase2()