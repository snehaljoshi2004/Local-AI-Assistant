import time
import json
import requests
from datetime import datetime

print("Simple Benchmark - Testing llama3.2:3b")
print("="*50)

# Hardcoded test prompts
test_prompts = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a Python function to add two numbers.",
    "Tell me a short joke.",
    "What is 2+2?"
]

results = []

for i, prompt in enumerate(test_prompts, 1):
    print(f"\nPrompt {i}: {prompt}")
    
    start_time = time.time()
    first_token_time = None
    response_text = ""
    token_count = 0
    
    # Make API request
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )
    
    # Process streaming response
    for line in response.iter_lines():
        if line:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            
            data = json.loads(line)
            if "response" in data:
                response_text += data["response"]
                token_count += 1
    
    total_time = time.time() - start_time
    tokens_per_second = token_count / total_time if total_time > 0 else 0
    
    result = {
        "prompt": prompt,
        "tokens_per_second": round(tokens_per_second, 2),
        "time_to_first_token_ms": round(first_token_time * 1000, 2),
        "total_time_seconds": round(total_time, 2),
        "tokens_generated": token_count,
        "response": response_text[:200]
    }
    
    results.append(result)
    
    print(f"  Tokens/sec: {tokens_per_second:.2f}")
    print(f"  TTFT: {first_token_time*1000:.2f}ms")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Tokens: {token_count}")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"benchmarks/results/phase1_{timestamp}.json"

import os
os.makedirs("benchmarks/results", exist_ok=True)

with open(filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*50}")
print(f" Results saved to: {filename}")
print(f"\nSummary:")
avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
avg_ttft = sum(r["time_to_first_token_ms"] for r in results) / len(results)
print(f"Average Tokens/Second: {avg_tps:.2f}")
print(f"Average TTFT: {avg_ttft:.2f}ms")