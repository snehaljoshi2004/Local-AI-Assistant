import time
import json
import requests
from datetime import datetime
import psutil
import GPUtil
import statistics
from pathlib import Path

class EnhancedLLMBenchmark:
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434"
        self.results = []
        
    def get_system_stats(self):
        """Get comprehensive system stats"""
        stats = {
            "ram_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1)
        }
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                stats["gpu"] = {
                    "memory_used_mb": gpus[0].memoryUsed,
                    "memory_total_mb": gpus[0].memoryTotal,
                    "utilization_percent": gpus[0].utilization,
                    "temperature_c": gpus[0].temperature
                }
        except:
            stats["gpu"] = None
            
        return stats
    
    def benchmark_prompt(self, prompt, prompt_id, category, run_number):
        """Benchmark a single prompt"""
        
        # Get baseline stats
        stats_before = self.get_system_stats()
        start_time = time.time()
        
        # Track tokens and timing
        first_token_time = None
        tokens = []
        token_timings = []
        response_text = ""
        
        # Make streaming request
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0}  # Deterministic for benchmarking
            },
            stream=True
        )
        
        # Process stream
        for line in response.iter_lines():
            if line:
                token_time = time.time()
                data = json.loads(line)
                
                if "response" in data:
                    if first_token_time is None:
                        first_token_time = token_time - start_time
                    
                    response_text += data["response"]
                    tokens.append(data["response"])
                    token_timings.append(token_time - start_time)
        
        total_time = time.time() - start_time
        stats_after = self.get_system_stats()
        
        # Calculate metrics
        num_tokens = len(tokens)
        tokens_per_second = num_tokens / total_time if total_time > 0 else 0
        
        # Calculate token timing percentiles
        if len(token_timings) > 1:
            inter_token_times = [token_timings[i] - token_timings[i-1] for i in range(1, len(token_timings))]
        else:
            inter_token_times = []
        
        result = {
            "prompt_id": prompt_id,
            "category": category,
            "prompt": prompt[:100],
            "run": run_number,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_time_seconds": round(total_time, 3),
                "time_to_first_token_ms": round(first_token_time * 1000, 2) if first_token_time else None,
                "tokens_generated": num_tokens,
                "tokens_per_second": round(tokens_per_second, 2),
                "avg_token_latency_ms": round(statistics.mean(inter_token_times) * 1000, 2) if inter_token_times else None,
                "token_latency_std_ms": round(statistics.stdev(inter_token_times) * 1000, 2) if len(inter_token_times) > 1 else None,
                "ram_usage_before_percent": stats_before["ram_percent"],
                "ram_usage_after_percent": stats_after["ram_percent"],
                "cpu_usage_before_percent": stats_before["cpu_percent"],
                "cpu_usage_after_percent": stats_after["cpu_percent"],
            },
            "gpu_metrics": stats_after.get("gpu"),
            "response": response_text
        }
        
        return result
    
    def run_benchmark_suite(self, prompts_file, runs_per_prompt=2):
        """Run benchmarks on all prompts"""
        
        # Load prompts
        with open(prompts_file, 'r') as f:
            suite = json.load(f)
        
        prompts = suite["prompts"]
        
        print(f"\n{'='*70}")
        print(f"ENHANCED BENCHMARK: {self.model_name}")
        print(f"Total prompts: {len(prompts)}")
        print(f"Runs per prompt: {runs_per_prompt}")
        print(f"Total runs: {len(prompts) * runs_per_prompt}")
        print(f"{'='*70}\n")
        
        all_results = []
        
        for idx, prompt_data in enumerate(prompts, 1):
            prompt_text = prompt_data["prompt"]
            prompt_id = prompt_data["id"]
            category = prompt_data["category"]
            
            print(f"[{idx}/{len(prompts)}] {category.upper()}: {prompt_id}")
            
            for run in range(runs_per_prompt):
                result = self.benchmark_prompt(prompt_text, prompt_id, category, run+1)
                all_results.append(result)
                
                # Print progress
                tps = result["metrics"]["tokens_per_second"]
                ttft = result["metrics"]["time_to_first_token_ms"]
                tokens = result["metrics"]["tokens_generated"]
                print(f"  Run {run+1}: {tps:.1f} tok/s, TTFT={ttft:.0f}ms, {tokens} tokens")
            
            print()
        
        return all_results
    
    def save_results(self, results, output_dir="benchmarks/results"):
        """Save results with analysis"""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/phase1_{self.model_name.replace(':', '_')}_{timestamp}.json"
        
        output = {
            "model": self.model_name,
            "timestamp": timestamp,
            "total_runs": len(results),
            "results": results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Generate summary
        summary = self.generate_summary(results)
        summary_file = f"{output_dir}/summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n Raw results: {filename}")
        print(f" Summary: {summary_file}")
        
        return filename, summary_file
    
    def generate_summary(self, results):
        """Generate summary statistics"""
        
        # Overall stats
        tps_values = [r["metrics"]["tokens_per_second"] for r in results]
        ttft_values = [r["metrics"]["time_to_first_token_ms"] for r in results if r["metrics"]["time_to_first_token_ms"]]
        token_counts = [r["metrics"]["tokens_generated"] for r in results]
        
        # Category breakdown
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"tps": [], "ttft": [], "tokens": []}
            categories[cat]["tps"].append(r["metrics"]["tokens_per_second"])
            categories[cat]["ttft"].append(r["metrics"]["time_to_first_token_ms"])
            categories[cat]["tokens"].append(r["metrics"]["tokens_generated"])
        
        summary = {
            "overall": {
                "avg_tokens_per_second": round(statistics.mean(tps_values), 2),
                "std_tokens_per_second": round(statistics.stdev(tps_values), 2) if len(tps_values) > 1 else 0,
                "min_tokens_per_second": round(min(tps_values), 2),
                "max_tokens_per_second": round(max(tps_values), 2),
                "avg_time_to_first_token_ms": round(statistics.mean(ttft_values), 2),
                "avg_tokens_generated": round(statistics.mean(token_counts), 1),
                "total_runs": len(results)
            },
            "by_category": {}
        }
        
        for cat, data in categories.items():
            summary["by_category"][cat] = {
                "avg_tokens_per_second": round(statistics.mean(data["tps"]), 2),
                "avg_time_to_first_token_ms": round(statistics.mean([x for x in data["ttft"] if x]), 2),
                "avg_tokens_generated": round(statistics.mean(data["tokens"]), 1),
                "num_runs": len(data["tps"])
            }
        
        return summary

if __name__ == "__main__":
    benchmark = EnhancedLLMBenchmark(model_name="llama3.2:3b")
    results = benchmark.run_benchmark_suite("prompts/test_suite.json", runs_per_prompt=2)
    benchmark.save_results(results)