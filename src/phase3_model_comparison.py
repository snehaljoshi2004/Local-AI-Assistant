import json
import time
import requests
from datetime import datetime
from pathlib import Path
import statistics
import psutil
import GPUtil

class ModelComparison:
    def __init__(self, models):
        self.models = models
        self.ollama_url = "http://localhost:11434"
        self.results = {}
        
    def get_system_stats(self):
        """Get current system stats"""
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
                    "memory_util_percent": gpus[0].memoryUtil * 100,
                    "utilization_percent": gpus[0].utilization
                }
            else:
                stats["gpu"] = None
        except:
            stats["gpu"] = None
            
        return stats
    
    def benchmark_model(self, model_name, prompts, num_runs=2):
        """Benchmark a single model on all prompts"""
        
        print(f"\n{'='*70}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*70}")
        
        # Get baseline system stats
        baseline_stats = self.get_system_stats()
        
        results = []
        
        for idx, prompt_data in enumerate(prompts, 1):
            prompt_text = prompt_data["prompt"]
            prompt_id = prompt_data["id"]
            category = prompt_data["category"]
            
            print(f"\n[{idx}/{len(prompts)}] {category.upper()}: {prompt_id}")
            
            for run in range(num_runs):
                start_time = time.time()
                first_token_time = None
                token_count = 0
                response_text = ""
                
                # Make streaming request
                try:
                    response = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": model_name,
                            "prompt": prompt_text,
                            "stream": True,
                            "options": {"temperature": 0}
                        },
                        stream=True,
                        timeout=120
                    )
                    
                    # Process stream
                    for line in response.iter_lines():
                        if line:
                            if first_token_time is None:
                                first_token_time = time.time() - start_time
                            
                            data = json.loads(line)
                            if "response" in data:
                                response_text += data["response"]
                                token_count += 1
                    
                    total_time = time.time() - start_time
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    total_time = time.time() - start_time
                    token_count = 0
                    first_token_time = None
                
                # Get system stats after run
                mem_after = self.get_system_stats()
                
                tokens_per_second = token_count / total_time if total_time > 0 else 0
                
                # Safely get GPU memory
                gpu_memory = None
                if mem_after.get("gpu") and isinstance(mem_after["gpu"], dict):
                    gpu_memory = mem_after["gpu"].get("memory_used_mb")
                
                result = {
                    "prompt_id": prompt_id,
                    "category": category,
                    "run": run + 1,
                    "metrics": {
                        "tokens_per_second": round(tokens_per_second, 2),
                        "time_to_first_token_ms": round(first_token_time * 1000, 2) if first_token_time else None,
                        "total_time_seconds": round(total_time, 2),
                        "tokens_generated": token_count,
                        "ram_usage_percent": mem_after.get("ram_percent", 0),
                        "gpu_memory_used_mb": gpu_memory,
                    },
                    "response_preview": response_text[:200] if response_text else "No response"
                }
                
                results.append(result)
                
                if tokens_per_second > 0:
                    ttft_ms = first_token_time * 1000 if first_token_time else 0
                    print(f"  Run {run+1}: {tokens_per_second:.1f} tok/s, TTFT={ttft_ms:.0f}ms, {token_count} tokens")
                else:
                    print(f"  Run {run+1}: Failed or no tokens generated")
            
        return {
            "model": model_name,
            "baseline_stats": baseline_stats,
            "results": results
        }
    
    def run_comparison(self, prompts_file, runs_per_prompt=2):
        """Run comparison across all models"""
        
        # Load prompts
        with open(prompts_file, 'r') as f:
            suite = json.load(f)
        
        prompts = suite["prompts"]
        
        print(f"\n{'#'*70}")
        print(f"MODEL COMPARISON STUDY")
        print(f"Models: {', '.join(self.models)}")
        print(f"Total prompts: {len(prompts)}")
        print(f"Runs per prompt: {runs_per_prompt}")
        print(f"Total runs per model: {len(prompts) * runs_per_prompt}")
        print(f"{'#'*70}")
        
        for model in self.models:
            print(f"\n\n{'='*70}")
            print(f"Starting benchmark for: {model}")
            print(f"{'='*70}")
            self.results[model] = self.benchmark_model(model, prompts, runs_per_prompt)
        
        return self.results
    
    def generate_comparison_report(self, output_dir="benchmarks/results"):
        """Generate comparison report across models"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Calculate per-model statistics
        report = {
            "timestamp": timestamp,
            "models": {}
        }
        
        for model, data in self.results.items():
            results = data["results"]
            
            if not results:
                report["models"][model] = {
                    "overall": {
                        "avg_tokens_per_second": 0,
                        "median_tokens_per_second": 0,
                        "min_tokens_per_second": 0,
                        "max_tokens_per_second": 0,
                        "std_tokens_per_second": 0,
                        "avg_time_to_first_token_ms": 0,
                        "avg_tokens_generated": 0,
                        "total_runs": 0
                    },
                    "by_category": {},
                    "gpu_memory_mb": None
                }
                continue
            
            tps_values = [r["metrics"]["tokens_per_second"] for r in results if r["metrics"]["tokens_per_second"] > 0]
            ttft_values = [r["metrics"]["time_to_first_token_ms"] for r in results if r["metrics"]["time_to_first_token_ms"]]
            token_counts = [r["metrics"]["tokens_generated"] for r in results]
            
            # Category breakdown
            categories = {}
            for r in results:
                cat = r["category"]
                if cat not in categories:
                    categories[cat] = {"tps": []}
                if r["metrics"]["tokens_per_second"] > 0:
                    categories[cat]["tps"].append(r["metrics"]["tokens_per_second"])
            
            category_stats = {}
            for cat, stats in categories.items():
                if stats["tps"]:
                    category_stats[cat] = {
                        "avg_tokens_per_second": round(statistics.mean(stats["tps"]), 2),
                        "num_runs": len(stats["tps"])
                    }
                else:
                    category_stats[cat] = {
                        "avg_tokens_per_second": 0,
                        "num_runs": 0
                    }
            
            # Get GPU memory from baseline stats
            gpu_memory = None
            if data.get("baseline_stats") and data["baseline_stats"].get("gpu"):
                if isinstance(data["baseline_stats"]["gpu"], dict):
                    gpu_memory = data["baseline_stats"]["gpu"].get("memory_used_mb")
            
            report["models"][model] = {
                "overall": {
                    "avg_tokens_per_second": round(statistics.mean(tps_values), 2) if tps_values else 0,
                    "median_tokens_per_second": round(statistics.median(tps_values), 2) if tps_values else 0,
                    "min_tokens_per_second": round(min(tps_values), 2) if tps_values else 0,
                    "max_tokens_per_second": round(max(tps_values), 2) if tps_values else 0,
                    "std_tokens_per_second": round(statistics.stdev(tps_values), 2) if len(tps_values) > 1 else 0,
                    "avg_time_to_first_token_ms": round(statistics.mean(ttft_values), 2) if ttft_values else 0,
                    "avg_tokens_generated": round(statistics.mean(token_counts), 1) if token_counts else 0,
                    "total_runs": len(results)
                },
                "by_category": category_stats,
                "gpu_memory_mb": gpu_memory
            }
        
        # Save report
        report_file = output_path / f"phase3_comparison_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*70}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        for model, stats in report["models"].items():
            print(f"\n{model.upper()}:")
            print(f"  Avg Tokens/Second: {stats['overall']['avg_tokens_per_second']:.2f}")
            print(f"  Min/Max Tokens/Second: {stats['overall']['min_tokens_per_second']:.2f} / {stats['overall']['max_tokens_per_second']:.2f}")
            print(f"  Avg TTFT: {stats['overall']['avg_time_to_first_token_ms']:.2f} ms")
            print(f"  Avg Tokens Generated: {stats['overall']['avg_tokens_generated']:.1f}")
            print(f"  Total Runs: {stats['overall']['total_runs']}")
            if stats['gpu_memory_mb']:
                print(f"  GPU Memory: {stats['gpu_memory_mb']:.0f} MB")
            
            # Show best category
            if stats['by_category']:
                best_cat = max(stats['by_category'].items(), key=lambda x: x[1]['avg_tokens_per_second'])
                print(f"  Best Category: {best_cat[0]} ({best_cat[1]['avg_tokens_per_second']:.2f} tok/s)")
        
        # Identify fastest model
        valid_models = {k: v for k, v in report["models"].items() if v["overall"]["avg_tokens_per_second"] > 0}
        if valid_models:
            fastest = max(valid_models.items(), key=lambda x: x[1]["overall"]["avg_tokens_per_second"])
            print(f"\n{'='*70}")
            print(f" FASTEST MODEL: {fastest[0].upper()}")
            print(f"   {fastest[1]['overall']['avg_tokens_per_second']:.2f} tokens/second")
            print(f"{'='*70}")
        
        # Identify model with best quality (most tokens generated on average)
        if valid_models:
            most_tokens = max(valid_models.items(), key=lambda x: x[1]["overall"]["avg_tokens_generated"])
            print(f"\n MOST DETAILED MODEL: {most_tokens[0].upper()}")
            print(f"   {most_tokens[1]['overall']['avg_tokens_generated']:.1f} avg tokens/response")
        
        return report_file

if __name__ == "__main__":
    # Define models to compare
    models_to_test = [
        "llama3.2:3b",
        "phi3:mini",
        "mistral"
    ]
    
    print("\n" + "="*70)
    print("PHASE 3: MODEL COMPARISON STUDY")
    print("="*70)
    print("\nModels to test:")
    for model in models_to_test:
        print(f"  - {model}")
    print("\nThis will run 30 prompts × 2 runs = 60 runs per model")
    print("Estimated time: 15-20 minutes per model")
    print("="*70)
    
    # Create comparison runner
    comparison = ModelComparison(models_to_test)
    
    # Run comparison
    results = comparison.run_comparison("prompts/test_suite.json", runs_per_prompt=2)
    
    # Generate report
    report_file = comparison.generate_comparison_report()
    
    print(f"\n Full comparison report saved to: {report_file}")
    print("\n Phase 3 Complete! ")
    print("\nProject Summary:")
    print("   Phase 1: Benchmarking framework built")
    print("   Phase 2: JSON validation + temperature experiments")
    print("   Phase 3: 3-model comparison completed")