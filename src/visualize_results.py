import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BenchmarkVisualizer:
    def __init__(self, results_dir="benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("benchmarks/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_latest_phase3_results(self):
        """Load the most recent phase3 comparison results"""
        phase3_files = list(self.results_dir.glob("phase3_comparison_*.json"))
        if not phase3_files:
            raise FileNotFoundError("No phase3 comparison results found")
        
        latest = max(phase3_files, key=lambda x: x.stat().st_mtime)
        with open(latest, 'r') as f:
            return json.load(f)
    
    def create_performance_comparison_chart(self, data):
        """Create bar chart comparing model performance"""
        models = list(data["models"].keys())
        tokens_per_sec = [data["models"][m]["overall"]["avg_tokens_per_second"] for m in models]
        
        # Clean model names for display
        display_names = [m.replace(":", "\n").upper() for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(display_names, tokens_per_sec, color=sns.color_palette("husl", len(models)))
        
        # Add value labels on bars
        for bar, value in zip(bars, tokens_per_sec):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tokens Per Second', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison\nHigher is Better', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(tokens_per_sec) * 1.15)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f" Saved: {self.output_dir / 'performance_comparison.png'}")
    
    def create_category_breakdown_chart(self, data):
        """Create grouped bar chart for category performance"""
        models = list(data["models"].keys())
        categories = ['factual', 'reasoning', 'coding', 'creative']
        
        # Prepare data
        plot_data = []
        for model in models:
            for cat in categories:
                if cat in data["models"][model]["by_category"]:
                    tps = data["models"][model]["by_category"][cat]["avg_tokens_per_second"]
                    plot_data.append({
                        'Model': model.upper(),
                        'Category': cat.capitalize(),
                        'Tokens/Second': tps
                    })
        
        df = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create grouped bar chart
        x = np.arange(len(categories))
        width = 0.25
        colors = sns.color_palette("husl", len(models))
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model.upper()]
            values = []
            for cat in categories:
                cat_data = model_data[model_data['Category'] == cat.capitalize()]
                values.append(cat_data['Tokens/Second'].values[0] if not cat_data.empty else 0)
            
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model.upper(), color=colors[i])
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tokens Per Second', fontsize=12, fontweight='bold')
        ax.set_title('Performance by Category Across Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in categories])
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_breakdown.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {self.output_dir / 'category_breakdown.png'}")
    
    def create_ttft_comparison_chart(self, data):
        """Create chart comparing Time To First Token"""
        models = list(data["models"].keys())
        ttft_values = [data["models"][m]["overall"]["avg_time_to_first_token_ms"] for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(models))
        bars = ax.barh(y_pos, ttft_values, color=sns.color_palette("RdYlGn_r", len(models)))
        
        # Add value labels
        for bar, value in zip(bars, ttft_values):
            ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                   f'{value:.0f}ms', va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([m.upper() for m in models])
        ax.set_xlabel('Time to First Token (milliseconds)', fontsize=12, fontweight='bold')
        ax.set_title('Response Latency Comparison\nLower is Better', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ttft_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f" Saved: {self.output_dir / 'ttft_comparison.png'}")
    
    def create_radar_chart(self, data):
        """Create radar chart for multi-dimensional comparison"""
        models = list(data["models"].keys())
        categories = ['Speed\n(Tok/s)', 'Low Latency\n(1/TTFT)', 'Detail\n(Tokens)', 'Consistency']
        
        # Normalize values for radar chart
        all_tps = [data["models"][m]["overall"]["avg_tokens_per_second"] for m in models]
        all_ttft = [data["models"][m]["overall"]["avg_time_to_first_token_ms"] for m in models]
        all_tokens = [data["models"][m]["overall"]["avg_tokens_generated"] for m in models]
        
        # Normalize (0-1 scale, higher is better)
        tps_norm = [tps / max(all_tps) for tps in all_tps]
        ttft_norm = [1 - (ttft / max(all_ttft)) for ttft in all_ttft]  # Lower is better
        tokens_norm = [tokens / max(all_tokens) for tokens in all_tokens]
        
        # Calculate consistency (using std deviation of TPS - lower std = more consistent)
        consistency = []
        for model in models:
            tps_values = [r["metrics"]["tokens_per_second"] for r in data["models"][model]["results"] 
                         if r["metrics"]["tokens_per_second"] > 0]
            std_tps = np.std(tps_values) if tps_values else 1
            # Normalize consistency (lower std = higher consistency score)
            consistency.append(1 / (1 + std_tps / 10))  # Scale adjustment
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        for i, model in enumerate(models):
            values = [tps_norm[i], ttft_norm[i], tokens_norm[i], consistency[i]]
            values += values[:1]  # Close the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=model.upper())
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart\n(0-1 Scale, Higher is Better)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'radar_chart.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f" Saved: {self.output_dir / 'radar_chart.png'}")
    
    def create_summary_dashboard(self, data):
        """Create a comprehensive summary dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(data["models"].keys())
        tokens_per_sec = [data["models"][m]["overall"]["avg_tokens_per_second"] for m in models]
        ttft_values = [data["models"][m]["overall"]["avg_time_to_first_token_ms"] for m in models]
        tokens_gen = [data["models"][m]["overall"]["avg_tokens_generated"] for m in models]
        
        # Subplot 1: Speed
        axes[0, 0].bar(models, tokens_per_sec, color='skyblue')
        axes[0, 0].set_title('Speed (Tokens/Second)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Tok/s')
        for i, v in enumerate(tokens_per_sec):
            axes[0, 0].text(i, v + 0.5, f'{v:.1f}', ha='center')
        
        # Subplot 2: Latency
        axes[0, 1].bar(models, ttft_values, color='lightcoral')
        axes[0, 1].set_title('Latency (TTFT in ms)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('ms')
        for i, v in enumerate(ttft_values):
            axes[0, 1].text(i, v + 20, f'{v:.0f}', ha='center')
        
        # Subplot 3: Detail
        axes[1, 0].bar(models, tokens_gen, color='lightgreen')
        axes[1, 0].set_title('Response Detail (Avg Tokens)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Tokens')
        for i, v in enumerate(tokens_gen):
            axes[1, 0].text(i, v + 2, f'{v:.0f}', ha='center')
        
        # Subplot 4: Efficiency (tokens per second per GB)
        # Estimate model sizes
        sizes = {'llama3.2:3b': 2.0, 'phi3:mini': 2.2, 'mistral': 4.1}
        efficiency = [tokens_per_sec[i] / sizes.get(m, 2.0) for i, m in enumerate(models)]
        axes[1, 1].bar(models, efficiency, color='gold')
        axes[1, 1].set_title('Efficiency (Tok/s per GB)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Tok/s/GB')
        for i, v in enumerate(efficiency):
            axes[1, 1].text(i, v + 0.2, f'{v:.1f}', ha='center')
        
        plt.suptitle('Model Benchmarking Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f" Saved: {self.output_dir / 'summary_dashboard.png'}")
    
    def generate_all_charts(self):
        """Generate all visualization charts"""
        print("\n" + "="*60)
        print("GENERATING PERFORMANCE VISUALIZATIONS")
        print("="*60)
        
        # Load data
        data = self.load_latest_phase3_results()
        
        # Generate all charts
        self.create_performance_comparison_chart(data)
        self.create_category_breakdown_chart(data)
        self.create_ttft_comparison_chart(data)
        self.create_radar_chart(data)
        self.create_summary_dashboard(data)
        
        print(f"\n All visualizations saved to: {self.output_dir}")
        print("\n Charts generated:")
        for chart in self.output_dir.glob("*.png"):
            print(f"   - {chart.name}")

if __name__ == "__main__":
    visualizer = BenchmarkVisualizer()
    visualizer.generate_all_charts()