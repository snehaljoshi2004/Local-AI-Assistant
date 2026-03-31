# Local AI Assistant - Offline LLM Inference System

A production-ready, fully offline AI assistant system that leverages small language models (3B-7B parameters) for private, local inference. Designed for privacy-conscious deployments requiring complete air-gapped operation.

## Overview

This project implements a comprehensive local AI assistant framework with benchmarking capabilities, multi-model support, structured output validation, and a responsive web interface. The system runs entirely on local hardware with no internet dependency after initial setup.

### Key Differentiators

- **Zero Data Leakage**: All inference occurs locally on your hardware. No API calls, no telemetry, no external dependencies.
- **Hardware-Aware Benchmarking**: Built-in performance measurement suite that profiles token throughput, latency, and memory utilization across different model architectures.
- **Structured Output Guarantees**: Pydantic-based JSON schema enforcement with automatic retry mechanisms for deterministic outputs.
- **Multi-Model Orchestration**: Seamless switching between different parameter scales (3B to 7B) for performance vs capability tradeoffs.



## Technical Specifications

### Model Support Matrix

| Model | Parameters | Quantization | VRAM Required | Optimal Use Case |
|-------|------------|--------------|---------------|------------------|
| Phi-3 Mini | 3.8B | 4-bit | 2.2 GB | High-speed inference, code generation |
| Llama 3.2 | 3B | 4-bit | 2.0 GB | Creative writing, conversational tasks |
| Mistral | 7B | 4-bit | 4.1 GB | Complex reasoning, detailed analysis |

### Performance Benchmarks

Tests conducted on NVIDIA RTX 4050 (6GB VRAM) with 30 diverse prompts across 4 categories:

| Model | Tokens/Second | TTFT (ms) | Avg Response Length | Category Performance |
|-------|---------------|-----------|---------------------|---------------------|
| Phi-3 Mini | 27.74 | 2189 | 129 tokens | Coding: 32.71 tok/s |
| Llama 3.2 3B | 20.11 | 2526 | 93 tokens | Creative: 26.32 tok/s |
| Mistral 7B | 14.18 | 2325 | 118 tokens | Balanced |

### Category-Specific Analysis

| Category | Llama 3.2 | Phi-3 Mini | Mistral | Optimal Model |
|----------|-----------|------------|---------|---------------|
| Factual QA | 12.86 tok/s | 26.24 tok/s | 13.81 tok/s | Phi-3 Mini |
| Reasoning | 19.75 tok/s | 25.67 tok/s | 14.13 tok/s | Phi-3 Mini |
| Code Generation | 25.94 tok/s | 32.71 tok/s | 18.01 tok/s | Phi-3 Mini |
| Creative Writing | 26.32 tok/s | 29.14 tok/s | 14.70 tok/s | Llama 3.2 |

## Features

### Phase 1: Performance Benchmarking Framework

- Token-level latency measurement (time-to-first-token, inter-token latency)
- Concurrent streaming response processing
- GPU memory utilization tracking (via NVML/GPUtil)
- CPU and RAM profiling during inference
- Statistical analysis across multiple runs
- JSON export of benchmark results

### Phase 2: Structured Output Engineering

- JSON schema enforcement with Pydantic validation
- Automatic retry mechanism with configurable attempts
- Temperature ablation studies (T=0 vs T=0.7)
- Output variance quantification and diversity scoring
- Semantic similarity analysis across temperature settings

### Phase 3: Comparative Model Analysis

- Standardized 30-prompt evaluation suite
- Category-wise performance breakdown
- Resource utilization profiling
- Efficiency metrics (tokens/second per GB VRAM)
- Automated report generation

### Web Interface Capabilities

- Real-time model switching without server restart
- Streaming response generation with SSE (Server-Sent Events)
- Adjustable inference parameters (temperature, max tokens)
- Conversation context preservation (configurable history length)
- Built-in performance benchmarking utility
- Dark theme optimized for extended usage

## Prerequisites

### Hardware Requirements

- CPU: x86_64 or ARM64 architecture
- RAM: 8 GB minimum (16 GB recommended)
- Storage: 10 GB free space for models and dependencies
- GPU: NVIDIA with 4+ GB VRAM (optional, falls back to CPU)

### Software Requirements

- Python 3.10 or higher
- Ollama 0.1.0 or higher
- Windows 10/11, Linux, or macOS

## Installation Guide

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/local-ai-assistant.git
cd local-ai-assistant
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Ollama

Download from: https://ollama.com/download

### Step 4: Download Models

```bash
ollama pull phi3:mini      # 2.2 GB - Recommended for most use cases
ollama pull llama3.2:3b    # 2.0 GB - Creative tasks
ollama pull mistral        # 4.1 GB - Complex reasoning
```
### Step 5: Launch Application
```bash
python src/web_ui_fixed.py
```

## User Interface
- Figure 1: Main chat interface with model selection, temperature control, and conversation area
path: docs/ui_screenshot.png

## Operational Guide

### Basic Usage
- Model Selection: Choose from available models using the dropdown menu
- Temperature Adjustment:
- 0.0-0.3: Deterministic, factual responses
- 0.4-0.7: Balanced creativity
- 0.8-1.0: Highly creative, varied outputs
- Conversation: Type messages in the input field and press Enter
- Clear History: Use the "Clear" button to reset conversation context

## Benchmarking

- Execute performance tests:

```bash
# Phase 1: Single model benchmark
python src/benchmark_phase1.py

# Phase 2: JSON validation and temperature experiments
python src/phase2_json_temperature.py

# Phase 3: Multi-model comparison
python src/phase3_model_comparison.py

# Generate visualization suite
python src/visualize_results.py
```

## Configuration Parameters

### Inference Parameters

| Parameter	| Range |	Default	| Description|
|--------|--------|---------|------------|
| temperature | 	0.0 | - 1.0	0.7	|Controls output randomness|
| num_predict	| 10 - 2000	| 500	| Maximum tokens to generate|
| repeat_penalty | 	1.0 | - 2.0	1.1	| Penalizes token repetition|
| top_k	| 1 - 100	| 40	| Vocabulary sampling filter|
| top_p	| 0.0 - 1.0	| 0.9	| Nucleus sampling threshold|

## Performance Optimization
- GPU Acceleration: Ensure NVIDIA drivers and CUDA are properly installed
- Model Quantization: Use 4-bit quantized versions for lower VRAM usage
- Batch Processing: For multiple prompts, use non-streaming mode
- Conversation Pruning: Limit history length to 5-10 exchanges

## Limitations
- Cannot access real-time data (weather, news, stocks) without external API integration
- Model knowledge cutoff dates: Llama 3.2 (March 2024), Phi-3 (October 2023), Mistral (June 2023)
- Maximum context window: 4096-8192 tokens depending on model
- No multimodal capabilities (image/audio processing)
