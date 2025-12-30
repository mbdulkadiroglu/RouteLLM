#!/usr/bin/env python3
"""
LLM-as-Router Comparison

Compares using Llama 3.3 70B as a router vs MF and BERT routers.
Measures:
- Time to make routing decision
- Routing accuracy (agreement with "ground truth" based on complexity)
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3.3:70b"

# Router prompt for Llama
ROUTER_SYSTEM_PROMPT = """You are a query complexity classifier. Your job is to determine if a query requires a powerful AI model or if a simpler model can handle it.

Respond with ONLY a single number from 1-5:
1 = Very simple (greetings, basic facts, simple math)
2 = Simple (straightforward questions, basic explanations)
3 = Medium (requires some reasoning or domain knowledge)
4 = Complex (multi-step reasoning, coding, analysis)
5 = Very complex (advanced math, expert-level tasks, creative writing)

Just respond with the number, nothing else."""

def llama_router(prompt: str) -> tuple[float, float]:
    """
    Use Llama 3.3 70B to classify query complexity.
    Returns (score 0-1, time_ms)
    """
    start_time = time.perf_counter()
    
    full_prompt = f"{ROUTER_SYSTEM_PROMPT}\n\nQuery: {prompt}\n\nComplexity (1-5):"
    
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": 5,  # Only need a single digit
                "temperature": 0,
            }
        },
        timeout=60
    )
    
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    
    if response.status_code != 200:
        return 0.5, elapsed_ms  # Default to middle
    
    result = response.json()
    text = result.get("response", "").strip()
    
    # Parse the complexity score (1-5) and convert to 0-1 scale
    try:
        score = int(text[0]) if text else 3
        score = max(1, min(5, score))  # Clamp to 1-5
        # Convert to win_rate style (higher = needs stronger model)
        win_rate = (score - 1) / 4  # Maps 1->0, 5->1
    except:
        win_rate = 0.5
    
    return win_rate, elapsed_ms


def test_ollama_connection():
    """Test if Ollama is running and model is available."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": LLAMA_MODEL, "prompt": "Hi", "stream": False, "options": {"num_predict": 1}},
            timeout=30
        )
        return response.status_code == 200
    except:
        return False


# Test prompts with expected complexity labels
test_prompts = [
    # Simple (expected: 1-2, win_rate ~0-0.25)
    ("What is 2 + 2?", "simple"),
    ("Hello, how are you?", "simple"),
    ("What color is the sky?", "simple"),
    ("Name the capital of France.", "simple"),
    ("What is Python?", "simple"),
    
    # Medium (expected: 2-3, win_rate ~0.25-0.5)
    ("Explain how photosynthesis works.", "medium"),
    ("Write a Python function to reverse a string.", "medium"),
    ("What are the main causes of World War I?", "medium"),
    ("How does a neural network learn?", "medium"),
    ("Explain the difference between TCP and UDP.", "medium"),
    
    # Complex (expected: 4-5, win_rate ~0.75-1.0)
    ("Prove that the square root of 2 is irrational using proof by contradiction.", "complex"),
    ("Write a Python implementation of the A* pathfinding algorithm with detailed comments.", "complex"),
    ("Analyze the economic implications of quantitative easing on emerging markets.", "complex"),
    ("Derive the Euler-Lagrange equation from the principle of least action.", "complex"),
    ("Design a distributed system architecture for a real-time collaborative document editor.", "complex"),
]

print("=" * 70)
print("LLM-AS-ROUTER COMPARISON")
print("=" * 70)

# Check Ollama
print("\nChecking Ollama connection...")
if not test_ollama_connection():
    print("ERROR: Cannot connect to Ollama or model not available!")
    print(f"Make sure Ollama is running and {LLAMA_MODEL} is pulled.")
    exit(1)
print(f"✓ Connected to Ollama with {LLAMA_MODEL}")

# Check OpenAI API for MF router
has_openai = bool(os.environ.get("OPENAI_API_KEY"))
if not has_openai:
    print("WARNING: OPENAI_API_KEY not set - skipping MF router")

# Load RouteLLM routers
print("\nLoading RouteLLM routers...")
from routellm.controller import Controller

routers_to_load = ["bert"]
if has_openai:
    routers_to_load.append("mf")

controller = Controller(
    routers=routers_to_load,
    strong_model="gpt-4-1106-preview",
    weak_model="mixtral-8x7b-instruct-v0.1",
)
print(f"✓ Loaded: {routers_to_load}")

# Run comparison
results = []

# Test Llama 3.3 Router
print(f"\n{'=' * 50}")
print(f"Testing: LLAMA 3.3 70B Router")
print("=" * 50)

llama_times = []
llama_scores = []

for prompt, expected in tqdm(test_prompts, desc="Llama 3.3"):
    score, time_ms = llama_router(prompt)
    llama_times.append(time_ms)
    llama_scores.append(score)
    results.append({
        "router": "llama3.3",
        "prompt": prompt[:40] + "..." if len(prompt) > 40 else prompt,
        "expected": expected,
        "win_rate": score,
        "time_ms": time_ms,
    })

print(f"\n  Timing: {np.mean(llama_times):.1f} ms avg (min: {np.min(llama_times):.1f}, max: {np.max(llama_times):.1f})")
print(f"  Throughput: {1000/np.mean(llama_times):.2f} req/s")

# Test BERT Router
print(f"\n{'=' * 50}")
print(f"Testing: BERT Router")
print("=" * 50)

bert_times = []
bert_scores = []

for prompt, expected in tqdm(test_prompts, desc="BERT"):
    start = time.perf_counter()
    score = controller.routers["bert"].calculate_strong_win_rate(prompt)
    elapsed = (time.perf_counter() - start) * 1000
    bert_times.append(elapsed)
    bert_scores.append(score)
    results.append({
        "router": "bert",
        "prompt": prompt[:40] + "..." if len(prompt) > 40 else prompt,
        "expected": expected,
        "win_rate": score,
        "time_ms": elapsed,
    })

print(f"\n  Timing: {np.mean(bert_times):.1f} ms avg")
print(f"  Throughput: {1000/np.mean(bert_times):.2f} req/s")

# Test MF Router (if available)
if has_openai:
    print(f"\n{'=' * 50}")
    print(f"Testing: MF Router")
    print("=" * 50)
    
    mf_times = []
    mf_scores = []
    
    for prompt, expected in tqdm(test_prompts, desc="MF"):
        start = time.perf_counter()
        score = controller.routers["mf"].calculate_strong_win_rate(prompt)
        elapsed = (time.perf_counter() - start) * 1000
        mf_times.append(elapsed)
        mf_scores.append(score)
        results.append({
            "router": "mf",
            "prompt": prompt[:40] + "..." if len(prompt) > 40 else prompt,
            "expected": expected,
            "win_rate": score,
            "time_ms": elapsed,
        })
    
    print(f"\n  Timing: {np.mean(mf_times):.1f} ms avg")
    print(f"  Throughput: {1000/np.mean(mf_times):.2f} req/s")

# Create comparison DataFrame
df = pd.DataFrame(results)

# Detailed results
print("\n" + "=" * 70)
print("DETAILED COMPARISON")
print("=" * 70)

print("\n" + "-" * 100)
print(f"{'Prompt':<45} {'Expected':<8} {'Llama':>8} {'BERT':>8}", end="")
if has_openai:
    print(f" {'MF':>8}", end="")
print(f" {'Llama ms':>10} {'BERT ms':>10}", end="")
if has_openai:
    print(f" {'MF ms':>10}", end="")
print()
print("-" * 100)

for prompt, expected in test_prompts:
    short = prompt[:42] + "..." if len(prompt) > 42 else prompt
    
    llama_row = df[(df["router"] == "llama3.3") & (df["prompt"].str.startswith(prompt[:30]))]
    bert_row = df[(df["router"] == "bert") & (df["prompt"].str.startswith(prompt[:30]))]
    
    print(f"{short:<45} {expected:<8}", end="")
    
    if not llama_row.empty:
        print(f" {llama_row['win_rate'].values[0]:>8.3f}", end="")
    if not bert_row.empty:
        print(f" {bert_row['win_rate'].values[0]:>8.3f}", end="")
    
    if has_openai:
        mf_row = df[(df["router"] == "mf") & (df["prompt"].str.startswith(prompt[:30]))]
        if not mf_row.empty:
            print(f" {mf_row['win_rate'].values[0]:>8.3f}", end="")
    
    if not llama_row.empty:
        print(f" {llama_row['time_ms'].values[0]:>9.1f}ms", end="")
    if not bert_row.empty:
        print(f" {bert_row['time_ms'].values[0]:>9.1f}ms", end="")
    
    if has_openai and not mf_row.empty:
        print(f" {mf_row['time_ms'].values[0]:>9.1f}ms", end="")
    
    print()

# Summary
print("\n" + "=" * 70)
print("SUMMARY: TIME vs ACCURACY")
print("=" * 70)

# Calculate "accuracy" as correlation with expected complexity
def calc_routing_accuracy(router_name):
    """Calculate how well router scores correlate with expected complexity."""
    router_df = df[df["router"] == router_name]
    
    # Map expected to numeric
    expected_map = {"simple": 0, "medium": 0.5, "complex": 1}
    expected_scores = [expected_map[e] for _, e in test_prompts]
    router_scores = router_df["win_rate"].values
    
    # Correlation
    correlation = np.corrcoef(expected_scores, router_scores)[0, 1]
    
    # Classification accuracy at threshold 0.5
    expected_binary = [1 if e == "complex" else 0 for _, e in test_prompts]
    router_binary = [1 if s >= 0.5 else 0 for s in router_scores]
    accuracy = sum(e == r for e, r in zip(expected_binary, router_binary)) / len(expected_binary)
    
    return correlation, accuracy

print("\n┌─────────────────┬─────────────┬───────────────┬─────────────┬─────────────────┐")
print("│ Router          │ Avg Time    │ Throughput    │ Correlation │ Classification  │")
print("│                 │ (ms)        │ (req/s)       │ w/ Expected │ Accuracy        │")
print("├─────────────────┼─────────────┼───────────────┼─────────────┼─────────────────┤")

for router_name in ["llama3.3", "bert"] + (["mf"] if has_openai else []):
    router_df = df[df["router"] == router_name]
    avg_time = router_df["time_ms"].mean()
    throughput = 1000 / avg_time
    corr, acc = calc_routing_accuracy(router_name)
    
    print(f"│ {router_name:<15} │ {avg_time:>9.1f}ms │ {throughput:>11.2f} │ {corr:>11.3f} │ {acc*100:>13.1f}% │")

print("└─────────────────┴─────────────┴───────────────┴─────────────┴─────────────────┘")

print("""
Interpretation:
- Correlation: How well does the router's score increase with complexity? (1.0 = perfect)
- Classification Accuracy: At threshold 0.5, what % of queries are correctly classified?

Key findings:
- Llama 3.3 70B is SLOWER but may have better semantic understanding
- BERT is FAST (~100x faster than Llama) but uses learned patterns
- MF is slower than BERT (API call) but often more accurate
""")

# Save results
df.to_csv("/home/mehmet/projects/RouteLLM/router_comparison_results.csv", index=False)
print(f"\nResults saved to: router_comparison_results.csv")
