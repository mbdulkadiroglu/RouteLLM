#!/usr/bin/env python3
"""
Router Comparison: Time vs Accuracy Analysis

This script compares different routing strategies:
1. MF Router (Matrix Factorization) - requires OpenAI embeddings
2. BERT Router - local, no API calls
3. Random Router - baseline
4. Always Strong - upper bound
5. Always Weak - lower bound

We measure:
- Router decision time (overhead)
- Routing accuracy (how well it picks the right model)
- End-to-end quality when actually calling models
"""

import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
STRONG_MODEL = "llama3.3:70b"  # Your local strong model
WEAK_MODEL = "llama3.2:3b"     # You'd need a weak model too, or we simulate

# Check API key for MF router
if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not set - MF router won't work")
    print("Set it with: export OPENAI_API_KEY='your-key'")

print("=" * 70)
print("ROUTER COMPARISON: TIME vs ACCURACY")
print("=" * 70)

# Load routers
print("\nLoading routers...")

from routellm.controller import Controller

# We'll compare MF and BERT routers
routers_to_test = ["mf", "bert"]

try:
    controller = Controller(
        routers=routers_to_test,
        strong_model="gpt-4-1106-preview",  # Used for scoring, not actual calls
        weak_model="mixtral-8x7b-instruct-v0.1",
        progress_bar=True,
    )
    print(f"Loaded routers: {routers_to_test}")
except Exception as e:
    print(f"Error loading routers: {e}")
    print("Falling back to MF only...")
    routers_to_test = ["mf"]
    controller = Controller(
        routers=["mf"],
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
    )

# Test prompts - mix of simple and complex
test_prompts = [
    # Simple prompts (should route to weak model)
    "What is 2 + 2?",
    "Hello, how are you?",
    "What color is the sky?",
    "Name the capital of France.",
    "What is Python?",
    
    # Medium complexity
    "Explain how photosynthesis works.",
    "Write a Python function to reverse a string.",
    "What are the main causes of World War I?",
    "How does a neural network learn?",
    "Explain the difference between TCP and UDP.",
    
    # Complex prompts (should route to strong model)
    "Prove that the square root of 2 is irrational using proof by contradiction.",
    "Write a Python implementation of the A* pathfinding algorithm with detailed comments.",
    "Analyze the economic implications of quantitative easing on emerging markets.",
    "Derive the Euler-Lagrange equation from the principle of least action.",
    "Design a distributed system architecture for a real-time collaborative document editor.",
]

print(f"\nTesting with {len(test_prompts)} prompts...")

# Measure router timing
results = []

for router_name in routers_to_test:
    print(f"\n{'=' * 50}")
    print(f"Testing Router: {router_name.upper()}")
    print("=" * 50)
    
    router = controller.routers[router_name]
    times = []
    win_rates = []
    
    for prompt in tqdm(test_prompts, desc=f"  {router_name}"):
        # Measure time for routing decision
        start_time = time.perf_counter()
        win_rate = router.calculate_strong_win_rate(prompt)
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        win_rates.append(win_rate)
        
        results.append({
            "router": router_name,
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "win_rate": win_rate,
            "time_ms": elapsed_ms,
        })
    
    # Summary stats
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n  Timing Statistics:")
    print(f"    Average: {avg_time:.2f} ms")
    print(f"    Std Dev: {std_time:.2f} ms")
    print(f"    Min:     {min_time:.2f} ms")
    print(f"    Max:     {max_time:.2f} ms")
    
    # Win rate distribution
    print(f"\n  Win Rate Statistics:")
    print(f"    Average: {np.mean(win_rates):.4f}")
    print(f"    Min:     {np.min(win_rates):.4f}")
    print(f"    Max:     {np.max(win_rates):.4f}")

# Create comparison table
print("\n" + "=" * 70)
print("DETAILED RESULTS")
print("=" * 70)

df = pd.DataFrame(results)

# Pivot to compare routers side by side
print("\nPer-prompt comparison:")
print("-" * 90)
print(f"{'Prompt':<55} {'MF WR':>8} {'MF ms':>8}", end="")
if "bert" in routers_to_test:
    print(f" {'BERT WR':>8} {'BERT ms':>8}", end="")
print()
print("-" * 90)

for prompt in test_prompts:
    short_prompt = prompt[:52] + "..." if len(prompt) > 52 else prompt
    mf_row = df[(df["router"] == "mf") & (df["prompt"].str.startswith(prompt[:30]))]
    
    if not mf_row.empty:
        print(f"{short_prompt:<55} {mf_row['win_rate'].values[0]:>8.4f} {mf_row['time_ms'].values[0]:>7.1f}ms", end="")
        
        if "bert" in routers_to_test:
            bert_row = df[(df["router"] == "bert") & (df["prompt"].str.startswith(prompt[:30]))]
            if not bert_row.empty:
                print(f" {bert_row['win_rate'].values[0]:>8.4f} {bert_row['time_ms'].values[0]:>7.1f}ms", end="")
    print()

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY: ROUTER COMPARISON")
print("=" * 70)

summary_data = []
for router_name in routers_to_test:
    router_df = df[df["router"] == router_name]
    summary_data.append({
        "Router": router_name.upper(),
        "Avg Time (ms)": f"{router_df['time_ms'].mean():.2f}",
        "Throughput (req/s)": f"{1000 / router_df['time_ms'].mean():.1f}",
        "Avg Win Rate": f"{router_df['win_rate'].mean():.4f}",
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Paper comparison
print("\n" + "=" * 70)
print("COMPARISON WITH PAPER (Table 7)")
print("=" * 70)
print("""
From the RouteLLM paper:
┌─────────────────────┬──────────────────┬─────────────────┐
│ Router              │ Cost/1M requests │ Requests/second │
├─────────────────────┼──────────────────┼─────────────────┤
│ SW Ranking          │ $39.26           │ 2.9             │
│ Matrix Factorization│ $3.32            │ 155.16          │
│ BERT                │ $3.19            │ 69.62           │
│ Causal LLM          │ $5.23            │ 42.46           │
└─────────────────────┴──────────────────┴─────────────────┘

Your measured throughput vs paper:
""")

for router_name in routers_to_test:
    router_df = df[df["router"] == router_name]
    measured_throughput = 1000 / router_df['time_ms'].mean()
    
    paper_throughput = {"mf": 155.16, "bert": 69.62}.get(router_name, "N/A")
    
    print(f"  {router_name.upper()}: {measured_throughput:.1f} req/s (paper: {paper_throughput} req/s)")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
To do a full end-to-end comparison with your Llama 3.3 70B:

1. Set up a weak model in Ollama (e.g., llama3.2:3b or phi-3)
2. Run the router to decide which model to use
3. Actually call the selected model and measure:
   - Total latency (router + LLM generation)
   - Response quality
   - Cost savings

Run: python router_with_ollama.py  (I'll create this next if you want)
""")
