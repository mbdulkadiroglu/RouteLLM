import matplotlib.pyplot as plt
import numpy as np

# --- DATA EXTRACTED FROM YOUR MT BENCH LOGS/PLOT ---

# 1. BASELINES (Weak = Mixtral, Strong = GPT-4)
weak_acc = 8.28
strong_acc = 9.22

# 2. RANDOM ROUTER (Diagonal Line)
# Connects (0, 8.28) to (100, 9.22)
random_x = [0, 100]
random_y = [weak_acc, strong_acc]

# 3. YOUR ROUTER (OLLAMA)
# As noted in your analysis, Ollama (Red) is nearly linear (Random Guessing)
# Points approximated from the red line in your image
ollama_x = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
ollama_y = [8.28, 8.47, 8.66, 8.85, 9.03, 9.22]

# 4. MATRIX FACTORIZATION (MF) - The "Winner"
# The blue line with high convexity
mf_x = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
mf_y = [8.28, 8.72, 8.84, 9.02, 9.08, 9.15, 9.18, 9.17, 9.19, 9.21, 9.22]

# 5. BERT
# The orange line
bert_x = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
bert_y = [8.28, 8.49, 8.79, 9.02, 9.10, 9.16, 9.17, 9.12, 9.13, 9.16, 9.22]

# 6. SW RANKING
# The green line
sw_x = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
sw_y = [8.28, 8.55, 8.66, 8.98, 9.05, 9.11, 9.17, 9.21, 9.22, 9.21, 9.22]

# 7. CAUSAL LLM
# The purple line (S-curve)
causal_x = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
causal_y = [8.28, 8.48, 8.61, 8.73, 8.94, 9.04, 9.13, 9.14, 9.15, 9.20, 9.22]

# --- PLOTTING ---
plt.figure(figsize=(8, 6))

# Plot Random Baseline FIRST (Black Dashed)
plt.plot(random_x, random_y, linestyle='--', color='black', label='Random', linewidth=1.5, alpha=0.6)

# Plot Routers
plt.plot(mf_x, mf_y, marker='o', markersize=4, label='MF', linewidth=2)
plt.plot(bert_x, bert_y, marker='o', markersize=4, label='BERT', linewidth=2)
plt.plot(sw_x, sw_y, marker='o', markersize=4, label='SW Ranking', linewidth=2)
plt.plot(causal_x, causal_y, marker='o', markersize=4, label='Causal LLM', linewidth=2, color='#9467bd')
plt.plot(ollama_x, ollama_y, marker='o', markersize=4, label='Ollama (Ours)', linewidth=2.5, color='#d62728')

# Plot Strong/Weak Horizontal Lines
plt.axhline(y=strong_acc, color='r', linestyle='--', alpha=0.5, label='GPT-4 Only')
plt.axhline(y=weak_acc, color='gray', linestyle='--', alpha=0.5, label='Mixtral Only')

# Styling
plt.title("Router Performance (MT Bench)", fontsize=14)
plt.xlabel("Cost (% Calls to GPT-4)", fontsize=12)
plt.ylabel("MT Bench Score (1-10)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10, loc='lower right')
plt.tight_layout()

# Save
output_file = "mt_bench_with_random.png"
plt.savefig(output_file, dpi=300)
print(f"Plot saved to {output_file}")