import matplotlib.pyplot as plt
import numpy as np

# --- DATA EXTRACTED FROM YOUR GSM8K LOGS ---

# 1. BASELINES
weak_acc = 63.73
strong_acc = 85.77

# 2. RANDOM ROUTER (Diagonal Line)
random_x = [0, 100]
random_y = [weak_acc, strong_acc]

# 3. YOUR ROUTER (OLLAMA)
# Extracted from your specific threshold logs:
# T=0.8 (0%), T=0.6 (67%), T=0.4 (89%), T=0.2 (99%), T=0.1 (100%)
ollama_x = [0.0, 67.25, 89.37, 99.77, 100.0]
ollama_y = [63.73, 81.56, 85.00, 85.85, 85.77]

# 4. CAUSAL LLM (The Specialist)
causal_x = [0.0, 10.02, 20.05, 29.99, 40.02, 50.04, 59.99, 70.01, 79.95, 89.98, 100.0]
causal_y = [63.73, 67.79, 71.00, 73.91, 76.66, 79.19, 81.18, 82.86, 83.70, 85.00, 85.77]

# 5. MATRIX FACTORIZATION (MF)
mf_x = [0.0, 10.02, 20.05, 29.99, 40.02, 50.04, 59.99, 70.01, 79.95, 89.98, 100.0]
mf_y = [63.73, 66.87, 69.24, 71.84, 75.13, 76.59, 78.65, 80.72, 83.17, 84.70, 85.77]

# 6. BERT
bert_x = [0.0, 10.02, 20.05, 29.99, 40.02, 49.96, 59.99, 70.01, 79.95, 89.98, 100.0]
bert_y = [63.73, 66.49, 69.01, 70.93, 73.37, 76.28, 78.19, 79.27, 81.56, 84.47, 85.77]

# 7. SW RANKING
sw_x = [0.0, 10.02, 20.05, 29.99, 40.02, 49.96, 59.99, 70.01, 79.95, 89.98, 100.0]
sw_y = [63.73, 66.49, 69.32, 72.53, 74.60, 77.81, 79.73, 81.26, 82.56, 84.39, 85.77]


# --- PLOTTING ---
plt.figure(figsize=(8, 6))

# Plot Random Baseline FIRST
plt.plot(random_x, random_y, linestyle='--', color='black', label='Random', linewidth=1.5, alpha=0.6)

# Plot Routers
plt.plot(mf_x, mf_y, marker='o', markersize=4, label='MF', linewidth=2)
plt.plot(bert_x, bert_y, marker='o', markersize=4, label='BERT', linewidth=2)
plt.plot(sw_x, sw_y, marker='o', markersize=4, label='SW Ranking', linewidth=2)
plt.plot(causal_x, causal_y, marker='o', markersize=4, label='Causal LLM', linewidth=2, color='#d62728')
plt.plot(ollama_x, ollama_y, marker='o', markersize=4, label='Ollama (Ours)', linewidth=2.5, color='#9467bd')

# Plot Strong/Weak Horizontal Lines
plt.axhline(y=strong_acc, color='r', linestyle='--', alpha=0.5, label='GPT-4 Only')
plt.axhline(y=weak_acc, color='gray', linestyle='--', alpha=0.5, label='Mixtral Only')

# Styling
plt.title("Router Performance (GSM8K)", fontsize=14)
plt.xlabel("Cost (% Calls to GPT-4)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()

# Save
output_file = "gsm8k_with_random.png"
plt.savefig(output_file, dpi=300)
print(f"Plot saved to {output_file}")