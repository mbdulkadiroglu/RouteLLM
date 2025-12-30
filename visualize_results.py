#!/usr/bin/env python3
"""
Create visualizations for Router Evaluation Results

This script creates plots showing:
1. Time vs Accuracy tradeoff
2. Cost vs Accuracy curves
3. Pareto frontier analysis
"""

import matplotlib.pyplot as plt
import numpy as np

# Results from 300 GSM8K samples evaluation
routers = {
    "Random": {
        "time_ms": 0.00,
        "thresholds": {
            0.1: {"accuracy": 81.7, "cost": 90.0},
            0.3: {"accuracy": 79.0, "cost": 69.0},
            0.5: {"accuracy": 74.3, "cost": 51.7},
            0.7: {"accuracy": 71.7, "cost": 34.0},
            0.9: {"accuracy": 68.3, "cost": 11.7},
        }
    },
    "MF (Matrix Factorization)": {
        "time_ms": 173.16,
        "thresholds": {
            0.1: {"accuracy": 83.3, "cost": 100.0},
            0.3: {"accuracy": 70.0, "cost": 23.7},
            0.5: {"accuracy": 65.3, "cost": 0.0},
            0.7: {"accuracy": 65.3, "cost": 0.0},
            0.9: {"accuracy": 65.3, "cost": 0.0},
        }
    },
    "BERT": {
        "time_ms": 14.14,
        "thresholds": {
            0.1: {"accuracy": 83.3, "cost": 100.0},
            0.3: {"accuracy": 83.3, "cost": 99.3},
            0.5: {"accuracy": 77.0, "cost": 59.3},
            0.7: {"accuracy": 66.0, "cost": 3.7},
            0.9: {"accuracy": 65.3, "cost": 0.0},
        }
    },
    "Llama 3.3 70B": {
        "time_ms": 361.13,
        "thresholds": {
            0.1: {"accuracy": 83.3, "cost": 100.0},
            0.3: {"accuracy": 82.3, "cost": 72.0},
            0.5: {"accuracy": 74.7, "cost": 25.0},
            0.7: {"accuracy": 65.3, "cost": 2.0},
            0.9: {"accuracy": 65.3, "cost": 0.0},
        }
    },
}

# Baselines
gpt4_accuracy = 83.3
mixtral_accuracy = 65.3

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Color scheme
colors = {
    "Random": "#888888",
    "MF (Matrix Factorization)": "#e74c3c",
    "BERT": "#3498db",
    "Llama 3.3 70B": "#2ecc71",
}

markers = {
    "Random": "s",
    "MF (Matrix Factorization)": "o",
    "BERT": "^",
    "Llama 3.3 70B": "D",
}

# ============================================================
# Plot 1: Cost vs Accuracy (Main Result)
# ============================================================
ax1 = axes[0, 0]

for router_name, data in routers.items():
    costs = [data["thresholds"][t]["cost"] for t in sorted(data["thresholds"].keys())]
    accs = [data["thresholds"][t]["accuracy"] for t in sorted(data["thresholds"].keys())]
    
    ax1.plot(costs, accs, 
             marker=markers[router_name], 
             color=colors[router_name],
             label=router_name,
             linewidth=2,
             markersize=8)

# Add baseline references
ax1.axhline(y=gpt4_accuracy, color='darkgreen', linestyle='--', alpha=0.7, label=f'GPT-4 Only ({gpt4_accuracy}%)')
ax1.axhline(y=mixtral_accuracy, color='darkred', linestyle='--', alpha=0.7, label=f'Mixtral Only ({mixtral_accuracy}%)')

ax1.set_xlabel('GPT-4 Calls (%)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Router Performance: Cost vs Accuracy\n(GSM8K, 300 samples)', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-5, 105)
ax1.set_ylim(60, 90)

# ============================================================
# Plot 2: Routing Time Comparison
# ============================================================
ax2 = axes[0, 1]

router_names = list(routers.keys())
times = [routers[r]["time_ms"] for r in router_names]
bar_colors = [colors[r] for r in router_names]

bars = ax2.bar(router_names, times, color=bar_colors, edgecolor='black')
ax2.set_ylabel('Routing Time (ms)', fontsize=12)
ax2.set_title('Average Routing Time per Query', fontsize=14)
ax2.set_yscale('log')

# Add value labels
for bar, time in zip(bars, times):
    height = bar.get_height()
    if height > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}ms',
                ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(bar.get_x() + bar.get_width()/2., 0.01,
                f'{time:.2f}ms',
                ha='center', va='bottom', fontsize=10)

ax2.set_ylim(0.001, 1000)
plt.setp(ax2.get_xticklabels(), rotation=15, ha='right')

# ============================================================
# Plot 3: Time-Accuracy Tradeoff at 50% Cost
# ============================================================
ax3 = axes[1, 0]

# Get accuracy at ~50% cost threshold
for router_name, data in routers.items():
    time = data["time_ms"]
    # Find threshold closest to 50% cost
    best_t = 0.5
    acc = data["thresholds"][best_t]["accuracy"]
    cost = data["thresholds"][best_t]["cost"]
    
    ax3.scatter(time if time > 0 else 0.01, acc, 
                s=200, 
                c=colors[router_name],
                marker=markers[router_name],
                label=f'{router_name}\n({cost:.0f}% cost)')

ax3.set_xlabel('Routing Time (ms, log scale)', fontsize=12)
ax3.set_ylabel('Accuracy (%)', fontsize=12)
ax3.set_title('Time vs Accuracy Tradeoff\n(at threshold=0.5)', fontsize=14)
ax3.set_xscale('log')
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0.001, 1000)

# ============================================================
# Plot 4: Efficiency Score (Accuracy per millisecond of routing)
# ============================================================
ax4 = axes[1, 1]

# Calculate efficiency: accuracy gain over baseline per ms of routing
efficiency_data = []
for router_name, data in routers.items():
    time = data["time_ms"] if data["time_ms"] > 0 else 0.001
    acc_at_50 = data["thresholds"][0.5]["accuracy"]
    cost_at_50 = data["thresholds"][0.5]["cost"]
    
    # Efficiency = (accuracy - mixtral baseline) / (routing time + cost overhead)
    accuracy_gain = acc_at_50 - mixtral_accuracy
    # Approximate cost: routing time + proportional API cost
    total_overhead = time  # just routing time for now
    
    efficiency = accuracy_gain / total_overhead if total_overhead > 0 else accuracy_gain * 1000
    efficiency_data.append((router_name, efficiency, acc_at_50, cost_at_50))

efficiency_data.sort(key=lambda x: x[1], reverse=True)

router_names_sorted = [x[0] for x in efficiency_data]
efficiencies = [x[1] for x in efficiency_data]
bar_colors_sorted = [colors[r] for r in router_names_sorted]

bars = ax4.bar(router_names_sorted, efficiencies, color=bar_colors_sorted, edgecolor='black')
ax4.set_ylabel('Accuracy Gain / Routing Time\n(% accuracy / ms)', fontsize=12)
ax4.set_title('Routing Efficiency Score', fontsize=14)
ax4.set_yscale('log')

# Add annotations
for i, (bar, (name, eff, acc, cost)) in enumerate(zip(bars, efficiency_data)):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
            f'{acc:.1f}% acc\n{cost:.0f}% cost',
            ha='center', va='bottom', fontsize=9)

plt.setp(ax4.get_xticklabels(), rotation=15, ha='right')

plt.tight_layout()
plt.savefig('/home/mehmet/projects/RouteLLM/router_evaluation_results.png', dpi=150, bbox_inches='tight')
print("Saved plot to router_evaluation_results.png")

# ============================================================
# Summary Table
# ============================================================
print("\n" + "="*80)
print("ROUTER COMPARISON SUMMARY")
print("="*80)
print("\nGSM8K Benchmark (300 samples)")
print(f"Baseline: GPT-4 = {gpt4_accuracy}%, Mixtral = {mixtral_accuracy}%")
print()
print("-"*80)
print(f"{'Router':<25} {'Time (ms)':<12} {'Acc@50% cost':<15} {'Cost@50%':<12} {'Best For'}")
print("-"*80)

recommendations = {
    "Random": "Quick baseline, no routing overhead",
    "MF (Matrix Factorization)": "Simple queries (sends most to weak model)",
    "BERT": "Best quality with moderate latency",
    "Llama 3.3 70B": "High accuracy with lowest cost (but slow)",
}

for router_name, data in routers.items():
    time = data["time_ms"]
    acc = data["thresholds"][0.5]["accuracy"]
    cost = data["thresholds"][0.5]["cost"]
    rec = recommendations[router_name]
    print(f"{router_name:<25} {time:>8.2f}ms   {acc:>10.1f}%      {cost:>8.1f}%    {rec}")

print("-"*80)
print("\nKEY INSIGHTS:")
print("1. BERT router offers best accuracy (77.0%) with fast routing (14ms)")
print("2. Llama 3.3 70B achieves 74.7% accuracy using only 25% GPT-4 calls")
print("3. MF router is too conservative - sends almost everything to weak model")
print("4. At threshold=0.5: BERT saves 40% cost with only 6% accuracy drop")
print("="*80)

plt.show()
