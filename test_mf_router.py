#!/usr/bin/env python3
"""
Quick test of the Matrix Factorization router from RouteLLM
"""
import os

# Check API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set!")
    print("Run: export OPENAI_API_KEY='your-key-here'")
    exit(1)

print("Loading MF router (this downloads the model from HuggingFace)...")

from routellm.controller import Controller

# Initialize the controller with the MF router
client = Controller(
    routers=["mf"],
    strong_model="gpt-4-1106-preview",
    weak_model="mixtral-8x7b-instruct-v0.1",
)

print("MF Router loaded successfully!\n")

# Test prompts - some simple, some complex
test_prompts = [
    "What is 2 + 2?",
    "Hello, how are you?",
    "Explain quantum entanglement in detail with mathematical formulations.",
    "Write a Python function to sort a list.",
    "Derive the Navier-Stokes equations from first principles and discuss their implications for turbulence.",
]

print("Testing routing decisions (threshold=0.5):\n")
print("-" * 80)

for prompt in test_prompts:
    # Get the win rate prediction
    win_rate = client.routers["mf"].calculate_strong_win_rate(prompt)
    
    # Determine which model would be chosen at threshold 0.5
    threshold = 0.5
    chosen_model = "STRONG (GPT-4)" if win_rate >= threshold else "WEAK (Mixtral)"
    
    # Truncate long prompts for display
    display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
    
    print(f"Prompt: {display_prompt}")
    print(f"  Strong win rate: {win_rate:.4f}")
    print(f"  Routed to: {chosen_model}")
    print()

print("-" * 80)
print("\nDone! The router is working correctly.")
print("\nTo adjust the routing threshold:")
print("  - Higher threshold (e.g., 0.7) = more queries go to weak model (cheaper)")
print("  - Lower threshold (e.g., 0.3) = more queries go to strong model (better quality)")
