import time
import pandas as pd
import torch
import os
import requests
from tqdm import tqdm
from routellm.controller import Controller

# --- CONFIGURATION ---
DATA_PATH = "routellm/evals/gsm8k/gsm8k_responses.csv"
NUM_SAMPLES = 50 
# The full list of routers from the paper + yours
ROUTERS = ["mf", "bert", "sw_ranking", "ollama", "causal_llm"] 

# Setup environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_real_prompts():
    print(f"Loading real prompts from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    prompts = df["prompt"].dropna().tolist()
    return prompts[:NUM_SAMPLES]

def unload_ollama_model(model_name="llama3.3"):
    """Forces Ollama to release VRAM so Causal LLM can load."""
    print("  > Attempting to unload Ollama model to free VRAM...")
    try:
        # Setting keep_alive=0 forces immediate unload
        requests.post("http://localhost:11434/api/chat", json={
            "model": model_name,
            "messages": [],
            "keep_alive": 0 
        })
        time.sleep(3) # Give it a moment to clear
        print("  > Ollama unloaded.")
    except Exception as e:
        print(f"  > Warning: Could not auto-unload Ollama: {e}")

def benchmark_router(name, prompts):
    print(f"\n--- Benchmarking: {name.upper()} ---")
    
    # 1. SPECIAL MEMORY HANDLING
    # Before loading Causal LLM (which needs VRAM), kill Ollama
    if name == "causal_llm":
        unload_ollama_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        # 2. INITIALIZE ROUTER
        # Note: sw_ranking needs OPENAI_API_KEY env var
        controller = Controller(
            routers=[name],
            strong_model="gpt-4-1106-preview",
            weak_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            progress_bar=False
        )
        router = controller.routers[name]
        
        # --- NEW: FORCE GPU FOR MF/BERT ---
        if name in ["mf", "bert"]:
            if torch.cuda.is_available():
                device = "cuda"
                print(f"  > Moving {name} to {device}...")
                try:
                    # Move all model components to GPU
                    if hasattr(router, 'model'):
                        router.model.to(device)
                        router.model.eval()  # Set to eval mode to avoid dropout issues
                    if hasattr(router, 'classifier'):
                        router.classifier.to(device)
                    if hasattr(router, 'tokenizer'):
                        # Some tokenizers have model_max_length or other configs
                        # but don't need .to(device)
                        pass
                    # CRITICAL: Update the router's device attribute if it exists
                    if hasattr(router, 'device'):
                        router.device = torch.device(device)
                    # Also try to set a _device attribute that some implementations use
                    if not hasattr(router, 'device'):
                        router.device = torch.device(device)
                    print(f"  > Successfully moved {name} to {device}")
                except Exception as e:
                    print(f"  ! Could not move to GPU: {e}")
        # ----------------------------------
        
        # 3. WARMUP
        try:
            router.calculate_strong_win_rate(prompts[0])
        except Exception as e:
            print(f"  ! Warmup failed (Check API Keys/VRAM): {e}")
            return None
        
        # 4. TIMING LOOP
        start_time = time.perf_counter()
        for prompt in tqdm(prompts, desc="Routing"):
            _ = router.calculate_strong_win_rate(prompt)
        end_time = time.perf_counter()
        
        # 5. CALCULATE METRICS
        total_time = end_time - start_time
        avg_latency = (total_time / len(prompts)) * 1000  # ms
        throughput = len(prompts) / total_time             # req/s
        
        print(f"  > Latency:    {avg_latency:.2f} ms/req")
        print(f"  > Throughput: {throughput:.2f} req/s")
        
        # 6. CLEANUP
        del router
        del controller
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {"Router": name, "Latency (ms)": avg_latency, "Throughput (req/s)": throughput}

    except Exception as e:
        print(f"  ! Failed to load/run {name}: {e}")
        return None

if __name__ == "__main__":
    prompts = load_real_prompts()
    print(f"Benchmarking on {len(prompts)} real GSM8K questions.")
    
    results = []
    for r in ROUTERS:
        res = benchmark_router(r, prompts)
        if res:
            results.append(res)

    print("\n\nFINAL SPEED RESULTS")
    print("="*60)
    df = pd.DataFrame(results)
    # Reorder columns for readability
    print(df[["Router", "Latency (ms)", "Throughput (req/s)"]].to_string(index=False))
    print("="*60)