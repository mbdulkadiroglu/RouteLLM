#!/usr/bin/env python3
"""
Comprehensive Router Evaluation

Uses the paper's pre-computed benchmark responses to compare:
1. MF (Matrix Factorization) router
2. BERT classifier router 
3. SW Ranking router
4. Causal LLM router
5. Llama 3.3 70B as router (via Ollama)
6. Random baseline

Measures:
- Routing decision time
- Accuracy at various cost thresholds
- Win rate on model selection
"""

import time
import json
import csv
import numpy as np
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Data Loading
# ============================================================

def load_gsm8k_data(max_samples: int = 200) -> list[dict]:
    """Load GSM8K pre-computed responses."""
    responses_file = Path(__file__).parent / "routellm/evals/gsm8k/gsm8k_responses.csv"
    
    data = []
    with open(responses_file, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            data.append({
                "prompt": row["prompt"],
                "weak_correct": row["mistralai/Mixtral-8x7B-Instruct-v0.1"] == "True",
                "strong_correct": row["gpt-4-1106-preview"] == "True",
            })
    return data

def load_mtbench_data(max_samples: int = 200) -> list[dict]:
    """Load MT-Bench judgements."""
    judgements_file = Path(__file__).parent / "routellm/evals/mt_bench/judgements.jsonl"
    
    # Group by question_id - we need both model scores
    scores = {}  # question_id -> {model -> score}
    prompts = {}  # question_id -> prompt
    
    with open(judgements_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["question_id"]
            model = entry["model"]
            score = entry["score"]
            
            if qid not in scores:
                scores[qid] = {}
            scores[qid][model] = score
            
            # Extract prompt from user_prompt
            if qid not in prompts:
                user_prompt = entry.get("user_prompt", "")
                # Extract the actual question
                if "### User:" in user_prompt:
                    parts = user_prompt.split("### User:")
                    if len(parts) > 1:
                        prompt = parts[1].split("### Assistant")[0].strip()
                        prompts[qid] = prompt
    
    # Convert to list
    data = []
    for qid in list(scores.keys())[:max_samples]:
        if "gpt-4-1106-preview" in scores[qid] and "mistralai/Mixtral-8x7B-Instruct-v0.1" in scores.get(qid, {}):
            data.append({
                "prompt": prompts.get(qid, f"Question {qid}"),
                "question_id": qid,
                "weak_score": scores[qid].get("mistralai/Mixtral-8x7B-Instruct-v0.1", 0),
                "strong_score": scores[qid].get("gpt-4-1106-preview", 0),
            })
    
    # Fallback: if Mixtral not in MT-Bench, use different models
    if len(data) == 0:
        models_found = set()
        for qid, model_scores in scores.items():
            models_found.update(model_scores.keys())
        print(f"Models in MT-Bench: {models_found}")
        
        # Try llama-2-70b-chat as weak model
        for qid in list(scores.keys())[:max_samples]:
            if "gpt-4-1106-preview" in scores[qid]:
                # Find any weaker model
                weak_model = None
                weak_score = 0
                for model, score in scores[qid].items():
                    if model != "gpt-4-1106-preview" and score < scores[qid]["gpt-4-1106-preview"]:
                        weak_model = model
                        weak_score = score
                        break
                
                if weak_model:
                    data.append({
                        "prompt": prompts.get(qid, f"Question {qid}"),
                        "question_id": qid,
                        "weak_score": weak_score,
                        "strong_score": scores[qid]["gpt-4-1106-preview"],
                        "weak_model": weak_model,
                    })
    
    return data

# ============================================================
# Router Implementations
# ============================================================

class BaseRouter:
    """Base class for routers."""
    name = "base"
    
    def route(self, prompt: str) -> tuple[str, float]:
        """
        Route a prompt.
        Returns: (model_choice, routing_time_seconds)
        model_choice is 'strong' or 'weak'
        """
        raise NotImplementedError
    
    def get_score(self, prompt: str) -> tuple[float, float]:
        """
        Get routing score for a prompt.
        Returns: (score, routing_time_seconds)
        Higher score = more likely to need strong model
        """
        raise NotImplementedError


class RandomRouter(BaseRouter):
    name = "random"
    
    def __init__(self, strong_prob: float = 0.5):
        self.strong_prob = strong_prob
    
    def get_score(self, prompt: str) -> tuple[float, float]:
        start = time.perf_counter()
        score = np.random.random()
        elapsed = time.perf_counter() - start
        return score, elapsed


class MFRouter(BaseRouter):
    """Matrix Factorization router using RouteLLM."""
    name = "mf"
    
    def __init__(self):
        from routellm.controller import Controller
        import os
        
        # Load controller with MF router
        self.controller = Controller(
            routers=["mf"],
            strong_model="gpt-4-1106-preview",
            weak_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        self.router = self.controller.routers["mf"]
    
    def get_score(self, prompt: str) -> tuple[float, float]:
        start = time.perf_counter()
        # Truncate prompt to avoid API errors (max ~8000 chars for embedding)
        truncated_prompt = prompt[:4000] if len(prompt) > 4000 else prompt
        try:
            # Router expects a raw string, not message list
            score = self.router.calculate_strong_win_rate(truncated_prompt)
        except Exception as e:
            print(f"      Warning: MF router error: {e}")
            score = 0.5  # Default to middle score on error
        elapsed = time.perf_counter() - start
        return score, elapsed


class BERTRouter(BaseRouter):
    """BERT classifier router."""
    name = "bert"
    
    def __init__(self):
        from routellm.controller import Controller
        
        self.controller = Controller(
            routers=["bert"],
            strong_model="gpt-4-1106-preview",
            weak_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        self.router = self.controller.routers["bert"]
    
    def get_score(self, prompt: str) -> tuple[float, float]:
        start = time.perf_counter()
        # Truncate prompt to avoid issues (BERT has 512 token limit)
        truncated_prompt = prompt[:2000] if len(prompt) > 2000 else prompt
        try:
            # Router expects a raw string, not message list
            score = self.router.calculate_strong_win_rate(truncated_prompt)
        except Exception as e:
            print(f"      Warning: BERT router error: {e}")
            score = 0.5
        elapsed = time.perf_counter() - start
        return score, elapsed


class SWRankingRouter(BaseRouter):
    """Similarity-Weighted Ranking router using HuggingFace datasets.
    
    Uses arena battle data from:
    - lmsys/lmsys-arena-human-preference-55k
    - routellm/gpt4_judge_battles
    
    And embeddings from:
    - routellm/arena_battles_embeddings
    - routellm/gpt4_judge_battles_embeddings
    """
    name = "sw_ranking"
    
    def __init__(self):
        from routellm.controller import Controller
        
        self.controller = Controller(
            routers=["sw_ranking"],
            strong_model="gpt-4-1106-preview",
            weak_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        self.router = self.controller.routers["sw_ranking"]
    
    def get_score(self, prompt: str) -> tuple[float, float]:
        start = time.perf_counter()
        # Truncate prompt for embedding API
        truncated_prompt = prompt[:4000] if len(prompt) > 4000 else prompt
        try:
            score = self.router.calculate_strong_win_rate(truncated_prompt)
        except Exception as e:
            print(f"      Warning: SW Ranking router error: {e}")
            score = 0.5
        elapsed = time.perf_counter() - start
        return score, elapsed


class CausalLLMRouter(BaseRouter):
    """Causal LLM router using fine-tuned Llama model from HuggingFace.
    
    Uses checkpoint from: routellm/causal_llm_gpt4_augmented
    Based on meta-llama/Meta-Llama-3-8B
    """
    name = "causal_llm"
    
    def __init__(self):
        from routellm.controller import Controller
        
        self.controller = Controller(
            routers=["causal_llm"],
            strong_model="gpt-4-1106-preview",
            weak_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        self.router = self.controller.routers["causal_llm"]
    
    def get_score(self, prompt: str) -> tuple[float, float]:
        start = time.perf_counter()
        # Truncate prompt to reasonable length
        truncated_prompt = prompt[:2000] if len(prompt) > 2000 else prompt
        try:
            score = self.router.calculate_strong_win_rate(truncated_prompt)
        except Exception as e:
            print(f"      Warning: Causal LLM router error: {e}")
            score = 0.5
        elapsed = time.perf_counter() - start
        return score, elapsed


class LlamaRouter(BaseRouter):
    """Use Llama 3.3 70B via Ollama as a router."""
    name = "llama3.3-70b"
    
    def __init__(self, model: str = "llama3.3:70b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
        
        self.system_prompt = """You are a routing classifier. Your job is to determine if a query requires a powerful AI model (like GPT-4) or if a simpler model can handle it.

Rate the query complexity from 0.0 to 1.0:
- 0.0-0.3: Simple queries (basic facts, simple math, straightforward questions)
- 0.3-0.6: Moderate queries (some reasoning, multi-step problems)
- 0.6-1.0: Complex queries (advanced reasoning, creative writing, complex analysis)

Respond with ONLY a single number between 0.0 and 1.0. No other text."""

    def get_score(self, prompt: str) -> tuple[float, float]:
        start = time.perf_counter()
        
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": f"{self.system_prompt}\n\nQuery to classify:\n{prompt[:500]}",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 10}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "0.5").strip()
                # Extract number from response
                try:
                    score = float(text.split()[0].strip())
                    score = max(0.0, min(1.0, score))
                except:
                    score = 0.5
            else:
                score = 0.5
        except Exception as e:
            score = 0.5
        
        elapsed = time.perf_counter() - start
        return score, elapsed


# ============================================================
# Evaluation
# ============================================================

@dataclass
class EvalResult:
    router_name: str
    avg_routing_time_ms: float
    std_routing_time_ms: float
    # For different thresholds, what's the accuracy and cost?
    threshold_results: dict  # threshold -> {accuracy, strong_pct, score}


def evaluate_router_on_gsm8k(router: BaseRouter, data: list[dict], thresholds: list[float]) -> EvalResult:
    """Evaluate a router on GSM8K data."""
    
    routing_times = []
    scores = []
    
    print(f"\n  Evaluating {router.name} router on {len(data)} samples...")
    
    for i, sample in enumerate(data):
        if i % 50 == 0:
            print(f"    Progress: {i}/{len(data)}")
        
        score, elapsed = router.get_score(sample["prompt"])
        scores.append(score)
        routing_times.append(elapsed * 1000)  # Convert to ms
    
    # For each threshold, compute accuracy
    threshold_results = {}
    
    for threshold in thresholds:
        correct = 0
        strong_calls = 0
        
        for i, sample in enumerate(data):
            use_strong = scores[i] >= threshold
            
            if use_strong:
                strong_calls += 1
                # Using strong model - check if strong is correct
                if sample["strong_correct"]:
                    correct += 1
            else:
                # Using weak model - check if weak is correct
                if sample["weak_correct"]:
                    correct += 1
        
        accuracy = correct / len(data) if data else 0
        strong_pct = strong_calls / len(data) if data else 0
        
        threshold_results[threshold] = {
            "accuracy": accuracy,
            "strong_pct": strong_pct,
        }
    
    return EvalResult(
        router_name=router.name,
        avg_routing_time_ms=np.mean(routing_times),
        std_routing_time_ms=np.std(routing_times),
        threshold_results=threshold_results,
    )


def evaluate_router_on_mtbench(router: BaseRouter, data: list[dict], thresholds: list[float]) -> EvalResult:
    """Evaluate a router on MT-Bench data."""
    
    routing_times = []
    scores = []
    
    print(f"\n  Evaluating {router.name} router on {len(data)} samples...")
    
    for i, sample in enumerate(data):
        if i % 50 == 0:
            print(f"    Progress: {i}/{len(data)}")
        
        score, elapsed = router.get_score(sample["prompt"])
        scores.append(score)
        routing_times.append(elapsed * 1000)
    
    # For each threshold, compute weighted average score
    threshold_results = {}
    
    for threshold in thresholds:
        total_score = 0
        strong_calls = 0
        
        for i, sample in enumerate(data):
            use_strong = scores[i] >= threshold
            
            if use_strong:
                strong_calls += 1
                total_score += sample["strong_score"]
            else:
                total_score += sample["weak_score"]
        
        avg_score = total_score / len(data) if data else 0
        strong_pct = strong_calls / len(data) if data else 0
        
        threshold_results[threshold] = {
            "avg_score": avg_score,
            "strong_pct": strong_pct,
        }
    
    return EvalResult(
        router_name=router.name,
        avg_routing_time_ms=np.mean(routing_times),
        std_routing_time_ms=np.std(routing_times),
        threshold_results=threshold_results,
    )


def print_gsm8k_results(results: list[EvalResult], data: list[dict]):
    """Print GSM8K evaluation results."""
    
    # Compute baseline accuracies
    all_strong_acc = sum(1 for d in data if d["strong_correct"]) / len(data)
    all_weak_acc = sum(1 for d in data if d["weak_correct"]) / len(data)
    
    print("\n" + "="*80)
    print("GSM8K EVALUATION RESULTS")
    print("="*80)
    print(f"\nBaselines:")
    print(f"  Always Strong (GPT-4):   {all_strong_acc*100:.1f}% accuracy, 100% cost")
    print(f"  Always Weak (Mixtral):   {all_weak_acc*100:.1f}% accuracy, 0% cost")
    
    print("\n" + "-"*80)
    print("Routing Times:")
    print("-"*80)
    for result in results:
        print(f"  {result.router_name:20s}: {result.avg_routing_time_ms:8.2f}ms ± {result.std_routing_time_ms:.2f}ms")
    
    print("\n" + "-"*80)
    print("Accuracy vs Cost Tradeoff (at various thresholds):")
    print("-"*80)
    
    # Print table header
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"{'Router':<20}", end="")
    for t in thresholds:
        print(f"  t={t} (acc/cost)", end="")
    print()
    print("-"*80)
    
    for result in results:
        print(f"{result.router_name:<20}", end="")
        for t in thresholds:
            if t in result.threshold_results:
                acc = result.threshold_results[t]["accuracy"] * 100
                cost = result.threshold_results[t]["strong_pct"] * 100
                print(f"  {acc:5.1f}%/{cost:4.1f}%  ", end="")
            else:
                print("       N/A       ", end="")
        print()
    
    print("\n" + "-"*80)
    print("Best Router at Each Cost Level:")
    print("-"*80)
    
    for t in thresholds:
        best_router = None
        best_acc = 0
        for result in results:
            if t in result.threshold_results:
                acc = result.threshold_results[t]["accuracy"]
                if acc > best_acc:
                    best_acc = acc
                    best_router = result.router_name
        
        cost = results[0].threshold_results[t]["strong_pct"] * 100 if results else 0
        print(f"  t={t}: Best = {best_router} with {best_acc*100:.1f}% accuracy at ~{cost:.0f}% GPT-4 cost")


def print_mtbench_results(results: list[EvalResult], data: list[dict]):
    """Print MT-Bench evaluation results."""
    
    all_strong_score = np.mean([d["strong_score"] for d in data]) if data else 0
    all_weak_score = np.mean([d["weak_score"] for d in data]) if data else 0
    
    print("\n" + "="*80)
    print("MT-BENCH EVALUATION RESULTS")
    print("="*80)
    print(f"\nBaselines:")
    print(f"  Always Strong (GPT-4):   {all_strong_score:.2f} avg score, 100% cost")
    print(f"  Always Weak:             {all_weak_score:.2f} avg score, 0% cost")
    
    print("\n" + "-"*80)
    print("Routing Times:")
    print("-"*80)
    for result in results:
        print(f"  {result.router_name:20s}: {result.avg_routing_time_ms:8.2f}ms ± {result.std_routing_time_ms:.2f}ms")
    
    print("\n" + "-"*80)
    print("Score vs Cost Tradeoff:")
    print("-"*80)
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"{'Router':<20}", end="")
    for t in thresholds:
        print(f"  t={t} (score/cost)", end="")
    print()
    print("-"*80)
    
    for result in results:
        print(f"{result.router_name:<20}", end="")
        for t in thresholds:
            if t in result.threshold_results:
                score = result.threshold_results[t]["avg_score"]
                cost = result.threshold_results[t]["strong_pct"] * 100
                print(f"  {score:5.2f}/{cost:4.1f}%  ", end="")
            else:
                print("       N/A       ", end="")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive router evaluation")
    parser.add_argument("--benchmark", choices=["gsm8k", "mtbench"], default="gsm8k",
                        help="Which benchmark to run")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--routers", nargs="+", 
                        default=["random", "mf", "bert", "sw_ranking", "causal_llm", "llama"],
                        help="Which routers to evaluate: random, mf, bert, sw_ranking, causal_llm, llama")
    args = parser.parse_args()
    
    print("="*80)
    print("COMPREHENSIVE ROUTER EVALUATION")
    print("="*80)
    print(f"Benchmark: {args.benchmark}")
    print(f"Samples: {args.samples}")
    print(f"Routers: {args.routers}")
    
    # Initialize routers
    routers = []
    
    print("\nInitializing routers...")
    
    if "random" in args.routers:
        print("  - Random router")
        routers.append(RandomRouter())
    
    if "mf" in args.routers:
        print("  - MF router (loading...)")
        try:
            routers.append(MFRouter())
            print("    ✓ MF router loaded")
        except Exception as e:
            print(f"    ✗ Failed to load MF router: {e}")
    
    if "bert" in args.routers:
        print("  - BERT router (loading...)")
        try:
            routers.append(BERTRouter())
            print("    ✓ BERT router loaded")
        except Exception as e:
            print(f"    ✗ Failed to load BERT router: {e}")
    
    if "sw_ranking" in args.routers:
        print("  - SW Ranking router (loading HuggingFace datasets...)")
        print("    Uses: lmsys/lmsys-arena-human-preference-55k")
        print("    Uses: routellm/gpt4_judge_battles")
        print("    Uses: routellm/arena_battles_embeddings")
        print("    Uses: routellm/gpt4_judge_battles_embeddings")
        try:
            routers.append(SWRankingRouter())
            print("    ✓ SW Ranking router loaded")
        except Exception as e:
            print(f"    ✗ Failed to load SW Ranking router: {e}")
    
    if "causal_llm" in args.routers:
        print("  - Causal LLM router (loading from HuggingFace...)")
        print("    Uses: routellm/causal_llm_gpt4_augmented")
        print("    Based on: meta-llama/Meta-Llama-3-8B")
        try:
            routers.append(CausalLLMRouter())
            print("    ✓ Causal LLM router loaded")
        except Exception as e:
            print(f"    ✗ Failed to load Causal LLM router: {e}")
    
    if "llama" in args.routers:
        print("  - Llama 3.3 70B router (via Ollama)")
        try:
            llama_router = LlamaRouter()
            # Test connection
            score, _ = llama_router.get_score("Test prompt")
            routers.append(llama_router)
            print("    ✓ Llama router connected")
        except Exception as e:
            print(f"    ✗ Failed to connect to Llama: {e}")
    
    # Load data
    print("\nLoading benchmark data...")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    if args.benchmark == "gsm8k":
        data = load_gsm8k_data(max_samples=args.samples)
        print(f"  Loaded {len(data)} GSM8K samples")
        
        # Evaluate each router
        results = []
        for router in routers:
            result = evaluate_router_on_gsm8k(router, data, thresholds)
            results.append(result)
        
        print_gsm8k_results(results, data)
        
    elif args.benchmark == "mtbench":
        data = load_mtbench_data(max_samples=args.samples)
        print(f"  Loaded {len(data)} MT-Bench samples")
        
        if len(data) == 0:
            print("  No MT-Bench data available with required model pairs!")
            return
        
        # Evaluate each router
        results = []
        for router in routers:
            result = evaluate_router_on_mtbench(router, data, thresholds)
            results.append(result)
        
        print_mtbench_results(results, data)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Time vs Quality Tradeoff")
    print("="*80)
    
    # Find router with best quality/time ratio at 50% cost threshold
    print("\nAt threshold=0.5 (moderate cost):")
    for result in results:
        if 0.5 in result.threshold_results:
            tr = result.threshold_results[0.5]
            if args.benchmark == "gsm8k":
                quality = tr["accuracy"] * 100
                quality_label = "accuracy"
            else:
                quality = tr.get("avg_score", 0)
                quality_label = "score"
            cost = tr["strong_pct"] * 100
            time_ms = result.avg_routing_time_ms
            
            print(f"  {result.router_name:20s}: {quality:.1f}% {quality_label}, {cost:.1f}% cost, {time_ms:.1f}ms routing time")


if __name__ == "__main__":
    main()
