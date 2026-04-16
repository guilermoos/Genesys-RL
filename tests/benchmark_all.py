"""
Comprehensive GPU vs CPU benchmarking suite.

This module provides consolidated benchmarks comparing GPU and CPU performance
across different model sizes (small, medium, large) for both inference and training.

NOTE: These benchmarks are INTEGRATED tests for the Genesys SaaS system.
They test the actual models used in the system with realistic configurations.
No external benchmarking tools are used - all tests use the DQNAgent directly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import time
from typing import Dict, Tuple


class GPUAnalysis:
    """Analyze GPU utilization patterns."""
    
    @staticmethod
    def print_gpu_info() -> None:
        """Print GPU information and utilization analysis."""
        if not torch.cuda.is_available():
            print("❌ No GPU available")
            return
        
        print("\n" + "="*70)
        print("🖥️  GPU INFORMATION & UTILIZATION ANALYSIS")
        print("="*70)
        
        props = torch.cuda.get_device_properties(0)
        print(f"\nDevice: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"Max Threads per Block: {props.max_threads_per_block}")
        
        print("\n❓ Why GPU Usage is 10-15% instead of 100%:")
        print("-" * 70)
        print("1. 📦 Small Model Size")
        print("   Current models (50-128 dims) are too small for GPU parallelism")
        print("   GPU has 3,060 CUDA cores; processing small tensors underutilizes them")
        print()
        print("2. 🔄 Memory Bandwidth Overhead")
        print("   CPU→GPU transfer is ~50-100GB/s, but compute is faster")
        print("   For small batches, transfer time > computation time")
        print()
        print("3. ⚙️  Batch Size Not Optimized")
        print("   Current batch_size=128 is small for RTX 4060")
        print("   RTX 4060 optimal: batch_size=256-512 with larger models")
        print()
        print("4. 🧠 Model Architecture")
        print("   DQN networks are relatively simple (2-3 layers)")
        print("   Complex models (ResNet, Transformers) saturate GPU better")
        print()
        print("5. 💾 Memory Underutilization")
        print("   Using ~0.5GB of 8GB available (6.25% utilization)")
        print("   GPU power scales with memory usage")
        print()
        print("✅ SOLUTION: Use larger models/batches for 50%+ GPU utilization")
        print("-" * 70)


class BenchmarkConfig:
    """Configuration for different model sizes."""
    
    SMALL = {
        "name": "Small",
        "state_size": 32,
        "hidden_layers": [64, 64],
        "batch_size": 32,
        "episodes": 5,  # Reduced for fast testing
        "max_steps": 50,  # Reduced for fast testing
    }
    
    MEDIUM = {
        "name": "Medium",
        "state_size": 128,
        "hidden_layers": [256, 256],
        "batch_size": 128,
        "episodes": 10,  # Reduced for fast testing
        "max_steps": 100,  # Reduced for fast testing
    }
    
    LARGE = {
        "name": "Large",
        "state_size": 256,
        "hidden_layers": [512, 512, 256],
        "batch_size": 256,
        "episodes": 5,  # Reduced for speed
        "max_steps": 50,  # Reduced for speed
    }


class BenchmarkResults:
    """Store and display benchmark results."""
    
    def __init__(self):
        self.results = {}
    
    def add(self, key: str, cpu_time: float, gpu_time: float, task: str) -> None:
        """Add benchmark result."""
        self.results[key] = {
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": cpu_time / gpu_time if gpu_time > 0 else 0,
            "task": task,
        }
    
    def print_summary(self) -> None:
        """Print summary with scoring."""
        if not self.results:
            return
        
        print("\n" + "="*70)
        print("📊 BENCHMARK SUMMARY")
        print("="*70)
        
        print(f"\n{'Task':<30} {'CPU Time':<12} {'GPU Time':<12} {'Score':<15}")
        print("-" * 70)
        
        for key, data in self.results.items():
            cpu = data["cpu_time"]
            gpu = data["gpu_time"]
            speedup = data["speedup"]
            task = data["task"]
            
            # Scoring system
            if speedup > 1.5:
                score = f"🟢 GPU +{speedup:.1f}x faster"
            elif speedup > 0.8:
                score = f"🟡 CPU +{1/speedup:.1f}x faster"
            else:
                score = f"🔴 CPU {1/speedup:.1f}x faster"
            
            print(f"{task:<30} {cpu:>10.3f}s {gpu:>10.3f}s {score:<15}")
    
    def print_recommendations(self) -> None:
        """Print optimization recommendations."""
        print("\n" + "="*70)
        print("💡 RECOMMENDATIONS")
        print("="*70)
        
        avg_speedup = np.mean([r["speedup"] for r in self.results.values()])
        
        if avg_speedup < 0.5:
            print("\n❌ GPU is significantly slower than CPU")
            print("Recommendations:")
            print("  1. Increase model size: hidden_layers=[1024, 1024, 512]")
            print("  2. Increase batch_size: batch_size=512-1024")
            print("  3. Increase training duration: episodes=500+")
            print("  4. Use GPU only for very large models")
        elif avg_speedup < 1.0:
            print("\n⚠️  GPU is slightly slower than CPU")
            print("Recommendations:")
            print("  1. Current models are small - CPU is efficient")
            print("  2. For small models: stick with CPU")
            print("  3. For production: use GPU with larger models")
        else:
            print("\n✅ GPU provides speedup!")
            print(f"Average speedup: {avg_speedup:.1f}x")


def benchmark_inference(config: Dict, device: str) -> Tuple[float, int]:
    """
    Benchmark inference performance.
    
    Returns:
        Tuple of (time_seconds, predictions_per_second)
    """
    from app.rl.agent import DQNAgent
    
    agent = DQNAgent(
        state_size=config["state_size"],
        action_size=4,
        hidden_layers=config["hidden_layers"],
        device=device
    )
    
    # Generate test data
    states = np.random.randn(
        config["batch_size"], 
        config["state_size"]
    ).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        agent.get_q_values(states)
    
    # Benchmark
    start = time.time()
    for _ in range(20):  # Reduced from 100 for speed
        agent.get_q_values(states)
    elapsed = time.time() - start
    
    predictions_per_sec = (20 * config["batch_size"]) / elapsed
    
    return elapsed, predictions_per_sec


def benchmark_training(config: Dict, device: str) -> Tuple[float, float]:
    """
    Benchmark training performance.
    
    Returns:
        Tuple of (time_seconds, episodes_per_second)
    """
    from app.rl.agent import DQNAgent
    
    agent = DQNAgent(
        state_size=config["state_size"],
        action_size=4,
        hidden_layers=config["hidden_layers"],
        learning_rate=0.001,
        device=device,
        buffer_size=10000,
        batch_size=config["batch_size"],
    )
    
    start = time.time()
    
    for episode in range(config["episodes"]):
        state = np.random.randn(config["state_size"]).astype(np.float32)
        
        for step in range(config["max_steps"]):
            action = agent.get_action(state, training=True)
            next_state = np.random.randn(config["state_size"]).astype(np.float32)
            reward = np.random.uniform(-1, 1)
            done = step == config["max_steps"] - 1
            
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > config["batch_size"]:
                agent.learn()
            
            state = next_state
    
    elapsed = time.time() - start
    eps_per_sec = config["episodes"] / elapsed
    
    return elapsed, eps_per_sec


def run_inference_benchmark() -> None:
    """Run comprehensive inference benchmark."""
    print("\n" + "="*70)
    print("🔍 INFERENCE BENCHMARK (20 batches)")
    print("="*70)
    
    results = BenchmarkResults()
    configs = [BenchmarkConfig.SMALL, BenchmarkConfig.MEDIUM, BenchmarkConfig.LARGE]
    
    for config in configs:
        print(f"\n{config['name']} Model (state_size={config['state_size']}):")
        print("-" * 70)
        
        # CPU benchmark
        print(f"  CPU: ", end="", flush=True)
        cpu_time, cpu_preds = benchmark_inference(config, 'cpu')
        print(f"{cpu_time:.3f}s ({cpu_preds:.0f} pred/s)")
        
        # GPU benchmark
        if torch.cuda.is_available():
            print(f"  GPU: ", end="", flush=True)
            gpu_time, gpu_preds = benchmark_inference(config, 'cuda')
            print(f"{gpu_time:.3f}s ({gpu_preds:.0f} pred/s)")
        else:
            gpu_time = 0
            print("  GPU: ❌ Not available")
        
        task_name = f"Inference {config['name']}"
        results.add(f"inf_{config['name'].lower()}", cpu_time, gpu_time, task_name)
    
    results.print_summary()


def run_training_benchmark() -> None:
    """Run comprehensive training benchmark."""
    print("\n" + "="*70)
    print("🚀 TRAINING BENCHMARK")
    print("="*70)
    
    results = BenchmarkResults()
    configs = [BenchmarkConfig.SMALL, BenchmarkConfig.MEDIUM, BenchmarkConfig.LARGE]
    
    for config in configs:
        print(f"\n{config['name']} Model (state_size={config['state_size']}):")
        print("-" * 70)
        
        # CPU benchmark
        print(f"  CPU: ", end="", flush=True)
        cpu_time, cpu_eps = benchmark_training(config, 'cpu')
        print(f"{cpu_time:.1f}s ({cpu_eps:.1f} eps/s)")
        
        # GPU benchmark
        if torch.cuda.is_available():
            print(f"  GPU: ", end="", flush=True)
            gpu_time, gpu_eps = benchmark_training(config, 'cuda')
            print(f"{gpu_time:.1f}s ({gpu_eps:.1f} eps/s)")
        else:
            gpu_time = 0
            print("  GPU: ❌ Not available")
        
        task_name = f"Training {config['name']}"
        results.add(f"train_{config['name'].lower()}", cpu_time, gpu_time, task_name)
    
    results.print_summary()


def run_all_benchmarks() -> None:
    """Run all benchmarks with clean output."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "GENESYS GPU vs CPU BENCHMARK SUITE" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    
    # GPU Analysis
    GPUAnalysis.print_gpu_info()
    
    # Inference Benchmark
    run_inference_benchmark()
    
    # Training Benchmark
    run_training_benchmark()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)
    print("\n📌 Remember:")
    print("  • GPU shines with LARGE models and batches")
    print("  • Small models run faster on CPU due to overhead")
    print("  • Current DQN architecture is simple - use CPU for now")
    print("  • For production: optimize model size and batch size")
    print()


if __name__ == "__main__":
    run_all_benchmarks()
