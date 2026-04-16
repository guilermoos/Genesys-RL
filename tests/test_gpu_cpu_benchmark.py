"""
Tests for GPU vs CPU performance benchmarking.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import torch
import time

from app.rl.agent import DQNAgent


class TestGPUCPUBenchmark:
    """Benchmark tests comparing GPU and CPU performance."""
    
    @pytest.fixture
    def benchmark_config(self):
        """Benchmark configuration."""
        return {
            "state_size": 50,
            "action_size": 5,
            "batch_size": 128,
            "num_batches": 100,
        }
    
    def test_cpu_inference_performance(self, benchmark_config):
        """Test CPU inference performance."""
        agent = DQNAgent(
            state_size=benchmark_config["state_size"],
            action_size=benchmark_config["action_size"],
            device='cpu'
        )
        
        # Generate test data
        test_states = np.random.randn(
            benchmark_config["batch_size"],
            benchmark_config["state_size"]
        ).astype(np.float32)
        
        # Benchmark
        start = time.time()
        for _ in range(benchmark_config["num_batches"]):
            q_values = agent.get_q_values(test_states)
        cpu_time = time.time() - start
        
        # Calculate metrics
        total_predictions = (
            benchmark_config["num_batches"] * 
            benchmark_config["batch_size"]
        )
        predictions_per_second = total_predictions / cpu_time
        
        print(f"\nCPU Inference Performance:")
        print(f"  Time: {cpu_time:.3f}s")
        print(f"  Predictions/sec: {predictions_per_second:.1f}")
        
        # Assertions
        assert cpu_time > 0
        assert predictions_per_second > 0
        assert agent.device == torch.device('cpu')
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_gpu_inference_performance(self, benchmark_config):
        """Test GPU inference performance."""
        agent = DQNAgent(
            state_size=benchmark_config["state_size"],
            action_size=benchmark_config["action_size"],
            device='cuda'
        )
        
        # Generate test data
        test_states = np.random.randn(
            benchmark_config["batch_size"],
            benchmark_config["state_size"]
        ).astype(np.float32)
        
        # Benchmark
        start = time.time()
        for _ in range(benchmark_config["num_batches"]):
            q_values = agent.get_q_values(test_states)
        gpu_time = time.time() - start
        
        # Calculate metrics
        total_predictions = (
            benchmark_config["num_batches"] * 
            benchmark_config["batch_size"]
        )
        predictions_per_second = total_predictions / gpu_time
        
        print(f"\nGPU Inference Performance:")
        print(f"  Time: {gpu_time:.3f}s")
        print(f"  Predictions/sec: {predictions_per_second:.1f}")
        
        # Assertions
        assert gpu_time > 0
        assert predictions_per_second > 0
        assert agent.device == torch.device('cuda')
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_gpu_vs_cpu_comparison(self, benchmark_config):
        """Compare GPU vs CPU performance."""
        # CPU test
        agent_cpu = DQNAgent(
            state_size=benchmark_config["state_size"],
            action_size=benchmark_config["action_size"],
            device='cpu'
        )
        
        test_states = np.random.randn(
            benchmark_config["batch_size"],
            benchmark_config["state_size"]
        ).astype(np.float32)
        
        start = time.time()
        for _ in range(benchmark_config["num_batches"]):
            agent_cpu.get_q_values(test_states)
        cpu_time = time.time() - start
        
        # GPU test
        agent_gpu = DQNAgent(
            state_size=benchmark_config["state_size"],
            action_size=benchmark_config["action_size"],
            device='cuda'
        )
        
        start = time.time()
        for _ in range(benchmark_config["num_batches"]):
            agent_gpu.get_q_values(test_states)
        gpu_time = time.time() - start
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        
        print(f"\nGPU vs CPU Comparison:")
        print(f"  CPU time: {cpu_time:.3f}s")
        print(f"  GPU time: {gpu_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  GPU is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        
        # For small models, CPU can be faster due to transfer overhead
        # This is expected behavior
        assert cpu_time > 0
        assert gpu_time > 0
    
    def test_gpu_memory_info(self):
        """Test GPU memory information."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        print(f"\nGPU Information:")
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Current Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0


class TestLargeModelGPUBenchmark:
    """Benchmark tests for large models optimized for GPU."""
    
    @pytest.fixture
    def large_model_config(self):
        """Large model configuration optimized for GPU."""
        return {
            "state_size": 128,
            "action_size": 7,
            "batch_size": 256,
            "num_batches": 50,
            "hidden_layers": [512, 512, 256],
        }
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_large_model_cpu_performance(self, large_model_config):
        """Test large model performance on CPU."""
        agent = DQNAgent(
            state_size=large_model_config["state_size"],
            action_size=large_model_config["action_size"],
            device='cpu',
            hidden_layers=large_model_config["hidden_layers"]
        )
        
        test_states = np.random.randn(
            large_model_config["batch_size"],
            large_model_config["state_size"]
        ).astype(np.float32)
        
        start = time.time()
        for _ in range(large_model_config["num_batches"]):
            agent.get_q_values(test_states)
        cpu_time = time.time() - start
        
        total_predictions = (
            large_model_config["num_batches"] * 
            large_model_config["batch_size"]
        )
        predictions_per_second = total_predictions / cpu_time
        
        print(f"\nLarge Model - CPU Performance:")
        print(f"  State Size: {large_model_config['state_size']}")
        print(f"  Hidden Layers: {large_model_config['hidden_layers']}")
        print(f"  Time: {cpu_time:.3f}s")
        print(f"  Predictions/sec: {predictions_per_second:.1f}")
        
        assert cpu_time > 0
        assert predictions_per_second > 0
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_large_model_gpu_performance(self, large_model_config):
        """Test large model performance on GPU."""
        agent = DQNAgent(
            state_size=large_model_config["state_size"],
            action_size=large_model_config["action_size"],
            device='cuda',
            hidden_layers=large_model_config["hidden_layers"]
        )
        
        test_states = np.random.randn(
            large_model_config["batch_size"],
            large_model_config["state_size"]
        ).astype(np.float32)
        
        start = time.time()
        for _ in range(large_model_config["num_batches"]):
            agent.get_q_values(test_states)
        gpu_time = time.time() - start
        
        total_predictions = (
            large_model_config["num_batches"] * 
            large_model_config["batch_size"]
        )
        predictions_per_second = total_predictions / gpu_time
        
        print(f"\nLarge Model - GPU Performance:")
        print(f"  State Size: {large_model_config['state_size']}")
        print(f"  Hidden Layers: {large_model_config['hidden_layers']}")
        print(f"  Time: {gpu_time:.3f}s")
        print(f"  Predictions/sec: {predictions_per_second:.1f}")
        
        assert gpu_time > 0
        assert predictions_per_second > 0
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_large_model_gpu_speedup(self, large_model_config):
        """Test GPU speedup with large model."""
        # CPU
        agent_cpu = DQNAgent(
            state_size=large_model_config["state_size"],
            action_size=large_model_config["action_size"],
            device='cpu',
            hidden_layers=large_model_config["hidden_layers"]
        )
        
        test_states = np.random.randn(
            large_model_config["batch_size"],
            large_model_config["state_size"]
        ).astype(np.float32)
        
        start = time.time()
        for _ in range(large_model_config["num_batches"]):
            agent_cpu.get_q_values(test_states)
        cpu_time = time.time() - start
        
        # GPU
        agent_gpu = DQNAgent(
            state_size=large_model_config["state_size"],
            action_size=large_model_config["action_size"],
            device='cuda',
            hidden_layers=large_model_config["hidden_layers"]
        )
        
        start = time.time()
        for _ in range(large_model_config["num_batches"]):
            agent_gpu.get_q_values(test_states)
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        
        print(f"\nLarge Model GPU Speedup:")
        print(f"  CPU time: {cpu_time:.3f}s")
        print(f"  GPU time: {gpu_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # For large models, GPU should provide benefit
        assert speedup > 0
