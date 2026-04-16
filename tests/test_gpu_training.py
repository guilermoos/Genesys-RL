"""
Tests for GPU-optimized training.
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
from app.templates.decision_optimization import DecisionOptimizationTemplate


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
class TestGPUOptimizedTraining:
    """Tests for GPU-optimized training configurations."""
    
    @pytest.fixture
    def small_model_config(self):
        """Small model configuration (baseline)."""
        return {
            "state_size": 50,
            "action_space": [0, 1, 2, 3, 4],
            "episodes": 50,
            "max_steps": 200,
            "batch_size": 128,
            "memory_size": 10000,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
        }
    
    @pytest.fixture
    def large_model_config(self):
        """Large model configuration (GPU-optimized)."""
        return {
            "state_size": 128,
            "action_space": [0, 1, 2, 3, 4, 5, 6],
            "episodes": 150,
            "max_steps": 250,
            "batch_size": 256,
            "memory_size": 100000,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
        }
    
    def test_small_model_training_speed(self, small_model_config):
        """Test small model training speed on GPU."""
        agent = DQNAgent(
            state_size=small_model_config["state_size"],
            action_size=len(small_model_config["action_space"]),
            device='cuda'
        )
        
        print(f"\nSmall Model Training (GPU):")
        print(f"  State size: {small_model_config['state_size']}")
        print(f"  Episodes: {small_model_config['episodes']}")
        
        start = time.time()
        total_reward = 0
        episode_rewards = []
        
        for episode in range(small_model_config["episodes"]):
            state = np.random.randn(small_model_config["state_size"]).astype(np.float32)
            episode_reward = 0
            
            for step in range(small_model_config["max_steps"]):
                # Get action from agent
                action = agent.get_action(state, training=True)
                
                # Simulate environment step
                next_state = np.random.randn(small_model_config["state_size"]).astype(np.float32)
                reward = np.random.uniform(-1, 1)
                done = np.random.random() < 0.1
                
                episode_reward += reward
                
                # Store experience
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                # Learn from replay buffer
                if len(agent.replay_buffer) > small_model_config["batch_size"]:
                    agent.learn()
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            total_reward += episode_reward
        
        training_time = time.time() - start
        avg_reward = total_reward / small_model_config["episodes"]
        eps_per_second = small_model_config["episodes"] / training_time
        
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Episodes/sec: {eps_per_second:.2f}")
        print(f"  Avg reward: {avg_reward:.2f}")
        
        assert training_time > 0
        assert eps_per_second > 0
    
    def test_large_model_training_speed(self, large_model_config):
        """Test large model training speed on GPU."""
        agent = DQNAgent(
            state_size=large_model_config["state_size"],
            action_size=len(large_model_config["action_space"]),
            device='cuda',
            hidden_layers=[512, 512, 256]
        )
        
        print(f"\nLarge Model Training (GPU):")
        print(f"  State size: {large_model_config['state_size']}")
        print(f"  Hidden layers: [512, 512, 256]")
        print(f"  Episodes: {large_model_config['episodes']}")
        
        start = time.time()
        total_reward = 0
        episode_rewards = []
        
        for episode in range(large_model_config["episodes"]):
            state = np.random.randn(large_model_config["state_size"]).astype(np.float32)
            episode_reward = 0
            
            for step in range(large_model_config["max_steps"]):
                # Get action from agent
                action = agent.get_action(state, training=True)
                
                # Simulate environment step
                next_state = np.random.randn(large_model_config["state_size"]).astype(np.float32)
                reward = np.random.uniform(-1, 1)
                done = np.random.random() < 0.1
                
                episode_reward += reward
                
                # Store experience
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                # Learn from replay buffer
                if len(agent.replay_buffer) > large_model_config["batch_size"]:
                    agent.learn()
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            total_reward += episode_reward
        
        training_time = time.time() - start
        avg_reward = total_reward / large_model_config["episodes"]
        eps_per_second = large_model_config["episodes"] / training_time
        
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Episodes/sec: {eps_per_second:.2f}")
        print(f"  Avg reward: {avg_reward:.2f}")
        
        assert training_time > 0
        assert eps_per_second > 0
    
    def test_batch_size_impact_on_gpu(self):
        """Test impact of batch size on GPU training speed."""
        state_size = 100
        action_size = 5
        batch_sizes = [32, 128, 256, 512]
        
        print(f"\nBatch Size Impact on GPU:")
        print(f"  State size: {state_size}")
        
        for batch_size in batch_sizes:
            agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                device='cuda'
            )
            
            # Generate training data
            states = np.random.randn(batch_size, state_size).astype(np.float32)
            
            start = time.time()
            for _ in range(100):
                agent.get_q_values(states)
            elapsed = time.time() - start
            
            throughput = (100 * batch_size) / elapsed
            
            print(f"  Batch {batch_size:3d}: {elapsed:.3f}s ({throughput:.0f} pred/s)")
            
            assert elapsed > 0
    
    def test_memory_size_impact(self):
        """Test impact of replay buffer size on memory usage."""
        state_size = 100
        action_size = 5
        memory_sizes = [10000, 50000, 100000]
        
        print(f"\nMemory Size Impact:")
        print(f"  State size: {state_size}")
        
        for memory_size in memory_sizes:
            agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                device='cuda',
                buffer_size=memory_size
            )
            
            # Fill replay buffer
            for i in range(min(memory_size // 10, 1000)):
                state = np.random.randn(state_size).astype(np.float32)
                action = np.random.randint(0, action_size)
                reward = np.random.uniform(-1, 1)
                next_state = np.random.randn(state_size).astype(np.float32)
                done = i % 100 == 0
                
                agent.replay_buffer.add(state, action, reward, next_state, done)
            
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            
            print(f"  Memory {memory_size:6d}: GPU allocated {memory_allocated:.2f} GB")
    
    def test_device_detection(self):
        """Test CUDA device detection and initialization."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        agent = DQNAgent(
            state_size=50,
            action_size=5,
            device='cuda'
        )
        
        print(f"\nDevice Detection:")
        print(f"  Device: {agent.device}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        print(f"  Device Count: {torch.cuda.device_count()}")
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        
        assert agent.device == torch.device('cuda')
        assert torch.cuda.is_available()
    
    def test_gpu_memory_cleanup(self):
        """Test GPU memory cleanup after training."""
        print(f"\nGPU Memory Cleanup Test:")
        
        initial_memory = torch.cuda.memory_allocated(0)
        print(f"  Initial memory: {initial_memory / 1e6:.2f} MB")
        
        # Create and train multiple agents
        agents = []
        for i in range(3):
            agent = DQNAgent(
                state_size=200,
                action_size=10,
                device='cuda'
            )
            agents.append(agent)
        
        during_memory = torch.cuda.memory_allocated(0)
        print(f"  During creation: {during_memory / 1e6:.2f} MB")
        
        # Clear agents
        agents.clear()
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated(0)
        print(f"  After cleanup: {final_memory / 1e6:.2f} MB")
        
        assert final_memory <= during_memory
