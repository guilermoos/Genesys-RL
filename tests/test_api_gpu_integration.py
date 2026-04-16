"""
Integration tests for GPU-optimized API workflows.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import requests
import json
import time
from typing import Dict, Optional

# Configuration
BASE_URL = "http://localhost:8000/v1"
TEST_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkNTRkOGRmMy05OWU3LTQzMzYtODRjZS05MTRjZjg3Y2QzNjkiLCJlbWFpbCI6ImRlbW9AZXhhbXBsZS5jb20iLCJleHAiOjE3NzYzOTU3MjB9.5DKtYXOuNmd9sLteH9Da65GaxJR84bpJCSet2jd1hx8"

HEADERS = {
    "Authorization": f"Bearer {TEST_TOKEN}",
    "Content-Type": "application/json"
}


@pytest.mark.integration
class TestGPUTrainingWorkflow:
    """Integration tests for GPU-optimized training workflows."""
    
    @staticmethod
    def create_project(name: str, description: str = "") -> str:
        """Create a new project and return its ID."""
        data = {
            "name": name,
            "description": description or name,
            "template_default": "decision_optimization"
        }
        
        response = requests.post(
            f"{BASE_URL}/projects",
            headers=HEADERS,
            json=data
        )
        
        assert response.status_code == 200
        return response.json()['id']
    
    @staticmethod
    def submit_training_job(
        project_id: str,
        template: str,
        config: Dict
    ) -> str:
        """Submit a training job and return its ID."""
        data = {
            "template": template,
            "config": config
        }
        
        response = requests.post(
            f"{BASE_URL}/jobs/projects/{project_id}/train",
            headers=HEADERS,
            json=data
        )
        
        assert response.status_code == 200
        return response.json()['id']
    
    @staticmethod
    def wait_for_job(job_id: str, timeout: int = 300) -> Dict:
        """Wait for a job to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{BASE_URL}/jobs/{job_id}",
                headers=HEADERS
            )
            
            job = response.json()
            
            if job['status'] in ['completed', 'failed']:
                return job
            
            time.sleep(1)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
    
    def test_small_model_training_workflow(self):
        """Test complete workflow with small model."""
        print("\n" + "=" * 70)
        print("TEST: Small Model Training Workflow")
        print("=" * 70)
        
        # Create project
        project_id = self.create_project("GPU Test - Small Model")
        print(f"✓ Project created: {project_id}")
        
        # Small model config
        config = {
            "state_size": 50,
            "action_space": [0, 1, 2, 3, 4],
            "episodes": 50,
            "max_steps": 200,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "batch_size": 128,
            "memory_size": 10000,
            "target_update_freq": 10,
            "use_gpu": True
        }
        
        # Submit training
        job_id = self.submit_training_job(
            project_id,
            "decision_optimization",
            config
        )
        print(f"✓ Training job submitted: {job_id}")
        
        # Wait for completion
        job = self.wait_for_job(job_id)
        print(f"✓ Training completed: {job['status']}")
        print(f"  Episodes: {job.get('total_episodes', 'N/A')}")
        print(f"  Avg Reward: {job.get('avg_reward', 'N/A'):.2f}")
        
        assert job['status'] == 'completed'
        assert job.get('avg_reward') is not None
    
    def test_large_model_training_workflow(self):
        """Test complete workflow with large GPU-optimized model."""
        print("\n" + "=" * 70)
        print("TEST: Large Model Training Workflow (GPU-Optimized)")
        print("=" * 70)
        
        # Create project
        project_id = self.create_project("GPU Test - Large Model")
        print(f"✓ Project created: {project_id}")
        
        # Large model config (GPU-optimized)
        config = {
            "state_size": 128,
            "action_space": [0, 1, 2, 3, 4, 5, 6],
            "episodes": 150,
            "max_steps": 250,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "batch_size": 256,
            "memory_size": 100000,
            "target_update_freq": 100,
            "use_gpu": True
        }
        
        # Submit training
        start_time = time.time()
        job_id = self.submit_training_job(
            project_id,
            "decision_optimization",
            config
        )
        print(f"✓ Training job submitted: {job_id}")
        
        # Wait for completion
        job = self.wait_for_job(job_id, timeout=600)
        training_time = time.time() - start_time
        
        print(f"✓ Training completed: {job['status']}")
        print(f"  Episodes: {job.get('total_episodes', 'N/A')}")
        print(f"  Total time: {training_time:.1f}s")
        print(f"  Avg Reward: {job.get('avg_reward', 'N/A'):.2f}")
        print(f"  Episodes/sec: {job.get('total_episodes', 0) / (training_time - 2):.2f}")
        
        assert job['status'] == 'completed'
        assert job.get('avg_reward') is not None
    
    def test_training_and_inference_workflow(self):
        """Test training followed by inference."""
        print("\n" + "=" * 70)
        print("TEST: Training + Inference Workflow")
        print("=" * 70)
        
        # Create project
        project_id = self.create_project("GPU Test - Training + Inference")
        print(f"✓ Project created: {project_id}")
        
        # Submit training
        config = {
            "state_size": 50,
            "action_space": [0, 1, 2, 3, 4],
            "episodes": 30,
            "max_steps": 200,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "batch_size": 128,
            "memory_size": 10000,
            "target_update_freq": 10,
            "use_gpu": True
        }
        
        job_id = self.submit_training_job(
            project_id,
            "decision_optimization",
            config
        )
        print(f"✓ Training submitted: {job_id}")
        
        # Wait for training
        job = self.wait_for_job(job_id)
        print(f"✓ Training completed")
        
        # Activate model
        response = requests.post(
            f"{BASE_URL}/models/projects/{project_id}/activate/{job['model_version_id']}",
            headers=HEADERS
        )
        assert response.status_code == 200
        print(f"✓ Model activated")
        
        # Test inference
        test_state = [0.1] * config["state_size"]
        
        inference_data = {"state": test_state}
        response = requests.post(
            f"{BASE_URL}/inference/projects/{project_id}/predict",
            headers=HEADERS,
            json=inference_data
        )
        
        assert response.status_code == 200
        result = response.json()
        print(f"✓ Single inference completed")
        print(f"  Q-values: {result.get('q_values', [])[:3]}... (first 3)")
        
        # Test batch inference
        batch_states = [[0.1 * (i+1)] * config["state_size"] for i in range(5)]
        batch_data = {"states": batch_states}
        
        response = requests.post(
            f"{BASE_URL}/inference/projects/{project_id}/predict/batch",
            headers=HEADERS,
            json=batch_data
        )
        
        assert response.status_code == 200
        result = response.json()
        print(f"✓ Batch inference completed (5 states)")
        print(f"  Predictions: {len(result.get('predictions', []))}")
    
    def test_multiple_concurrent_projects(self):
        """Test multiple concurrent training projects."""
        print("\n" + "=" * 70)
        print("TEST: Multiple Concurrent Projects")
        print("=" * 70)
        
        num_projects = 3
        project_ids = []
        job_ids = []
        
        # Create and submit multiple projects
        for i in range(num_projects):
            project_id = self.create_project(f"GPU Test - Concurrent {i+1}")
            project_ids.append(project_id)
            
            config = {
                "state_size": 50,
                "action_space": [0, 1, 2, 3, 4],
                "episodes": 25,
                "max_steps": 200,
                "gamma": 0.99,
                "learning_rate": 0.001,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "batch_size": 128,
                "memory_size": 10000,
                "target_update_freq": 10,
                "use_gpu": True
            }
            
            job_id = self.submit_training_job(
                project_id,
                "decision_optimization",
                config
            )
            job_ids.append(job_id)
            print(f"✓ Project {i+1} training submitted: {job_id}")
        
        # Wait for all to complete
        for i, job_id in enumerate(job_ids):
            job = self.wait_for_job(job_id)
            print(f"✓ Project {i+1} completed: {job['status']}")
        
        print(f"\n✓ All {num_projects} projects completed successfully")


@pytest.mark.integration
class TestGPUAPIEndpoints:
    """Tests for GPU-related API endpoints."""
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{BASE_URL}/health", headers=HEADERS)
        assert response.status_code == 200
        
        health = response.json()
        print(f"\nHealth Check:")
        print(f"  Status: {health.get('status')}")
        print(f"  Timestamp: {health.get('timestamp')}")
    
    def test_projects_endpoint(self):
        """Test projects listing endpoint."""
        response = requests.get(f"{BASE_URL}/projects", headers=HEADERS)
        assert response.status_code == 200
        
        projects = response.json()
        print(f"\nProjects Endpoint:")
        print(f"  Total projects: {len(projects)}")
        if projects:
            print(f"  First project: {projects[0].get('name')}")
    
    def test_templates_endpoint(self):
        """Test templates listing endpoint."""
        response = requests.get(f"{BASE_URL}/templates", headers=HEADERS)
        assert response.status_code == 200
        
        templates = response.json()
        print(f"\nTemplates Endpoint:")
        print(f"  Available templates: {len(templates)}")
        for template in templates:
            print(f"    - {template.get('name')}")
