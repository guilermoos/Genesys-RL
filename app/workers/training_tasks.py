"""
Celery tasks for training jobs.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from app.workers.celery_app import celery_app
from app.db.session import SessionLocal
from app.services.training_service import TrainingService
from app.services.model_service import ModelService
from app.templates.base import TemplateRegistry
from app.rl.agent import DQNAgent
from app.rl.trainer import Trainer
from app.utils.config import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def run_training_job(self, job_id: str) -> Dict[str, Any]:
    """
    Execute a training job asynchronously.
    
    Args:
        job_id: Training job ID
        
    Returns:
        Training results
    """
    db = SessionLocal()
    
    try:
        # Get job
        job = db.query(__import__("app.models.training_job", fromlist=["TrainingJob"]).TrainingJob).filter(
            __import__("app.models.training_job", fromlist=["TrainingJob"]).TrainingJob.id == job_id
        ).first()
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        logger.info(f"Starting training job {job_id}")
        
        # Update status to running
        TrainingService.update_job_status(db, job_id, "running")
        
        # Create environment from template
        template_class = TemplateRegistry.get(job.template)
        env_config = job.config.get("env_config") or {}
        
        # Merge with default config
        default_config = template_class({}).get_default_config()
        if default_config and env_config:
            default_config.update(env_config)
        
        environment = template_class(default_config)
        
        # Extract training config
        config = job.config
        
        state_size = environment.get_state_size()
        action_size = environment.get_action_size()
        
        # Create DQN agent
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_layers=config.get("hidden_layers", [128, 128]),
            learning_rate=config.get("learning_rate", 0.001),
            gamma=config.get("gamma", 0.99),
            epsilon_start=config.get("epsilon_start", 1.0),
            epsilon_end=config.get("epsilon_end", 0.01),
            epsilon_decay=config.get("epsilon_decay", 0.995),
            buffer_size=config.get("memory_size", 10000),
            batch_size=config.get("batch_size", 64),
            target_update_freq=config.get("target_update_freq", 100),
        )
        
        # Setup trainer
        settings = get_settings()
        save_dir = os.path.join(settings.MODELS_PATH, job.project_id, job_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # Progress callback to update task state
        def progress_callback(progress_data: Dict[str, Any]):
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": progress_data["episode"],
                    "total": progress_data["total_episodes"],
                    "reward": progress_data["reward"],
                    "avg_reward_100": progress_data["avg_reward_100"],
                    "epsilon": progress_data["epsilon"],
                }
            )
        
        trainer = Trainer(
            agent=agent,
            environment=environment,
            save_dir=save_dir,
            checkpoint_freq=config.get("checkpoint_freq", 100),
            log_freq=config.get("log_freq", 10),
            progress_callback=progress_callback,
        )
        
        # Train
        episodes = config.get("episodes", 1000)
        max_steps = config.get("max_steps", 500)
        
        metrics = trainer.train(
            num_episodes=episodes,
            max_steps_per_episode=max_steps,
        )
        
        # Save final model
        model_path = trainer.save_final_model("model.pt")
        
        # Create model version
        model = ModelService.create_model_version(
            db=db,
            project_id=job.project_id,
            job_id=job_id,
            name=f"Model v{db.query(__import__('app.models.model_version', fromlist=['ModelVersion']).ModelVersion).filter(__import__('app.models.model_version', fromlist=['ModelVersion']).ModelVersion.project_id == job.project_id).count() + 1}",
            artifact_path=model_path,
            config={
                "state_size": state_size,
                "action_size": action_size,
                "template": job.template,
                "hyperparameters": agent.get_config(),
            },
            metrics=trainer.get_training_summary(),
        )
        
        # Update job as completed
        summary = trainer.get_training_summary()
        TrainingService.update_job_status(
            db,
            job_id,
            "completed",
            metrics=summary,
        )
        
        logger.info(f"Training job {job_id} completed successfully")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "model_id": model.id,
            "metrics": summary,
        }
    
    except SoftTimeLimitExceeded:
        logger.error(f"Training job {job_id} exceeded time limit")
        TrainingService.update_job_status(
            db,
            job_id,
            "failed",
            error_message="Training exceeded time limit",
        )
        raise
    
    except Exception as e:
        logger.exception(f"Training job {job_id} failed: {e}")
        
        # Update job as failed
        TrainingService.update_job_status(
            db,
            job_id,
            "failed",
            error_message=str(e),
        )
        
        # Retry on certain errors
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job {job_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60)
        
        raise
    
    finally:
        db.close()
