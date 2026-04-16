#!/bin/bash

# Genesys API Examples using curl
# ================================

BASE_URL="http://localhost:8000/v1"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Genesys API Examples ===${NC}"
echo ""

# 1. Health Check
echo -e "${YELLOW}1. Health Check${NC}"
curl -s "$BASE_URL/health" | jq .
echo ""

# 2. Register User (or skip if already exists)
echo -e "${YELLOW}2. Register User${NC}"
REGISTER_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Demo User",
    "email": "demo@example.com",
    "password": "demopassword123"
  }')

echo "$REGISTER_RESPONSE" | jq .
echo ""

# 3. Login and Extract Token
echo -e "${YELLOW}3. Login and Extract Token${NC}"
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@example.com",
    "password": "demopassword123"
  }')

echo "$LOGIN_RESPONSE" | jq .
API_KEY=$(echo "$LOGIN_RESPONSE" | jq -r '.access_token // empty')

if [ -z "$API_KEY" ]; then
  echo -e "${YELLOW}Failed to get access token. Exiting.${NC}"
  exit 1
fi

echo -e "${GREEN}Token obtained: ${API_KEY:0:20}...${NC}"
echo ""

# 4. List Templates
echo -e "${YELLOW}4. List Templates${NC}"
curl -s "$BASE_URL/templates" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 5. Create Project
echo -e "${YELLOW}5. Create Project${NC}"
PROJECT_RESPONSE=$(curl -s -X POST "$BASE_URL/projects" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GridWorld Demo",
    "description": "Demo project for GridWorld navigation",
    "template_default": "grid_world"
  }')

echo "$PROJECT_RESPONSE" | jq .
PROJECT_ID=$(echo "$PROJECT_RESPONSE" | jq -r '.id // empty')

if [ -z "$PROJECT_ID" ]; then
  echo -e "${YELLOW}Failed to create project. Exiting.${NC}"
  exit 1
fi

echo -e "${GREEN}Project created with ID: $PROJECT_ID${NC}"
echo ""

# 6. List Projects
echo -e "${YELLOW}6. List Projects${NC}"
curl -s "$BASE_URL/projects" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 7. Start Training (GridWorld)
echo -e "${YELLOW}7. Start Training${NC}"
echo -e "${BLUE}Note: This job is queued. To execute, you need Celery running in another terminal:${NC}"
echo -e "${BLUE}  celery -A app.workers.celery_app worker --loglevel=info${NC}"
TRAINING_RESPONSE=$(curl -s -X POST "$BASE_URL/jobs/projects/$PROJECT_ID/train" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "template": "grid_world",
    "config": {
      "state_size": 8,
      "action_space": [0, 1, 2, 3],
      "episodes": 50,
      "max_steps": 100,
      "gamma": 0.99,
      "learning_rate": 0.001,
      "epsilon_start": 1.0,
      "epsilon_end": 0.01,
      "epsilon_decay": 0.995,
      "batch_size": 64,
      "memory_size": 10000,
      "target_update_freq": 100
    }
  }')

echo "$TRAINING_RESPONSE" | jq .
JOB_ID=$(echo "$TRAINING_RESPONSE" | jq -r '.id // empty')

if [ -z "$JOB_ID" ]; then
  echo -e "${YELLOW}Failed to start training. Exiting.${NC}"
  exit 1
fi

echo -e "${GREEN}Training job started with ID: $JOB_ID${NC}"
echo ""

# 8. Check Training Status
echo -e "${YELLOW}8. Check Training Status${NC}"
curl -s "$BASE_URL/jobs/$JOB_ID" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 9. Get Training Logs
echo -e "${YELLOW}9. Get Training Logs${NC}"
curl -s "$BASE_URL/jobs/$JOB_ID/logs?lines=50" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 10. List Models
echo -e "${YELLOW}10. List Models${NC}"
curl -s "$BASE_URL/models/projects/$PROJECT_ID/models" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 11. Get Training Metrics
echo -e "${YELLOW}11. Get Training Metrics${NC}"
curl -s "$BASE_URL/jobs/$JOB_ID/metrics" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 12. Run Inference
echo -e "${YELLOW}12. Run Inference${NC}"
echo -e "${BLUE}Note: This will fail with 'No active model' if training hasn't completed.${NC}"
echo -e "${BLUE}Wait for the training job to finish before running inference.${NC}"
INFERENCE_RESPONSE=$(curl -s -X POST "$BASE_URL/inference/projects/$PROJECT_ID/predict" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": [0.0, 0.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
  }')

echo "$INFERENCE_RESPONSE" | jq .
echo ""

# 13. Batch Inference
echo -e "${YELLOW}13. Batch Inference${NC}"
echo -e "${BLUE}Note: This will also fail if no active model exists.${NC}"
curl -s -X POST "$BASE_URL/inference/projects/$PROJECT_ID/predict/batch" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "states": [
      [0.0, 0.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0],
      [0.2, 0.2, 0.8, 0.8, 0.5, 1.0, 1.0, 0.5],
      [0.5, 0.5, 0.8, 0.8, 0.0, 0.5, 1.0, 0.0]
    ]
  }' | jq .
echo ""

# 14. Get Project Stats
echo -e "${YELLOW}14. Get Project Stats${NC}"
curl -s "$BASE_URL/projects/$PROJECT_ID/stats" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 15. Get Inference Stats
echo -e "${YELLOW}15. Get Inference Stats${NC}"
curl -s "$BASE_URL/inference/projects/$PROJECT_ID/stats?hours=24" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# 16. Cancel Training (uncomment to use)
# echo -e "${YELLOW}16. Cancel Training${NC}"
# curl -s -X POST "$BASE_URL/jobs/$JOB_ID/cancel" \
#   -H "Authorization: Bearer $API_KEY" | jq .
# echo ""

echo -e "${GREEN}=== Examples Complete ===${NC}"
echo -e "${BLUE}Summary:${NC}"
echo "  Token: ${API_KEY:0:20}..."
echo "  Project ID: $PROJECT_ID"
echo "  Job ID: $JOB_ID"
