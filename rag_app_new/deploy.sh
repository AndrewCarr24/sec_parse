#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
REGION="${AWS_REGION:-us-east-1}"
APP_NAME="ragAgent"
IMAGE_TAG="${1:-latest}"

# ─── Step 1: Bootstrap CDK (first time only) ────────────────────────────────
echo "==> Bootstrapping CDK (if needed)..."
cd "$(dirname "$0")/infra"
npm install
npx cdk bootstrap "aws://$(aws sts get-caller-identity --query Account --output text)/${REGION}" 2>/dev/null || true

# ─── Step 2: Deploy ECR stack first ─────────────────────────────────────────
echo "==> Deploying ECR stack..."
npx cdk deploy "${APP_NAME}-EcrStack" --require-approval never

# ─── Step 3: Get ECR repository URI ─────────────────────────────────────────
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${APP_NAME,,}-agent"
IMAGE_URI="${ECR_REPO}:${IMAGE_TAG}"

echo "==> ECR repository: ${ECR_REPO}"
echo "==> Image URI: ${IMAGE_URI}"

# ─── Step 4: Build and push Docker image ────────────────────────────────────
echo "==> Logging into ECR..."
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Building Docker image..."
cd "$(dirname "$0")/.."  # back to project root
docker build -f rag_app_new/Dockerfile -t "${IMAGE_URI}" .

echo "==> Pushing Docker image..."
docker push "${IMAGE_URI}"

# ─── Step 5: Deploy AgentCore stack ─────────────────────────────────────────
echo "==> Deploying AgentCore stack..."
cd rag_app_new/infra
npx cdk deploy "${APP_NAME}-AgentCoreStack" \
  -c "imageUri=${IMAGE_URI}" \
  --require-approval never

echo ""
echo "==> Deployment complete!"
echo "    Check the stack outputs above for MemoryId, RuntimeId, etc."
