#!/bin/bash
#
# DataSentience Deployment Script
# Deploys multi-agent AI system to AWS SageMaker with NVIDIA NIM integration
#
# Prerequisites:
# - AWS CLI configured with SageMaker permissions
# - Docker installed and running
# - NVIDIA API key from build.nvidia.com
#
# Usage: ./deploy.sh
#

set -e

# Configuration
AWS_REGION="us-east-1"
STACK_NAME="datasentience-stack"
ECR_REPO="datasentience"
DOCKER_TAG="sagemaker-final"

echo "DataSentience Deployment"
echo "======================="

# Utility function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validate prerequisites
echo "Checking prerequisites..."
if ! command_exists aws; then
    echo "Error: AWS CLI not found. Install from https://aws.amazon.com/cli/"
    exit 1
fi

if ! command_exists docker; then
    echo "Error: Docker not found. Install from https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker daemon not running. Start Docker and retry."
    exit 1
fi

echo "Prerequisites check passed"

# Get NVIDIA API key
if [ -z "$NVIDIA_API_KEY" ]; then
    echo ""
    echo "NVIDIA API Key Required"
    echo "Get your key from: https://build.nvidia.com/explore/discover"
    echo -n "Enter NVIDIA API Key: "
    read -s NVIDIA_API_KEY
    echo ""
fi

if [ -z "$NVIDIA_API_KEY" ]; then
    echo "Error: NVIDIA API key is required"
    exit 1
fi

# Get AWS account information
echo "Getting AWS account information..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
if [ $? -ne 0 ]; then
    echo "Error: Failed to get AWS account ID. Check your AWS credentials."
    exit 1
fi

echo "AWS Account ID: $AWS_ACCOUNT_ID"

# Create ECR repository if needed
echo "Setting up ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION >/dev/null 2>&1 || {
    echo "Creating ECR repository..."
    aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION
}

# Login to ECR
echo "Authenticating with ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push Docker image
echo "Building Docker image..."
FULL_IMAGE_NAME="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$DOCKER_TAG"

# Build with Docker v2 manifest format (required for SageMaker)
docker build --platform linux/amd64 -t $FULL_IMAGE_NAME .

echo "Pushing image to ECR..."
docker push $FULL_IMAGE_NAME

echo "Docker image deployed: $FULL_IMAGE_NAME"

# Deploy CloudFormation stack
echo "Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file deploy.yaml \
    --stack-name $STACK_NAME \
    --parameter-overrides NvidiaApiKey=$NVIDIA_API_KEY DockerTag=$DOCKER_TAG InstanceType=ml.g5.xlarge \
    --capabilities CAPABILITY_NAMED_IAM \
    --region $AWS_REGION

if [ $? -eq 0 ]; then
    echo "CloudFormation deployment successful"
else
    echo "Error: CloudFormation deployment failed"
    exit 1
fi

# Get stack outputs
echo "Retrieving deployment information..."
ENDPOINT_NAME=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $AWS_REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`EndpointName`].OutputValue' \
    --output text)

INVOCATIONS_URL=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $AWS_REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`InvocationsURL`].OutputValue' \
    --output text)

PING_URL=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $AWS_REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`PingURL`].OutputValue' \
    --output text)

# Wait for endpoint to be ready
echo "Waiting for SageMaker endpoint to be ready..."
echo "This typically takes 8-15 minutes..."

while true; do
    STATUS=$(aws sagemaker describe-endpoint \
        --endpoint-name $ENDPOINT_NAME \
        --region $AWS_REGION \
        --query 'EndpointStatus' \
        --output text 2>/dev/null || echo "NotFound")

    case $STATUS in
        "InService")
            echo "SageMaker endpoint is ready"
            break
            ;;
        "Creating"|"Updating")
            echo -n "."
            sleep 30
            ;;
        "Failed"|"OutOfService")
            echo "Error: Endpoint deployment failed"
            exit 1
            ;;
        *)
            echo -n "."
            sleep 30
            ;;
    esac
done

# Test deployment
echo "Testing deployment..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $PING_URL)
if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "Health check passed"
else
    echo "Warning: Health check returned HTTP $HEALTH_RESPONSE"
fi

# Create frontend deployment instructions
cat > frontend_deploy.md << EOF
# Frontend Deployment Instructions

## S3 Static Website Hosting

\`\`\`bash
# Build React application
cd ui
npm install
npm run build

# Create S3 bucket
BUCKET_NAME="datasentience-frontend-\$(date +%s)"
aws s3 mb s3://\$BUCKET_NAME --region us-east-1

# Configure static website hosting
aws s3 website s3://\$BUCKET_NAME --index-document index.html

# Upload application files
aws s3 sync build/ s3://\$BUCKET_NAME/ --acl public-read

# Update API configuration in React app to point to:
# $INVOCATIONS_URL
\`\`\`

## Deployment Endpoints

- **Health Check**: $PING_URL
- **API Endpoint**: $INVOCATIONS_URL
- **Metrics**: ${INVOCATIONS_URL/invocations/metrics}

## Testing

\`\`\`bash
curl -X POST $INVOCATIONS_URL \\
  -H 'Content-Type: application/json' \\
  -d '{"query": "Analyze current system performance", "scenario_type": "demo"}'
\`\`\`
EOF

# Deployment summary
echo ""
echo "DEPLOYMENT COMPLETE"
echo "=================="
echo "SageMaker Endpoint: $ENDPOINT_NAME"
echo "Health Check: $PING_URL"
echo "API Endpoint: $INVOCATIONS_URL"
echo "Metrics: ${INVOCATIONS_URL/invocations/metrics}"
echo ""
echo "Next Steps:"
echo "1. Deploy frontend using instructions in frontend_deploy.md"
echo "2. Configure React app API_URL to point to SageMaker endpoint"
echo "3. Test complete application workflow"
echo ""
echo "Deployment ready for production use"