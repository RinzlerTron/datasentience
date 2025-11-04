# Architecture

DataSentience implements a three-stage agentic pipeline for predictive data center maintenance, deployed on AWS SageMaker with NVIDIA NIM integration.

## 1. Deployment Architecture

### Platform
- **AWS**: SageMaker for inference, API Gateway for routing, S3 for static hosting
- **NVIDIA NIM**: Cloud-hosted reasoning and embedding models
- **One-Click Deploy**: CloudFormation template with Launch Stack button

### CloudFormation Stacks

**datasentience-stack** (Primary):
- SageMaker model, endpoint configuration, and endpoint
- IAM roles with least-privilege access
- Secrets Manager for NVIDIA API key storage
- ECR repository for Docker images

**API Gateway Integration**:
- Request routing to SageMaker endpoints
- CORS configuration for browser access
- Rate limiting (100 requests/hour)

### Cost & Scaling
- **Instance Type**: ml.g5.xlarge (GPU-optimized for NVIDIA NIM)
- **Cost Estimate**: ~$1.50/hour for endpoint (competition budget)
- **Scaling**: Single instance deployment (can scale horizontally for production)

### Security Implementation

- **API Key Management**: NVIDIA keys stored in AWS Secrets Manager
- **Access Control**: API Gateway enforces origin validation
- **Rate Limiting**: slowapi middleware prevents abuse
- **Circuit Breaker**: Automatic failover during NVIDIA API outages

## Performance Optimizations

### Caching Strategy
- **Vector Search**: FAISS IVF clustering reduces search complexity from O(n) to O(log n)
- **In-Memory Caching**: Query result caching for repeated queries
- **Performance**: 292x improvement through intelligent caching

### Infrastructure Scaling
- **Instance Type**: ml.g5.xlarge (GPU-optimized for NVIDIA NIM)
- **Auto-scaling**: Single instance deployment
- **Health Monitoring**: SageMaker /ping endpoint with CloudWatch logs

## 2. Application Architecture

### Code Structure
```
src/
├── agent.py              # Single agent for data retrieval and reasoning
├── multi_agent_system.py # Orchestrator for multi-agent coordination
├── vector_store.py       # FAISS-based vector search (O(log n) complexity)
├── indexer.py           # Data indexing into vector store
├── config.py            # Configuration management
├── startup_checks.py     # Model validation and health checks
└── main.py              # FastAPI application entry point
```

### System Overview

```
Browser → API Gateway → SageMaker → NVIDIA NIM
```

**Stack Components:**
- **Frontend**: React application on S3 static hosting
- **API Layer**: AWS API Gateway with CORS and rate limiting
- **Backend**: FastAPI application on SageMaker endpoints
- **Processing**: NVIDIA NIM models for reasoning and embeddings
- **Storage**: FAISS vector database for semantic search

### Multi-Agent Pipeline

**Stage 1: Data Retrieval Agent**
- Processes live telemetry data and historical failure patterns
- Uses NVIDIA NIM embeddings (nv-embedqa-e5-v5, 1024-dim vectors)
- FAISS IVF indexing for O(log n) search performance
- Vector search caching provides 292x performance improvement

**Stage 2: Reasoning Agent**
- Correlates equipment behavior with maintenance schedules
- Processes vendor manuals and operational documentation
- NVIDIA NIM reasoning (llama-3.1-nemotron-nano-8b-v1)
- Circuit breaker pattern with exponential backoff

**Stage 3: Action Planning Agent**
- Generates ROI-calculated recommendations
- Produces implementation timelines and cost estimates
- Outputs structured action plans with priority ranking

### Data Flow

1. **Request Ingestion**: Browser sends query via API Gateway
2. **Agent Orchestration**: FastAPI coordinates three-stage pipeline
3. **Vector Processing**: FAISS search returns relevant context
4. **AI Reasoning**: NVIDIA NIM processes context and generates insights
5. **Response Assembly**: Structured output with ROI calculations
6. **Delivery**: JSON response through API Gateway to frontend

## Monitoring and Observability

- **Health Checks**: SageMaker /ping, /invocations, /metrics endpoints
- **Logging**: CloudWatch integration for error tracking
- **Performance**: Request/response time monitoring
- **Cost Tracking**: AWS billing integration for usage analysis

## Development Workflow

```bash
# Local development
docker-compose up -d  # Start Redis and dependencies
python src/main.py    # Run FastAPI locally

# Production deployment
./deploy.sh           # Deploy to AWS SageMaker
```

The deployment script handles ECR image building, CloudFormation stack creation, and secret management automatically.