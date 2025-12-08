<a id="readme-top"></a>

<!-- PROJECT LOGO & TITLE -->

<div align="center">
  <a href="https://github.com/opencloudhub">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-light.svg">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-dark.svg">
    <!-- Fallback -->
    <img alt="OpenCloudHub Logo" src="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-dark.svg" style="max-width:700px; max-height:175px;">
  </picture>
  </a>

<h1 align="center">Wine Classifier - MLOps Demo</h1>

<p align="center">
    Scikit-learn wine classification with a modern MLOps pipeline featuring MLflow tracking and Ray for distributed training and serving.<br />
    <a href="https://github.com/opencloudhub"><strong>Explore OpenCloudHub »</strong></a>
  </p>
</div>

______________________________________________________________________

<details>
  <summary>📑 Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#thesis-context">Thesis Context</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#infrastructure">Infrastructure Options</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

______________________________________________________________________

<h2 id="about">🎯 About</h2>

This repository demonstrates an end-to-end MLOps pipeline for wine quality classification using scikit-learn and the UCI Wine Quality dataset. It showcases modern machine learning practices including:

- **Experiment Tracking** - Full lineage from data to deployed model via MLflow
- **Data Version Control** - Reproducible datasets using DVC with S3-compatible storage (MinIO)
- **Model Registry** - Centralized model versioning and lifecycle management
- **Containerized Workflows** - Multi-stage Docker builds for training and serving
- **Production Serving** - Ray Serve with FastAPI for scalable inference

______________________________________________________________________

<h2 id="thesis-context">📚 Thesis Context</h2>

This repository serves as a practical demonstration for my thesis on MLOps practices. It illustrates key concepts:

| Concept                    | Implementation                                                   | Tool                  |
| -------------------------- | ---------------------------------------------------------------- | --------------------- |
| **Experiment Tracking**    | All training runs logged with parameters, metrics, and artifacts | MLflow                |
| **Data Versioning**        | Dataset versions tracked and linked to model training            | DVC + MinIO           |
| **Model Registry**         | Models versioned with automatic lineage tracking                 | MLflow Model Registry |
| **Reproducibility**        | Environment variables (workflow tags) ensure traceability        | Pydantic Settings     |
| **CI/CD Integration**      | Automated builds and training pipelines                          | GitHub Actions + Argo |
| **Serving Infrastructure** | Scalable model serving with hot-reload capability                | Ray Serve + FastAPI   |

**MLflow Integration Highlights:**

- `mlflow.sklearn.autolog()` captures training metrics automatically
- Models registered with `registered_model_name` for versioning
- Training runs tagged with `argo_workflow_uid`, `docker_image_tag`, `dvc_data_version`
- Serving layer loads models via `mlflow.sklearn.load_model(model_uri)`
- Data metadata logged as artifacts for full provenance

______________________________________________________________________

<h2 id="features">✨ Features</h2>

- 🔬 **Experiment Tracking** - MLflow integration with autologging and model registry
- 📦 **Data Versioning** - DVC-managed datasets with S3/MinIO backend
- 🐳 **Multi-Stage Docker** - Optimized builds for dev, training, and serving
- ⚡ **Distributed Training** - Ray + joblib backend for parallel model fitting
- 🚀 **Production Serving** - Ray Serve with FastAPI, health checks, and batch inference
- 🔄 **CI/CD Pipelines** - GitHub Actions for code quality, image builds, and training
- 🏷️ **Workflow Tagging** - Full traceability via environment-based data contracts
- 🧪 **DevContainer Support** - VS Code development environment included

______________________________________________________________________

<h2 id="architecture">🏗️ Architecture</h2>

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CI/CD Pipeline                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐   │
│  │ GitHub       │──▶│ Docker Build │──▶│ Argo Workflows               │   │
│  │ Actions      │    │ (training/   │    │ (Orchestrates training jobs) │   │
│  └──────────────┘    │  serving)    │    └──────────────────────────────┘   │
│                      └──────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Training Pipeline                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ DVC          │──▶│ Ray + Joblib │──▶│ MLflow       │                   │
│  │ (Load data   │    │ (Distributed │    │ (Log metrics,│                   │
│  │  from MinIO) │    │  training)   │    │  register    │                   │
│  └──────────────┘    └──────────────┘    │  model)      │                   │
│                                          └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Serving Pipeline                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ MLflow       │──▶│ Ray Serve    │──▶│ FastAPI      │                   │
│  │ (Load model  │    │ (Deployment  │    │ (REST API,   │                   │
│  │  by URI)     │    │  management) │    │  /predict)   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Components:**

| Component         | Purpose                                       | Files                                             |
| ----------------- | --------------------------------------------- | ------------------------------------------------- |
| **Training**      | Model training with scikit-learn pipeline     | `src/training/train.py`, `data.py`, `config.py`   |
| **Serving**       | REST API for inference                        | `src/serving/serve.py`, `schemas.py`, `config.py` |
| **Data Loading**  | DVC-based data fetching from S3/MinIO         | `src/training/data.py`                            |
| **Configuration** | Pydantic-based settings management            | `src/*/config.py`                                 |
| **Logging**       | Rich-formatted logging with Ray compatibility | `src/_utils/logging.py`                           |

______________________________________________________________________

<h2 id="getting-started">🚀 Getting Started</h2>

### Prerequisites

- Docker
- VS Code with DevContainers extension (recommended)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/opencloudhub/ai-ml-sklearn.git
   cd ai-ml-sklearn
   ```

1. **Open in DevContainer** (Recommended)

   VSCode: `Ctrl+Shift+P` → `Dev Containers: Rebuild and Reopen in Container`

   Or **setup locally without DevContainer**:

   ```bash
   # Install UV
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync --dev
   ```

1. **Choose your infrastructure backend** (see next section)

______________________________________________________________________

<h2 id="infrastructure">🛠️ Infrastructure Options</h2>

Choose the infrastructure that fits your needs:

| Option                 | Use Case                               | Complexity      | Production-like |
| ---------------------- | -------------------------------------- | --------------- | --------------- |
| **Local Compose**      | Quick prototyping, simple tests        | ⭐ Easy         | ❌              |
| **Minikube**           | Integration testing, GitOps validation | ⭐⭐ Medium     | ✅              |
| **Production Cluster** | Real deployments, CI/CD pipelines      | ⭐⭐⭐ Advanced | ✅✅            |

### Option 1: Local Compose Stack (Quick & Dirty)

Lightweight Docker Compose stack for rapid iteration — no Kubernetes required.

```bash
# Clone and start the local stack
git clone https://github.com/OpenCloudHub/local-compose-stack.git
cd local-compose-stack
docker compose up -d
```

**Services available:**

| Service       | URL                   | Credentials          |
| ------------- | --------------------- | -------------------- |
| MLflow UI     | http://localhost:5000 | —                    |
| MinIO Console | http://localhost:9001 | `admin` / `admin123` |
| MinIO API     | http://localhost:9000 | `admin` / `admin123` |

**Configure this project:**

```bash
# Load environment for local compose
set -a && source .env.docker && set +a

# Start Ray head node
ray start --head
```

> 📖 See [OpenCloudHub/local-compose-stack](https://github.com/OpenCloudHub/local-compose-stack) for full documentation.

### Option 2: Minikube (Production-like Local)

Full Kubernetes stack with Helm charts — mirrors production infrastructure.

```bash
# Start minikube with sufficient resources
minikube start --cpus=4 --memory=8g

# Deploy the MLOps stack (MLflow, MinIO, Argo, etc.)
# See OpenCloudHub infrastructure repo for Helm charts
```

**Configure this project:**

```bash
# Load environment for minikube
set -a && source .env.minikube && set +a

# Start Ray head node
ray start --head
```

> ⚠️ **Note:** When running MinIO on minikube with DevContainer attached to host network, you may need to rebuild the container after cluster restarts.

### Option 3: Production Cluster

For CI/CD pipelines and production deployments:

1. Trigger training via GitHub Actions workflow dispatch
1. Argo Workflows orchestrates the training job
1. Models are registered in MLflow Model Registry
1. Serving images are built and deployed automatically

**Trigger production training:**

```
https://github.com/OpenCloudHub/ai-ml-sklearn/actions/workflows/train.yaml
```

______________________________________________________________________

<h2 id="usage">📖 Usage</h2>

### Training

**Basic training (after loading environment):**

```bash
python src/training/train.py --C 0.9
```

**Using Ray Job API (production-like):**

```bash
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python src/training/train.py
```

### Model Serving

**Start serving with hot-reload (development):**

```bash
serve run src.serving.serve:app_builder model_uri="models:/ci.wine-classifier/1" --reload
```

**Production deployment:**

```bash
serve build src.serving.serve:app_builder -o src/serving/serve_config.yaml
serve deploy src/serving/serve_config.yaml
```

**API Endpoints:**

| Endpoint        | Description                                |
| --------------- | ------------------------------------------ |
| `GET /`         | Service info and status                    |
| `GET /health`   | Health check                               |
| `GET /info`     | Model metadata (URI, run ID, data version) |
| `POST /predict` | Batch wine quality predictions             |

Access Swagger docs at `http://localhost:8000/docs`

______________________________________________________________________

<h2 id="configuration">⚙️ Configuration</h2>

### Environment Variables

The project uses Pydantic Settings for configuration. Required environment variables:

| Variable                | Description                       | Example                 |
| ----------------------- | --------------------------------- | ----------------------- |
| `MLFLOW_TRACKING_URI`   | MLflow server URL                 | `http://localhost:8081` |
| `ARGO_WORKFLOW_UID`     | Workflow run identifier           | `DEV` (local)           |
| `DOCKER_IMAGE_TAG`      | Docker image tag for traceability | `DEV` (local)           |
| `DVC_DATA_VERSION`      | Dataset version from DVC          | `wine-quality-v0.2.0`   |
| `AWS_ACCESS_KEY_ID`     | MinIO/S3 access key               | -                       |
| `AWS_SECRET_ACCESS_KEY` | MinIO/S3 secret key               | -                       |
| `AWS_ENDPOINT_URL`      | MinIO/S3 endpoint                 | `http://minio:9000`     |

### Training Configuration

See `src/training/config.py` for all training settings:

- `mlflow_experiment_name` - MLflow experiment name (default: `wine-quality`)
- `mlflow_registered_model_name` - Model registry name (default: `dev.wine-classifier`)
- `dvc_repo` - DVC registry repository URL
- `random_state` - Random seed for reproducibility (default: `42`)

### Serving Configuration

See `src/serving/config.py` for serving settings:

- `expected_num_features` - Number of input features (default: `12`)
- `request_max_length` - Max batch size for predictions (default: `1000`)

______________________________________________________________________

<h2 id="workflow-tags">🏷️ Workflow Tags & Data Contract</h2>

This project relies on a small CI/CD "data contract" provided via environment variables ("workflow tags") that must be present for automated and reproducible training runs and correct MLflow tagging. Follow these rules:

- Required workflow tags (set by Argo workflows in production):

  - ARGO_WORKFLOW_UID — unique identifier for the workflow run (use "DEV" for local dev)
  - DOCKER_IMAGE_TAG — image tag used for the training run (use "DEV" for local dev)
  - DVC_DATA_VERSION — DVC data version (e.g. wine-quality-v0.2.0). NOTE: this should take precedence over any CLI --data-version argument.

- Why these matter:

  - All training runs are automatically tagged in MLflow with the workflow tags (ARGO_WORKFLOW_UID, DOCKER_IMAGE_TAG, DVC_DATA_VERSION) so you can trace models back to the exact pipeline and dataset.
  - The training code reads DVC_DATA_VERSION from the environment and will prefer it over CLI args to ensure reproducible CI/CD runs.
  - If DVC_DATA_VERSION is not supplied in CI, the training run may not match the intended dataset version.

- Data provenance:

  - Training data and metadata are versioned in a DVC registry. The code loads data via DVC and reads artifacts from the MinIO S3-compatible bucket.
  - The training job logs the DVC metadata (metadata.json) into MLflow as an artifact so datasets are traceable.

- MinIO (S3) credentials required:

  - The runtime needs access to the MinIO bucket where DVC stores artifacts. Provide the following environment variables (examples in .env.sample):
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_ENDPOINT_URL
  - In Kubernetes/Argo, these are provided from secrets (see training workflow template). Locally, create a `.env` or export these variables.

- Local development notes:

  - For local development you can set workflow tags to "DEV" (except DVC_DATA_VERSION which you should set to the dataset tag you want to use, or leave blank and pass --data-version locally).
  - Example .env usage:
    ```
    cp .env.sample .env
    # edit .env to set DVC_DATA_VERSION and MinIO creds for local testing
    ```
  - When running training locally you can still pass --data-version; CI will override it when DVC_DATA_VERSION is set in the env.

- MLflow tagging policy:

  - Always include workflow tags on MLflow runs. The training entrypoint in src/training/train.py already applies these tags automatically:
    - argo_workflow_uid, docker_image_tag, dvc_data_version
  - This lets you filter experiments by workflow run, image tag and dataset version in the MLflow UI.

- Quick checklist before submitting a training run in CI:

  1. Ensure Docker image with code is published and DOCKER_IMAGE_TAG is set.
  1. Ensure ARGO_WORKFLOW_UID is set by the workflow.
  1. Ensure DVC_DATA_VERSION points to the correct dataset version.
  1. Ensure MinIO credentials (accesskey/secretkey and endpoint) are available to the job.
  1. If developing locally and running Minio on local kind cluster and attaching your devcontainer to the host network, you might need to rebuild this container after cluster restarts

<p align="right">(<a href="#readme-top">back to top</a>)</p>

______________________________________________________________________

<h2 id="project-structure">📁 Project Structure</h2>

```
ai-ml-sklearn/
├── src/
│   ├── training/                       # Training pipeline
│   │   ├── train.py                    # Main training script with MLflow logging
│   │   ├── data.py                     # DVC data loading from S3/MinIO
│   │   └── config.py                   # Training configuration (Pydantic)
│   ├── serving/                        # Model serving (Ray Serve + FastAPI)
│   │   ├── serve.py                    # Ray Serve deployment with FastAPI
│   │   ├── schemas.py                  # Pydantic request/response models
│   │   └── config.py                   # Serving configuration
│   └── _utils/                         # Shared utilities
│       └── logging.py                  # Rich logging configuration
├── tests/
│   └── test_wine_classifier.py         # API integration tests
├── notebooks/
│   └── exploring wine_dataset.ipynb    # Data exploration notebook
├── .github/workflows/                   # CI/CD workflows
│   ├── ci-code-quality.yaml            # Linting and code quality checks
│   ├── ci-docker-build-push.yaml       # Docker image builds (training/serving)
│   └── train.yaml                      # MLOps training pipeline dispatch
├── .env.docker                          # Environment for local-compose-stack
├── .env.minikube                        # Environment for minikube setup
├── Dockerfile                           # Multi-stage build (dev/training/serving)
├── pyproject.toml                       # Project dependencies and config
└── uv.lock                              # Dependency lock file
```

______________________________________________________________________

<h2 id="contributing">👥 Contributing</h2>

Contributions are welcome! This project follows OpenCloudHub's contribution standards.

Please see our [Contributing Guidelines](https://github.com/opencloudhub/.github/blob/main/.github/CONTRIBUTING.md) and [Code of Conduct](https://github.com/opencloudhub/.github/blob/main/.github/CODE_OF_CONDUCT.md) for more details.

______________________________________________________________________

<h2 id="license">📄 License</h2>

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for more information.

______________________________________________________________________

<h2 id="contact">📬 Contact</h2>

Organization Link: [https://github.com/OpenCloudHub](https://github.com/OpenCloudHub)

Project Link: [https://github.com/opencloudhub/ai-ml-sklearn](https://github.com/opencloudhub/ai-ml-sklearn)

______________________________________________________________________

<h2 id="acknowledgements">🙏 Acknowledgements</h2>

- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) - The dataset used for classification
- [MLflow](https://mlflow.org/) - ML lifecycle management and model registry
- [Ray](https://ray.io/) - Distributed computing and model serving
- [DVC](https://dvc.org/) - Data version control
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Pydantic](https://docs.pydantic.dev/) - Data validation and settings management
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal output

<p align="right">(<a href="#readme-top">back to top</a>)</p>
