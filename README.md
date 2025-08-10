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
    Scikit-learn wine classification with complete MLOps pipeline featuring MLflow tracking, hyperparameter optimization, and production-ready deployment patterns.<br />
    <a href="https://github.com/opencloudhub"><strong>Explore OpenCloudHub »</strong></a>
  </p>

<p align="center">
    <a href="https://github.com/opencloudhub/ai-ml-sklearn/graphs/contributors">
      <img src="https://img.shields.io/github/contributors/opencloudhub/ai-ml-sklearn.svg?style=for-the-badge" alt="Contributors">
    </a>
    <a href="https://github.com/opencloudhub/ai-ml-sklearn/network/members">
      <img src="https://img.shields.io/github/forks/opencloudhub/ai-ml-sklearn.svg?style=for-the-badge" alt="Forks">
    </a>
    <a href="https://github.com/opencloudhub/ai-ml-sklearn/stargazers">
      <img src="https://img.shields.io/github/stars/opencloudhub/ai-ml-sklearn.svg?style=for-the-badge" alt="Stars">
    </a>
    <a href="https://github.com/opencloudhub/ai-ml-sklearn/issues">
      <img src="https://img.shields.io/github/issues/opencloudhub/ai-ml-sklearn.svg?style=for-the-badge" alt="Issues">
    </a>
    <a href="https://github.com/opencloudhub/ai-ml-sklearn/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/opencloudhub/ai-ml-sklearn.svg?style=for-the-badge" alt="License">
    </a>
  </p>
</div>

______________________________________________________________________

<details>
  <summary>📑 Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#mlops-pipeline">MLOps Pipeline</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

______________________________________________________________________

<!-- # TODO: adjust readme -->

<h2 id="about">🍷 About</h2>

This repository demonstrates a complete MLOps pipeline for wine classification using scikit-learn and the UCI Wine dataset. It showcases production-ready machine learning practices including experiment tracking, hyperparameter optimization, model registration, and containerized deployment.

**Key Technologies:**

- **ML Framework**: Scikit-learn (Logistic Regression)
- **Experiment Tracking**: MLflow
- **Hyperparameter Optimization**: Optuna
- **Containerization**: Docker with multi-stage builds
- **Dependency Management**: UV (fast Python package manager)
- **Development**: DevContainers for consistent environments

<p align="right">(<a href="#readme-top">back to top</a>)</p>

______________________________________________________________________

<h2 id="features">✨ Features</h2>

- 🔬 **Experiment Tracking**: Complete MLflow integration with model registry
- 🎯 **Hyperparameter Tuning**: Automated optimization using Optuna
- 🐳 **Containerized Training**: Docker-based training environment
- 📊 **Model Evaluation**: Comprehensive metrics and visualization
- 🚀 **CI/CD Ready**: GitHub Actions workflows for automated training
- 📁 **MLflow Projects**: Standardized, reproducible ML workflows
- 🔄 **Model Registration**: Threshold-based automatic model promotion
- 🧪 **Development Environment**: VS Code DevContainer setup

<p align="right">(<a href="#readme-top">back to top</a>)</p>

______________________________________________________________________

<h2 id="getting-started">🚀 Getting Started</h2>

### Prerequisites

- Docker and Docker Compose
- VS Code with DevContainers extension (recommended)
- MLflow tracking server (for remote tracking)

### Local Development

1. **Clone the repository**

   ```bash
   git clone https://github.com/opencloudhub/ai-ml-sklearn.git
   cd ai-ml-sklearn
   ```

1. **Open in DevContainer** (Recommended)

   ```bash
   code .
   # VS Code will prompt to reopen in container
   ```

1. **Or setup locally**

   ```bash
   # Install UV
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync --dev
   ```

### Quick Start

```bash
# Start Mlflow locally
mlflow server --host 0.0.0.0 --port 8081

# Export needed env variables
export MLFLOW_TRACKING_URI=http://172.17.0.1:8081
export MLFLOW_EXPERIMENT_NAME=wine-quality

# Run basic training
uv run src/training.py

# Run hyperparameter optimization
uv run src/hyperparameter_tuning.py --n_trials 5

# Using MLflow Project
mlflow run . --entry-point training
mlflow run . --entry-point hyperparameter_tuning -P n_trials=50
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

______________________________________________________________________

<h2 id="usage">💻 Usage</h2>

### Training Options

**Basic Training:**

```bash
python src/training.py --C 1.0 --max_iter 100 --solver lbfgs
```

**Hyperparameter Optimization:**

```bash
python src/hyperparameter_tuning.py --n_trials 50 --test_size 0.2
```

**MLflow Projects:**

```bash
# Training with parameters
mlflow run . --entry-point training \
  -P C=0.5 \
  -P max_iter=200 \
  -P solver=saga

# Hyperparameter tuning
mlflow run . --entry-point hyperparameter_tuning \
  -P n_trials=100
```

### Model Registration

Set environment variables for automatic model registration:

```bash
export MLFLOW_TRACKING_URI=http://172.17.0.1:8081
export MLFLOW_EXPERIMENT_NAME=wine-quality
export REGISTERED_MODEL_NAME="staging.wine_classifier"
export REGISTERED_MODEL_THRESHOLD="0.85"

python src/training.py
```

### Docker Usage

```bash
# Build training image
docker build --target prod -t wine-classifier .

# Run training
docker run --rm \
  -v $(pwd):/mlflow/projects/code \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  wine-classifier python src/training.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

______________________________________________________________________

<h2 id="project-structure">📁 Project Structure</h2>

```
ai-ml-sklearn/
├── notebooks/
│   └── exploring_wine_dataset.ipynb    # Data exploration
├── src/
│   ├── training.py                     # Main training script
│   ├── hyperparameter_tuning.py       # Optuna optimization
│   ├── evaluate.py                     # Model evaluation utilities
│   └── _utils/
│       ├── get_or_create_experiment.py # MLflow experiment management
│       ├── logging_callback.py         # Optuna logging callbacks
│       └── logging_config.py           # Logging configuration
├── tests/                              # Unit tests
├── .devcontainer/                      # VS Code DevContainer config
├── .github/workflows/                  # CI/CD workflows
├── Dockerfile                          # Multi-stage container build
├── MLproject                           # MLflow project definition
├── pyproject.toml                      # Project dependencies and config
└── uv.lock                            # Dependency lock file
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

______________________________________________________________________

<h2 id="mlops-pipeline">🔄 MLOps Pipeline</h2>

### 1. Development & Experimentation

- Local development in DevContainers
- Jupyter notebooks for data exploration
- MLflow experiment tracking

### 2. Training & Optimization

- Automated hyperparameter tuning with Optuna
- Model evaluation and metrics logging
- Threshold-based model registration

### 3. Model Registry

- Automatic promotion to staging registry
- Model versioning and lineage tracking
- Performance comparison and rollback capability

### 4. Deployment (Planned)

- KServe model serving integration
- GitOps-based deployment automation
- Monitoring and drift detection

<p align="right">(<a href="#readme-top">back to top</a>)</p>

______________________________________________________________________

<h2 id="contributing">👥 Contributing</h2>

Contributions are welcome! This project follows OpenCloudHub's contribution standards.

Please see our [Contributing Guidelines](https://github.com/opencloudhub/.github/blob/main/.github/CONTRIBUTING.md) and [Code of Conduct](https://github.com/opencloudhub/.github/blob/main/.github/CODE_OF_CONDUCT.md) for more details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

______________________________________________________________________

<h2 id="license">📄 License</h2>

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

______________________________________________________________________

<h2 id="contact">📬 Contact</h2>

Organization Link: [https://github.com/OpenCloudHub](https://github.com/OpenCloudHub)

Project Link: [https://github.com/opencloudhub/ai-ml-sklearn](https://github.com/opencloudhub/ai-ml-sklearn)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

______________________________________________________________________

<h2 id="acknowledgements">🙏 Acknowledgements</h2>

- [UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine) - The dataset used for classification
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Optuna](https://optuna.org/) - Hyperparameter optimization framework
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager

<p align="right">(<a href="#readme-top">back to top</a>)</p>
