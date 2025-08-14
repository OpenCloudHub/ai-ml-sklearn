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
    Scikit-learn wine classification with a modern MLOps pipeline featuring MLflow tracking, Ray for distributed training and serving, hyperparameter optimization, and production-ready deployment patterns.<br />
    <a href="https://github.com/opencloudhub"><strong>Explore OpenCloudHub »</strong></a>
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

<h2 id="about">🍷 About</h2>

This repository demonstrates a complete MLOps pipeline for wine classification using scikit-learn and the UCI Wine dataset. It showcases production-ready machine learning practices including experiment tracking, hyperparameter optimization, model registration, and containerized deployment.\
**Ray** is used for distributed training and scalable model serving.

**Key Technologies:**

- **ML Framework**: Scikit-learn (Logistic Regression)
- **Distributed Training & Serving**: Ray
- **Experiment Tracking**: MLflow
- **Hyperparameter Optimization**: Optuna
- **Containerization**: Docker
- **Dependency Management**: UV
- **Development**: DevContainers for consistent environments

______________________________________________________________________

<h2 id="features">✨ Features</h2>

- 🔬 **Experiment Tracking**: MLflow integration with model registry
- 🎯 **Hyperparameter Tuning**: Automated optimization using Optuna
- 🐳 **Containerized Training**: Docker-based training environment
- ⚡ **Distributed Training & Serving**: Ray for scalable workflows
- 📊 **Model Evaluation**: Comprehensive metrics and visualization
- 🚀 **CI/CD Ready**: GitHub Actions workflows for automated training
- 📁 **MLflow Projects**: Standardized, reproducible ML workflows
- 🔄 **Model Registration**: Threshold-based automatic model promotion
- 🧪 **Development Environment**: VS Code DevContainer setup

______________________________________________________________________

<h2 id="getting-started">🚀 Getting Started</h2>

### Prerequisites

- Docker and Docker Compose
- VS Code with DevContainers extension (recommended)
- MLflow tracking server (for remote tracking)
- Ray (for distributed training/serving)

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

______________________________________________________________________

### MLflow Tracking Server

Start MLflow locally (accessible from Docker containers):

```bash
mlflow server --host 0.0.0.0 --port 8081
export MLFLOW_TRACKING_URI=http://0.0.0.0:8081
export MLFLOW_EXPERIMENT_NAME=wine-quality
export MLFLOW_TRACKING_INSECURE_TLS=true
```

______________________________________________________________________

### Ray Development Workflow

#### 1. Start a Local Ray Cluster

```bash
ray start --head
```

#### 2. Training Workflows

Submit Ray jobs for training and hyperparameter optimization:

```bash
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python src/training/train.py
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python src/training/optimize_hyperparameters.py
```

#### 3. Model Serving with Ray Serve

Make sure you have promoted a model to prod.wine-classifier with @champion alias, as service is looking for that
To run the model serving application locally:

```bash
serve run --working-dir /workspace/project src.serving.wine_classifier:deployment
```

______________________________________________________________________

<h2 id="usage">💻 Usage</h2>

#### Training

```bash
python src/training/train.py --C 1.0 --max_iter 100 --solver lbfgs
```

#### Hyperparameter Optimization

```bash
python src/training/optimize_hyperparameters.py --n_trials 50 --test_size 0.2
```

#### Local Model Serving

```bash
serve run --working-dir /workspace/project src.serving.wine_classifier:deployment
```

To test the model, run:

```bash
python tests/test_wine_classifier.py
```

You can also visit the Swagger documentetion of the Application at `http://localhost:8000/docs`

______________________________________________________________________

<h2 id="project-structure">📁 Project Structure</h2>

```
ai-ml-sklearn/
├── src/
│   ├── training/                       # Training and optimization scripts
│   │   ├── train.py
│   │   ├── optimize_hyperparameters.py
│   │   └── evaluate.py
│   ├── serving/                        # Model serving (Ray Serve/FastAPI)
│   │   └── wine_classifier.py
│   └── _utils/                         # Shared utilities
│       ├── get_or_create_experiment.py
│       ├── logging_callback.py
│       └── logging_config.py
├── tests/                              # Unit tests
├── .devcontainer/                      # VS Code DevContainer config
├── .github/workflows/                  # CI/CD workflows
├── Dockerfile                          # Multi-stage container build
├── MLproject                           # MLflow project definition
├── pyproject.toml                      # Project dependencies and config
└── uv.lock                             # Dependency lock file
```

______________________________________________________________________

<h2 id="mlops-pipeline">🔄 MLOps Pipeline</h2>

1. **Development & Experimentation**

   - Local development in DevContainers
   - Jupyter notebooks for data exploration
   - MLflow experiment tracking

1. **Training & Optimization**

   - Distributed training and hyperparameter tuning with Ray and Optuna
   - Model evaluation and metrics logging
   - Threshold-based model registration

1. **Model Registry**

   - Automatic promotion to staging registry
   - Model versioning and lineage tracking
   - Performance comparison and rollback capability

1. **Deployment**

   - Ray Serve for scalable, production-ready model serving
   - (Planned) KServe integration and GitOps-based deployment automation

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

- [UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine) - The dataset used for classification
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Optuna](https://optuna.org/) - Hyperparameter optimization framework
- [Ray](https://ray.io/) - Distributed computing and serving
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager

<p align="right">(<a href="#readme-top">back to top</a>)</p>
