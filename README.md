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
    Scikit-learn wine classification with a modern MLOps pipeline featuring MLflow tracking and Ray for distributed training, hyperparameter optimization and serving.<br />
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
    <li><a href="#workflow-tags">CI/CD Tagging<a/></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

______________________________________________________________________

<h2 id="about">🍷 About</h2>

This repository demonstrates an example implementation for wine classification using scikit-learn and the UCI Wine dataset. It showcases combining machine learning practices including experiment tracking, hyperparameter optimization, model registration, and containerized deployment and serves as demonstration within the OpenCloudHub project.\\

______________________________________________________________________

<h2 id="features">✨ Features</h2>

- 🔬 **Experiment Tracking**: MLflow integration with model registry
- 🎯 **Hyperparameter Tuning**: Automated optimization using Optuna
- 🐳 **Containerized Training**: Docker-based training environment with UV
- ⚡ **Distributed Training & Serving**: Ray for scalable workflows
- 🚀 **CI/CD Ready**: GitHub Actions workflows for automated training and CI
- 🧪 **Development Environment**: VS Code DevContainer setup

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

1. **Start local MLflow tracking server**

   ```bash
      mlflow server --host 0.0.0.0 --port 8081
   ```

   Access at `http://localhost:8081`

1. **Start local Ray cluster**

   ```bash
      ray start --head
   ```

   Access dashboard at `http://127.0.0.1:8265`

You're now ready to develop, train and serve models locally!

### Training

**Basic training:**

```bash
python src/training/train.py --C 0.9
```

or use the Job API like we would do in practise too

```bash
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python src/training/train.py
```

### Model Serving

Ensure you have a trained model to load either from local folder or from mlflow by setting the 'MODEL_URI' environment variable.

**Start the serving application:**

```bash
serve run src.serving.serve:app_builder model_uri="models:/ci.wine-classifier/9" --reload
```

or even better and more production ready, run:

```bash
serve build src.serving.serve:app_builder -o src/serving/serve_config.yaml
serve deploy src/serving/serve_config.yaml
```

Access Swagger docs at `http://localhost:8000/docs`

### Production Training

Trigger the workflow dispatch in Github Actions at `https://github.com/OpenCloudHub/ai-ml-sklearn/actions/workflows/train.yaml`

______________________________________________________________________

<h2 id="workflow-tags">🏷️ Workflow tags, DVC data version & MinIO access</h2>

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
│   ├── training/                       # Training and optimization scripts
│   │   ├── train.py
│   │   ├── optimize_hyperparameters.py
│   │   └── evaluate.py
│   ├── serving/                        # Model serving (Ray Serve/FastAPI)
│   │   └── wine_classifier.py
│   └── _utils/                         # Shared utilities
│       ├── get_or_create_experiment.py
│       ├── logging_config.py
│       └── mlflow_tags.py
├── tests/                              # Unit tests
├── .devcontainer/                      # VS Code DevContainer config
├── .github/workflows/                  # CI/CD workflows
├── Dockerfile                          # Multi-stage container build
├── MLproject                           # MLflow project definition
├── pyproject.toml                      # Project dependencies and config
└── uv.lock                             # Dependency lock file
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

- [UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine) - The dataset used for classification
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Optuna](https://optuna.org/) - Hyperparameter optimization framework
- [Ray](https://ray.io/) - Distributed computing and serving
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager

<p align="right">(<a href="#readme-top">back to top</a>)</p>
