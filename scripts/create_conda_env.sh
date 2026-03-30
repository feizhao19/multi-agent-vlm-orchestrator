#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-mavlm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
INSTALL_LOCAL_DEPS="${INSTALL_LOCAL_DEPS:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS_DIR="${PROJECT_ROOT}/requirements"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi

# This environment variable avoids Conda plugin issues seen on some systems.
export CONDA_NO_PLUGINS=true

echo "Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
conda create --solver classic -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip

echo "Installing project into '${ENV_NAME}'"
conda run --live-stream -n "${ENV_NAME}" python -m pip install --upgrade pip
echo "Installing Python dependencies from ${REQUIREMENTS_DIR}/dev.txt"
conda run --live-stream -n "${ENV_NAME}" python -m pip install -r "${REQUIREMENTS_DIR}/dev.txt"
echo "Installing this project in editable mode"
conda run --live-stream -n "${ENV_NAME}" python -m pip install -e "${PROJECT_ROOT}"

if [[ "${INSTALL_LOCAL_DEPS}" == "1" ]]; then
  echo "Installing optional local VLM dependencies into '${ENV_NAME}'"
  conda run --live-stream -n "${ENV_NAME}" python -m pip install -r "${REQUIREMENTS_DIR}/local-cu126.txt"
fi

cat <<EOF

Conda environment created.

Activate it with:
  conda activate ${ENV_NAME}

Test it with:
  cd ${PROJECT_ROOT}
  PYTHONPATH=src python -m multi_agent_vlm_orchestrator.cli validate \\
    --models configs/models.json \\
    --scripts configs/scripts.json

Optional:
  INSTALL_LOCAL_DEPS=1 ./scripts/create_conda_env.sh ${ENV_NAME}

This installs by default:
  requirements/dev.txt

And if INSTALL_LOCAL_DEPS=1:
  requirements/local-cu126.txt

PyTorch is not installed by this script.
Install it yourself before local VLM runs, for example:
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
EOF
