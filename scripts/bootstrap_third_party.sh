#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
REPO_DIR="${THIRD_PARTY_DIR}/HrSegNet4CrackSegmentation"
REPO_URL="https://github.com/CHDyshli/HrSegNet4CrackSegmentation.git"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "Official repo already exists: ${REPO_DIR}"
  git -C "${REPO_DIR}" pull --ff-only
else
  git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

echo "Ready: ${REPO_DIR}"
