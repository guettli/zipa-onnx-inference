#!/usr/bin/env bash
# Bash Strict Mode: https://github.com/guettli/bash-strict-mode
trap 'echo -e "\n🚨 Warning: A command has failed. Exiting the script. Line was ($0:$LINENO): $(sed -n "${LINENO}p" "$0" 2>/dev/null || true)"; exit 3' ERR
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_CONF="$SCRIPT_DIR/../../pp/deploy.conf"
if [[ ! -f "$DEPLOY_CONF" ]]; then
    echo "Error: deploy.conf not found at $DEPLOY_CONF"
    exit 1
fi
# shellcheck disable=SC1090
source "$DEPLOY_CONF"

ZIPA_USER="zipa-onnx-inference"

echo "Building web app..."
cd "$SCRIPT_DIR/../web"
pnpm build
cd "$SCRIPT_DIR/.."

echo "Syncing static files..."
rsync -az --delete --stats web/dist/ "root@$REMOTE_HOST:/home/$ZIPA_USER/public_html/"

echo ""
echo "=== Deployment Complete ==="
echo "https://$REMOTE_HOST/zipa-onnx-inference/"
