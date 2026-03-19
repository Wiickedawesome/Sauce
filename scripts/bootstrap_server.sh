#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/bootstrap_server.sh — One-time server setup for Sauce
#
# Run this on a fresh VPS (Contabo, DO, etc.) to install Docker, clone the repo,
# and prepare for GitHub Actions CI/CD deployments.
#
# Usage (from your local machine):
#   ssh root@YOUR_VPS_IP 'bash -s' < scripts/bootstrap_server.sh
#
# After this script completes:
#   1. Copy your .env file:  scp .env root@YOUR_VPS_IP:/root/Sauce/.env
#   2. Set GitHub repo secrets: VPS_HOST, VPS_USER, VPS_SSH_KEY, VPS_APP_PATH
#   3. Push to main — GitHub Actions handles the rest.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

APP_PATH="/root/Sauce"
REPO_URL="https://github.com/Wiickedawesome/Sauce.git"

echo "=== Sauce Server Bootstrap ==="
echo "Target: ${APP_PATH}"
echo ""

# ── 1. System updates ────────────────────────────────────────────────────────
echo "--- Updating system packages ---"
apt-get update -qq
apt-get upgrade -y -qq

# ── 2. Install Docker ────────────────────────────────────────────────────────
if command -v docker &>/dev/null; then
    echo "--- Docker already installed: $(docker --version) ---"
else
    echo "--- Installing Docker ---"
    apt-get install -y -qq ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable docker
    systemctl start docker
    echo "--- Docker installed: $(docker --version) ---"
fi

# ── 3. Install basic tools ───────────────────────────────────────────────────
echo "--- Installing utilities ---"
apt-get install -y -qq git sqlite3 fail2ban ufw

# ── 4. Firewall ──────────────────────────────────────────────────────────────
echo "--- Configuring firewall ---"
ufw allow OpenSSH
ufw --force enable
echo "--- Firewall active (SSH allowed) ---"

# ── 5. Clone repo ────────────────────────────────────────────────────────────
if [[ -d "${APP_PATH}/.git" ]]; then
    echo "--- Repo already exists at ${APP_PATH}, pulling latest ---"
    cd "${APP_PATH}"
    git pull origin main
else
    echo "--- Cloning repo ---"
    git clone "${REPO_URL}" "${APP_PATH}"
    cd "${APP_PATH}"
fi

# ── 6. Create data directories ───────────────────────────────────────────────
mkdir -p "${APP_PATH}/data/logs"

# ── 7. Set up swap (safety net for memory) ────────────────────────────────────
if [[ ! -f /swapfile ]]; then
    echo "--- Creating 2G swap ---"
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "--- Swap enabled ---"
else
    echo "--- Swap already exists ---"
fi

# ── 8. Generate deploy SSH key ───────────────────────────────────────────────
DEPLOY_KEY="/root/.ssh/sauce_deploy"
if [[ ! -f "${DEPLOY_KEY}" ]]; then
    echo "--- Generating deploy SSH key ---"
    ssh-keygen -t ed25519 -f "${DEPLOY_KEY}" -N "" -C "sauce-deploy"
    cat "${DEPLOY_KEY}.pub" >> /root/.ssh/authorized_keys
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  IMPORTANT: Copy the PRIVATE key below into GitHub secret      ║"
    echo "║  Settings → Secrets → VPS_SSH_KEY                              ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    cat "${DEPLOY_KEY}"
    echo ""
else
    echo "--- Deploy key already exists at ${DEPLOY_KEY} ---"
fi

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy .env:       scp .env root@THIS_IP:${APP_PATH}/.env"
echo "  2. GitHub secrets:  VPS_HOST=<this IP>"
echo "                      VPS_USER=root"
echo "                      VPS_SSH_KEY=<private key printed above>"
echo "                      VPS_APP_PATH=${APP_PATH}"
echo "  3. Push to main — CI/CD will build and start the container."
echo ""
echo "  Or start manually:  cd ${APP_PATH} && docker compose -f docker/docker-compose.yml up --build -d"
