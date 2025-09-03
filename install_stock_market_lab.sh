#!/bin/bash

# =============================================================================
# STOCK MARKET LAB INSTALLATION SCRIPT
# For WSL2 Ubuntu systems
# =============================================================================

echo "==============================================="
echo "STOCK MARKET LAB INSTALLATION SCRIPT"
echo "==============================================="
echo ""

# Function to check if command succeeded
check_success() {
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: $1 failed!"
        exit 1
    fi
    echo "✅ SUCCESS: $1 completed"
}

# Function to install package if not already installed
install_if_missing() {
    local package=$1
    local install_cmd=$2
    
    if ! dpkg -l | grep -q "^ii.*$package"; then
        echo "Installing $package..."
        sudo apt update
        eval "$install_cmd"
        check_success "Installation of $package"
    else
        echo "$package is already installed"
    fi
}

# Update system packages
echo "Updating system packages..."
sudo apt update
check_success "System package update"

# Install Docker
echo ""
echo "Installing Docker..."
install_if_missing "docker.io" "sudo apt install -y docker.io"
install_if_missing "docker-compose" "sudo apt install -y docker-compose"

# Start and enable Docker service
echo ""
echo "Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker
check_success "Docker service startup"

# Add current user to docker group
echo ""
echo "Adding user to docker group..."
sudo usermod -aG docker $USER
check_success "Added user to docker group"

# Install Node.js
echo ""
echo "Installing Node.js..."
install_if_missing "nodejs" "curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt-get install -y nodejs"
check_success "Node.js installation"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
cd /workspace/stock-market-lab/agents-python
pip3 install -r requirements.txt
check_success "Python dependencies installation"

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
cd /workspace/stock-market-lab/frontend-react
npm install
check_success "Frontend dependencies installation"

# Verify installations
echo ""
echo "Verifying installations..."

# Check Docker
echo "Checking Docker..."
docker --version
if [ $? -eq 0 ]; then
    echo "✅ Docker is available"
else
    echo "❌ Docker check failed"
fi

# Check Node.js
echo "Checking Node.js..."
node --version
if [ $? -eq 0 ]; then
    echo "✅ Node.js is available"
else
    echo "❌ Node.js check failed"
fi

# Check Python
echo "Checking Python packages..."
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
if [ $? -eq 0 ]; then
    echo "✅ PyTorch is available"
else
    echo "❌ PyTorch check failed"
fi

echo ""
echo "==============================================="
echo "INSTALLATION SUMMARY"
echo "==============================================="
echo "✅ Docker installed and running"
echo "✅ Node.js installed"
echo "✅ Python dependencies installed"
echo "✅ Frontend dependencies installed"
echo ""
echo "To start the application:"
echo "1. Change to infra directory: cd /workspace/stock-market-lab/infra"
echo "2. Start services: docker-compose up -d"
echo "3. Access dashboard at: http://localhost:3001"
echo ""
echo "To develop frontend:"
echo "1. Change to frontend directory: cd /workspace/stock-market-lab/frontend-react"
echo "2. Start dev server: npm start"
echo ""
echo "==============================================="
echo "INSTALLATION COMPLETE"
echo "==============================================="