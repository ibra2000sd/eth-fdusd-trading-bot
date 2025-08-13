#!/bin/bash

# ETH/FDUSD Advanced Trading Bot - Installation Script
# This script automates the complete installation and setup process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root for security reasons"
fi

# Banner
echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                ETH/FDUSD Advanced Trading Bot               ║
║                    Installation Script                      ║
║                                                              ║
║  Sophisticated algorithmic trading system with proprietary  ║
║  mathematical models for identifying market bottoms & tops  ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

log "Starting ETH/FDUSD Trading Bot installation..."

# Check system requirements
log "Checking system requirements..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    error "Python 3 is required but not installed"
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
fi

log "Python version check passed: $PYTHON_VERSION"

# Check available memory
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$MEMORY_GB" -lt 8 ]; then
    warn "System has ${MEMORY_GB}GB RAM. 16GB+ recommended for optimal performance"
fi

# Check disk space
DISK_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$DISK_SPACE" -lt 10 ]; then
    warn "Available disk space: ${DISK_SPACE}GB. 100GB+ recommended"
fi

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    DISTRO=$(lsb_release -si 2>/dev/null || echo "Unknown")
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    warn "Unknown operating system: $OSTYPE"
    OS="unknown"
fi

log "Detected OS: $OS"

# Install system dependencies
log "Installing system dependencies..."

if [[ "$OS" == "linux" ]]; then
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            curl \
            git \
            wget \
            postgresql-client \
            redis-tools \
            python3-pip \
            python3-venv \
            python3-dev \
            libpq-dev \
            libffi-dev \
            libssl-dev
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum update -y
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            curl \
            git \
            wget \
            postgresql \
            redis \
            python3-pip \
            python3-devel \
            postgresql-devel \
            libffi-devel \
            openssl-devel
    fi
elif [[ "$OS" == "macos" ]]; then
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        log "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew update
    brew install postgresql redis git wget
fi

# Install TA-Lib
log "Installing TA-Lib..."

if [[ "$OS" == "linux" ]]; then
    # Download and compile TA-Lib
    cd /tmp
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr/local
    make
    sudo make install
    sudo ldconfig
    cd ~
elif [[ "$OS" == "macos" ]]; then
    brew install ta-lib
fi

# Create project directory
PROJECT_DIR="$HOME/eth-fdusd-trading-bot"
log "Creating project directory: $PROJECT_DIR"

if [ -d "$PROJECT_DIR" ]; then
    warn "Project directory already exists. Backing up..."
    mv "$PROJECT_DIR" "${PROJECT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Download the trading bot
log "Downloading ETH/FDUSD Trading Bot..."

# If this script is part of the package, copy files
if [ -f "../src/main.py" ]; then
    cp -r ../* .
else
    # Download from repository (placeholder - replace with actual repo)
    git clone https://github.com/ibra2000sd/eth-fdusd-trading-bot.git .
fi

# Create virtual environment
log "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
log "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
log "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
log "Creating necessary directories..."
mkdir -p logs data/sample_data data/backtest_results config

# Copy environment template
log "Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    log "Created .env file from template"
    warn "Please edit .env file with your Binance API credentials"
fi

# Setup database (optional)
read -p "Do you want to set up PostgreSQL database? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Setting up PostgreSQL database..."
    
    # Check if PostgreSQL is running
    if ! pgrep -x "postgres" > /dev/null; then
        if [[ "$OS" == "linux" ]]; then
            sudo systemctl start postgresql
            sudo systemctl enable postgresql
        elif [[ "$OS" == "macos" ]]; then
            brew services start postgresql
        fi
    fi
    
    # Create database and user
    sudo -u postgres psql << EOF
CREATE DATABASE trading_bot;
CREATE USER trading_user WITH PASSWORD 'trading_password_123';
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;
\q
EOF
    
    log "Database setup completed"
fi

# Setup Redis (optional)
read -p "Do you want to set up Redis? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Setting up Redis..."
    
    if [[ "$OS" == "linux" ]]; then
        sudo systemctl start redis-server
        sudo systemctl enable redis-server
    elif [[ "$OS" == "macos" ]]; then
        brew services start redis
    fi
    
    log "Redis setup completed"
fi

# Create systemd service (Linux only)
if [[ "$OS" == "linux" ]]; then
    read -p "Do you want to create a systemd service? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Creating systemd service..."
        
        sudo tee /etc/systemd/system/eth-trading-bot.service > /dev/null << EOF
[Unit]
Description=ETH/FDUSD Advanced Trading Bot
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python src/main.py --mode production
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl daemon-reload
        sudo systemctl enable eth-trading-bot
        
        log "Systemd service created and enabled"
    fi
fi

# Run initial tests
log "Running initial system tests..."

# Test Python imports
python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from bot.trading_engine import TradingEngine
    from analyzers.mathematical_models import MathematicalModels
    from risk_management.risk_manager import RiskManager
    print('✓ Core modules import successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

# Test configuration
python3 src/main.py --validate

log "Installation completed successfully!"

echo -e "${GREEN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                    Installation Complete!                   ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BLUE}Next Steps:${NC}"
echo "1. Edit the .env file with your Binance API credentials:"
echo "   nano .env"
echo ""
echo "2. Test the configuration:"
echo "   source venv/bin/activate"
echo "   python src/main.py --validate"
echo ""
echo "3. Run backtesting (optional):"
echo "   python src/main.py --backtest"
echo ""
echo "4. Start the trading bot:"
echo "   python src/main.py --mode production"
echo ""
echo "5. Or use systemd service (if created):"
echo "   sudo systemctl start eth-trading-bot"
echo "   sudo systemctl status eth-trading-bot"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "- Always test with testnet first (BINANCE_TESTNET=true)"
echo "- Review risk management settings before live trading"
echo "- Monitor logs regularly: tail -f logs/trading_bot.log"
echo ""
echo -e "${GREEN}Documentation:${NC}"
echo "- Installation Guide: docs/installation.md"
echo "- Configuration: docs/configuration.md"
echo "- Trading Strategy: docs/trading_strategy.md"
echo ""
echo -e "${GREEN}Support:${NC}"
echo "- GitHub Issues: https://github.com/ibra2000sd/eth-fdusd-trading-bot/issues"
echo "- Documentation: https://your-docs-site.com"
echo ""

log "Happy trading! 🚀"

