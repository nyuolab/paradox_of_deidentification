
#!/bin/bash
# Create environment and install dependencies using uv
if [ "$(sw_vers -productName)" = "macOS" ]; then
    brew install rust
fi

# Check if uv is installed, if not install it
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi
conda deactivate
uv venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements.txt
