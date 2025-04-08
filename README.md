# Llama 4 Inference Guide

A comprehensive guide to setting up and running Llama 4 inference on high-performance hardware.

## Prerequisites

- **Hardware Requirements**: 
  - 8x NVIDIA H100 GPUs (minimum 5 GPUs for small context)
  - Alternative: NVIDIA A100 GPUs
  - Note: With INT4 quantization, fewer GPUs may be sufficient

## Installation Steps

### 1. Install Miniconda

```bash
# Download Miniconda installer
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installer
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts:
# - Accept license: yes
# - Installation location: [press ENTER for default]
# - Initialize Miniconda: yes

# Restart the terminal afterwards to apply the changes

```

### 2. Create Project Directory and Clone Repository

```bash
# Create and navigate to project directory
mkdir llama4
cd llama4

# Clone the repository (replace with actual repository URL)
git clone https://github.com/your-org/llama4-inference.git
cd llama4-inference
```

### 3. Set Up Python Environment

```bash
# Create a new conda environment with Python 3.12
conda create -n llamainference python=3.12

# Activate the environment
conda activate llamainference

# Install required packages
pip install -r requirements.txt
```

### 4. Authenticate with Hugging Face

```bash
# Log in to Hugging Face to access model files
huggingface-cli login
# Enter your Hugging Face token when prompted
```

## Running Inference

### Option 1: Serving with Web Interface

```bash
# Step 1: Start the model server in a tmux session
tmux new-session -s llama4-server
bash serve.sh
# Detach from tmux session with Ctrl+b then d

# Step 2: Start the Streamlit interface in another tmux session
tmux new-session -s llama4-ui
conda activate llamainference  # Make sure to activate environment again
streamlit run streamlit_chat.py --server.port 8080 --server.address 0.0.0.0
# Detach from tmux session with Ctrl+b then d
```

Access the web interface at: `http://[NODE_IP]:8080`

### Option 2: Batch Inference

For processing multiple inputs without the web interface:

```bash
# Activate environment if not already active
conda activate llamainference

# Run batch inference script
python batch_inference.py
```

## Additional Information

### Model Configuration

The model server (`serve.sh`) uses the following default configuration:
- Model: Llama 4 Maverick (17B Active/128B Total parameters)
- Context length: 430K Tokens
- Precision: FP8

### Monitoring and Management

To view or reattach to running tmux sessions:
```bash
# List sessions
tmux ls

# Reattach to a session
tmux attach-session -t llama4-server
# or
tmux attach-session -t llama4-ui
```

### Troubleshooting

- **Out of memory errors**: Reduce batch size or enable more aggressive quantization (Could do INT4)
- **Slow inference**: Check GPU utilization and network bandwidth

