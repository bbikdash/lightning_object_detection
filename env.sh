

# Determine the directory of the current script
# Works for both bash and zsh
if [ -n "$BASH_SOURCE" ]; then
    # bash
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
elif [ -n "$ZSH_VERSION" ]; then
    # zsh
    DIR="${0:a:h}"
else
    echo "Unsupported shell. Please use bash or zsh."
    exit 1
fi

# The environment name
ENV_NAME=".venv"

# Set the ENV environmental variable
export LOCAL_ENV="$DIR"

# Deactivate any currently active virtual environment
# Check if the 'deactivate' function exists
type deactivate &>/dev/null
if [ $? -eq 0 ]; then
    echo "Deactivating the current virtual environment."
    deactivate
fi

# Check if the virtual environment already exists
if env_exists "$DIR/$ENV_NAME"; then
    echo "The virtual environment $ENV_NAME already exists. Activating it."
else
    echo "The virtual environment $ENV_NAME does not exist. Creating it."
    # Create the virtual environment
    python3 -m venv "$ENV_NAME"
    echo "Virtual environment $ENV_NAME created."
fi

# Activate the virtual environment
source "$DIR/$ENV_NAME/bin/activate"

# Install requirements if they exist in the requirements.txt file
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt"
    pip install --upgrade pip wheel setuptools
    pip install -e .
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

echo "Virtual environment $ENV_NAME is now active."