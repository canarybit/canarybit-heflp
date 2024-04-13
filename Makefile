# Name of the virtual env
VENV_NAME = .venv

# Path to requirements.txt file
REQUIREMENTS = ./requirements.txt

# Check if Python version is 3.8 or higher
PYTHON_VERSION := $(shell python3 -c "import sys; print('1' if sys.version_info >= (3, 8) else '0')")

# Check if pip is installed
HAS_PIP := $(shell command -v pip 2> /dev/null)

# Check if python-venv is installed
HAS_VENV := $(shell command -v python3 -m venv 2> /dev/null)

# Default target
all: initialization

# Create the virtual environment
initialization: check-prerequisites
	python3 -m venv $(VENV_NAME)
	@echo "Virtual environment $(VENV_NAME) created."
	@echo "Installing dependencies from $(REQUIREMENTS)..."
	@$(VENV_NAME)/bin/pip install -r $(REQUIREMENTS)
	@echo "Dependencies installed."
	mkdir -p .tmp
	mkdir -p .tmp/logs
	@echo "Done."
	@echo "To activate the virtual env, please use:"
	@echo "    source .venv/bin/activate"
	@echo "or"
	@echo "    make help"
	@echo ""

# Check if all prerequisites are ready 
check-prerequisites:
# Check if Python version is 3.8 or higher
ifeq ($(PYTHON_VERSION),0)
	@echo "Python 3.8 or higher is required. ref: https://realpython.com/installing-python/"
	@exit 1
endif
# Check if pip is installed
ifeq ($(HAS_PIP),)
	@echo "pip is not installed. Please install python3-pip first. ref: https://pip.pypa.io/en/stable/installation/"
	@exit 1
endif
ifeq ($(HAS_VENV),)
	@echo "Python venv is not installed. Installing..."
	python3 -m pip install virtualenv
endif

# Extract and pack the server code into one tar.gz file
pack-server:
	bash ./scripts/extract_server.sh -v -o .

# Generate certificates for SSL connections
gen-certificates:
	bash ./scripts/generate_cert.sh

# Help message
help:
	@echo "To activate the virtual env: source .venv/bin/activate"
	@echo "To deactivate the virtual env: deactivate"
	@echo "To extract and pack server: make pack_server"
	@echo "To generate certs and keys for SSL: make gen_certificates"
	@echo "To clean up the virtual env: make clean"

# Clean up virtual environment
clean:
	rm -rf $(VENV_NAME)

.PHONY: all initialization check-prerequisites pack_server gen_certificates activate deactivate