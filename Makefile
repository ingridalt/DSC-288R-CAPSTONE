ENV_NAME = poverty_prediction
PYTHON_VERSION = 3.9

# setup: Create env, install deps, AND register the kernel for Jupyter
setup:
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y
	conda run -n $(ENV_NAME) pip install -r requirements.txt
	conda run -n $(ENV_NAME) python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python (Poverty Project)"
	@echo "Setup complete. Open your notebook and select 'Poverty Project' as your kernel."

# update: Just refreshes packages
update:
	conda run -n $(ENV_NAME) pip install -r requirements.txt

# freeze: Syncs requirements.txt
freeze:
	conda run -n $(ENV_NAME) pipreqs . --force


setup:
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y
	conda run -n $(ENV_NAME) pip install -r requirements.txt
	conda run -n $(ENV_NAME) python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python (Poverty Project)"
	@echo "-----------------------------------------------------------"
	@echo "SETUP COMPLETE"
	@echo "1. Run 'conda activate $(ENV_NAME)' to work in  terminal"
	@echo "2. Or  open Jupyter and select the '$(ENV_NAME)' kernel."
	@echo "-----------------------------------------------------------"