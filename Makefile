# Variables
NOTEBOOK=Flood_Prediction_FINAL.ipynb
ENV_NAME=env
REQUIREMENTS=requirements.txt

# Targets
.PHONY: all install run convert convert-pdf clean

# Default target: install dependencies and run the notebook
all: install run

# Create a virtual environment and install dependencies
install:
	python3 -m venv $(ENV_NAME)
	$(ENV_NAME)/bin/pip install -r $(REQUIREMENTS)

# Run the notebook
run:
	$(ENV_NAME)/bin/jupyter nbconvert --to notebook --execute $(NOTEBOOK)

# Convert notebook to HTML
convert:
	$(ENV_NAME)/bin/jupyter nbconvert --to html $(NOTEBOOK)

# Convert notebook to PDF (requires additional tools like LaTeX)
convert-pdf:
	$(ENV_NAME)/bin/jupyter nbconvert --to pdf $(NOTEBOOK)

# Clean up the generated files and virtual environment
clean:
	rm -rf $(ENV_NAME) $(NOTEBOOK).nbconvert $(NOTEBOOK).html $(NOTEBOOK).pdf
