#!/bin/bash

# Run all pipeline scripts in sequence
# Usage: ./scripts/run_all.sh

set -e  # Exit on any error

echo "========================================"
echo "Water Quality SA - Full Pipeline"
echo "========================================"

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.11+"
    exit 1
fi

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: No virtual environment detected."
    echo "Recommended: Run 'source venv/bin/activate' first"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Copy .env.example and configure."
    exit 1
fi

echo ""
echo "Step 1: Training models..."
python -m src.wqsa.modeling.train_cv
if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "Step 2: Generating predictions..."
python -m src.wqsa.modeling.predict
if [ $? -ne 0 ]; then
    echo "Error: Prediction failed"
    exit 1
fi

echo ""
echo "Step 3: Generating documentation..."
python -m src.wqsa.docs.generate_model_card
python -m src.wqsa.docs.generate_business_plan
python -m src.wqsa.docs.generate_video_script

echo ""
echo "Step 4: Exporting to HTML and PDF..."
python -m src.wqsa.docs.export_to_html
python -m src.wqsa.docs.export_to_pdf

echo ""
echo "========================================"
echo "Pipeline completed successfully!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  - Models: models/"
echo "  - Submission: artifacts/submission.csv"
echo "  - Docs: artifacts/*.md, *.html, *.pdf"
echo ""
