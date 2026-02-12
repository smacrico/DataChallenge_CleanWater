# PowerShell script for Water Quality SA Predictor
# Windows alternative to Makefile commands

param(
    [Parameter(Position=0)]
    [ValidateSet('help', 'install', 'lint', 'format', 'test', 'smoke', 'train', 'predict', 'docs', 'export', 'all', 'clean')]
    [string]$Command = 'help'
)

function Show-Help {
    Write-Host "Available commands:" -ForegroundColor Cyan
    Write-Host "  install    - Install dependencies and pre-commit hooks"
    Write-Host "  lint       - Run linters (ruff, black, isort) in check mode"
    Write-Host "  format     - Auto-format code with ruff, black, isort"
    Write-Host "  test       - Run pytest test suite"
    Write-Host "  smoke      - Run notebook smoke tests"
    Write-Host "  train      - Run model training pipeline"
    Write-Host "  predict    - Generate submission.csv"
    Write-Host "  docs       - Generate documentation (Model Card, Business Plan, Video Script)"
    Write-Host "  export     - Export docs to HTML and PDF"
    Write-Host "  all        - Run full end-to-end pipeline"
    Write-Host "  clean      - Remove generated files and caches"
    Write-Host ""
    Write-Host "Usage: .\run.ps1 <command>" -ForegroundColor Yellow
    Write-Host "Example: .\run.ps1 install" -ForegroundColor Yellow
}

function Install-Dependencies {
    Write-Host "Installing dependencies..." -ForegroundColor Green
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    pre-commit install
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "pre-commit install failed. You may need to install it first: pip install pre-commit"
    }
    Write-Host "Installation completed!" -ForegroundColor Green
}

function Run-Lint {
    Write-Host "Running linters..." -ForegroundColor Green
    ruff check src/ tests/
    black --check src/ tests/
    isort --check-only src/ tests/
}

function Run-Format {
    Write-Host "Formatting code..." -ForegroundColor Green
    ruff check --fix src/ tests/
    black src/ tests/
    isort src/ tests/
    Write-Host "Code formatted!" -ForegroundColor Green
}

function Run-Tests {
    Write-Host "Running tests..." -ForegroundColor Green
    pytest -v --cov=src/wqsa --cov-report=term-missing
}

function Run-Smoke {
    Write-Host "Running smoke tests..." -ForegroundColor Green
    pytest tests/ -v -k smoke
    
    Write-Host "Running notebook smoke test..." -ForegroundColor Green
    try {
        python -m nbclient --execute notebooks/01_ingest_and_stage_check.ipynb --timeout=300
    }
    catch {
        Write-Warning "Notebook smoke test failed or nbclient not installed"
    }
}

function Run-Train {
    Write-Host "Training models..." -ForegroundColor Green
    python -m src.wqsa.modeling.train_cv
}

function Run-Predict {
    Write-Host "Generating predictions..." -ForegroundColor Green
    python -m src.wqsa.modeling.predict
}

function Run-Docs {
    Write-Host "Generating documentation..." -ForegroundColor Green
    python -m src.wqsa.docs.generate_model_card
    python -m src.wqsa.docs.generate_business_plan
    python -m src.wqsa.docs.generate_video_script
    Write-Host "Documentation generated!" -ForegroundColor Green
}

function Run-Export {
    Write-Host "Exporting documentation..." -ForegroundColor Green
    python -m src.wqsa.docs.export_to_html
    python -m src.wqsa.docs.export_to_pdf
    Write-Host "Documentation exported!" -ForegroundColor Green
}

function Run-All {
    Write-Host "Running full pipeline..." -ForegroundColor Cyan
    Run-Lint
    if ($LASTEXITCODE -ne 0) { Write-Error "Linting failed"; return }
    
    Run-Tests
    if ($LASTEXITCODE -ne 0) { Write-Error "Tests failed"; return }
    
    Run-Train
    if ($LASTEXITCODE -ne 0) { Write-Error "Training failed"; return }
    
    Run-Predict
    if ($LASTEXITCODE -ne 0) { Write-Error "Prediction failed"; return }
    
    Run-Docs
    if ($LASTEXITCODE -ne 0) { Write-Error "Documentation generation failed"; return }
    
    Run-Export
    if ($LASTEXITCODE -ne 0) { Write-Error "Export failed"; return }
    
    Write-Host "Full pipeline completed successfully!" -ForegroundColor Green
}

function Clean-Project {
    Write-Host "Cleaning project..." -ForegroundColor Green
    
    # Remove Python cache
    Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Include .pytest_cache -Recurse -Force | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Include .ipynb_checkpoints -Recurse -Force | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Filter "*.pyc" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
    
    # Remove coverage files
    Remove-Item htmlcov -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item .coverage -Force -ErrorAction SilentlyContinue
    
    # Remove model files (optional - uncomment if needed)
    # Get-ChildItem -Path models -Filter "*.pkl" | Remove-Item -Force -ErrorAction SilentlyContinue
    # Get-ChildItem -Path models -Filter "*.joblib" | Remove-Item -Force -ErrorAction SilentlyContinue
    
    Write-Host "Cleanup completed!" -ForegroundColor Green
}

# Main execution
switch ($Command) {
    'help'    { Show-Help }
    'install' { Install-Dependencies }
    'lint'    { Run-Lint }
    'format'  { Run-Format }
    'test'    { Run-Tests }
    'smoke'   { Run-Smoke }
    'train'   { Run-Train }
    'predict' { Run-Predict }
    'docs'    { Run-Docs }
    'export'  { Run-Export }
    'all'     { Run-All }
    'clean'   { Clean-Project }
    default   { Show-Help }
}
