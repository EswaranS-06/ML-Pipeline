Write-Host "Starting SIEM Dashboard..." -ForegroundColor Green
Write-Host ""
Write-Host "Make sure you have installed all dependencies:" -ForegroundColor Yellow
Write-Host "pip install -r requirements.txt" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting Streamlit server..." -ForegroundColor Green

# Check if streamlit is installed
try {
    $streamlitCheck = python -c "import streamlit; print('OK')" 2>$null
    if ($streamlitCheck -ne "OK") {
        Write-Host "Streamlit not found. Installing..." -ForegroundColor Yellow
        pip install streamlit plotly psutil
    }
}
catch {
    Write-Host "Error checking Streamlit installation" -ForegroundColor Red
}

# Run the dashboard
streamlit run app.py