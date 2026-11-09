from pathlib import Path
from src.generate_data import build_dataset
from src.pipeline import run_pipeline

if __name__ == '__main__':
    build_dataset()
    summary = run_pipeline()
    print('Project run complete. Summary:')
    for key, value in summary.items():
        print(f'  {key}: {value}')
    print('See evidence/ for forecasts, evaluation metrics, and expiry risk outputs.')

