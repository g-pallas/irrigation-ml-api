# Irrigation Recommendation Model

This starter model predicts an irrigation action for your capstone app using soil readings collected in the field.

#Hello

## Target classes

- `irrigate_now`: irrigation should be applied now
- `schedule_soon`: irrigation should be scheduled soon
- `hold_irrigation`: no irrigation is needed yet

The current primary model path uses the USDA SCAN Arizona hourly dataset with multi-year Walnut Gulch #1 data and a threshold-based three-class output. The USDA SCAN daily workflow is kept as a baseline comparison path, and the Mendeley workflow remains as a secondary comparison path.

## Input features

- `moisture`
- `temperature`
- `humidity`
- `zone`

## Install

```powershell
pip install -r requirements_irrigation.txt
```

## Train

```powershell
python train_irrigation_model.py --csv prepared_soil_data_scan_hourly.csv --model-out irrigation_model_scan_hourly.pkl
```

## Primary USDA SCAN hourly workflow

Use the hourly Arizona exports when you want the main high-resolution model for the app:

```powershell
python prepare_usda_scan_hourly.py --input "C:\Users\kenji\Downloads\usdascan\hourly\2026_ALL_YEAR=2024_hourly.csv" --output prepared_soil_data_scan_hourly.csv --zone "Walnut Gulch #1, AZ" --label-mode three_class --date-col "Date" --time-col "Time" --moisture-col "SMS.I-1:-2 (pct) (loam)" --temperature-col "STO.I-1:-2 (degC)" --humidity-col "RHUM.I-1 (pct)"
```

Train the hourly model:

```powershell
python train_irrigation_model.py --csv prepared_soil_data_scan_hourly.csv --model-out irrigation_model_scan_hourly.pkl
```

Run the full hourly pipeline in one command:

```powershell
.\run_scan_hourly_pipeline.ps1
```

The hourly pipeline now defaults to `three_class`, which uses threshold-based labels that are easier to explain for real irrigation decisions.

For the current Walnut Gulch hourly dataset, the default thresholds are tuned to the actual moisture scale in the data:

- `dry_threshold = 6`
- `wet_threshold = 10`

That produces a usable three-class split on this dataset while still keeping the meaning intuitive:

- `irrigate_now` when moisture is below the effective dry threshold
- `schedule_soon` when moisture is between the dry and wet thresholds
- `hold_irrigation` when moisture is above the effective wet threshold

The effective thresholds are adjusted slightly by humidity and temperature:

- higher humidity raises the thresholds a little
- hotter temperatures lower the thresholds a little
- very cool temperatures raise the thresholds a little

Use multiple hourly years:

```powershell
.\run_scan_hourly_pipeline.ps1 -InputFiles "C:\Users\kenji\Downloads\usdascan\hourly\2026_ALL_YEAR=2021_hourly.csv","C:\Users\kenji\Downloads\usdascan\hourly\2026_ALL_YEAR=2022_hourly.csv","C:\Users\kenji\Downloads\usdascan\hourly\2026_ALL_YEAR=2023_hourly.csv","C:\Users\kenji\Downloads\usdascan\hourly\2026_ALL_YEAR=2024_hourly.csv","C:\Users\kenji\Downloads\usdascan\hourly\2026_ALL_YEAR=2025_hourly.csv"
```

Evaluate the primary hourly model:

```powershell
python evaluate_irrigation_model.py --csv prepared_soil_data_scan_hourly.csv --model irrigation_model_scan_hourly.pkl --out-dir evaluation_outputs_scan_hourly
```

## USDA SCAN daily baseline workflow

For the USDA SCAN Arizona direction using 2-inch soil moisture, 2-inch soil temperature, and relative humidity:

```powershell
python prepare_usda_scan_daily.py --input scan_arizona_daily.csv --output prepared_soil_data_scan.csv --zone "Walnut Gulch #1, AZ"
```

Then train:

```powershell
python train_irrigation_model.py --csv prepared_soil_data_scan.csv --model-out irrigation_model_scan.pkl
```

For the Walnut Gulch #1 export with explicit 2-inch headers:

```powershell
python prepare_usda_scan_daily.py --input "C:\Users\kenji\Downloads\usdascan\2026_ALL_YEAR=2024.csv" --output prepared_soil_data_scan.csv --zone "Walnut Gulch #1, AZ" --date-col "Date" --moisture-col "SMS.I-1:-2 (pct) (loam)" --temperature-col "STO.I-1:-2 (degC)" --humidity-col "RHUM.I-1 (pct)"
```

The SCAN prep script now:

- skips the metadata rows before the real USDA header
- strips weird extra spaces in SCAN column names
- removes invalid sensor sentinels like `-99.9`
- supports combining multiple yearly files in one run

To merge multiple years:

```powershell
python prepare_usda_scan_daily.py --input "C:\Users\kenji\Downloads\usdascan\2026_ALL_YEAR=2022.csv" "C:\Users\kenji\Downloads\usdascan\2026_ALL_YEAR=2023.csv" "C:\Users\kenji\Downloads\usdascan\2026_ALL_YEAR=2024.csv" --output prepared_soil_data_scan.csv --zone "Walnut Gulch #1, AZ" --date-col "Date" --moisture-col "SMS.I-1:-2 (pct) (loam)" --temperature-col "STO.I-1:-2 (degC)" --humidity-col "RHUM.I-1 (pct)"
```

If the binary split is still too imbalanced for the Arizona site, try quantile labels:

```powershell
python prepare_usda_scan_daily.py --input "C:\Users\kenji\Downloads\usdascan\2026_ALL_YEAR=2022.csv" "C:\Users\kenji\Downloads\usdascan\2026_ALL_YEAR=2023.csv" "C:\Users\kenji\Downloads\usdascan\2026_ALL_YEAR=2024.csv" --output prepared_soil_data_scan.csv --zone "Walnut Gulch #1, AZ" --label-mode quantile --date-col "Date" --moisture-col "SMS.I-1:-2 (pct) (loam)" --temperature-col "STO.I-1:-2 (degC)" --humidity-col "RHUM.I-1 (pct)"
```

Run the SCAN pipeline in one command:

```powershell
.\run_scan_pipeline.ps1
```

Use multiple years:

```powershell
.\run_scan_pipeline.ps1 -InputFiles "C:\Users\kenji\Downloads\usdascan\2026_ALL_YEAR=2022.csv","C:\Users\kenji\Downloads\usdascan\2026_ALL_YEAR=2023.csv","C:\Users\kenji\Downloads\usdascan\2026_ALL_YEAR=2024.csv"
```

Force a different label mode:

```powershell
.\run_scan_pipeline.ps1 -LabelMode binary
```

Evaluate the primary SCAN model:

```powershell
python evaluate_irrigation_model.py --csv prepared_soil_data_scan.csv --model irrigation_model_scan.pkl --out-dir evaluation_outputs_scan
```

## Secondary Mendeley workflow

Prepare the Mendeley workbook:

```powershell
python prepare_mendeley_daily_average.py --input "C:\Users\kenji\Downloads\mendeley_3w3pf3vnd4-1\unzipped\Agricultural Irrigation Control Data\DailyAverageSensedData1.xlsx" --output prepared_soil_data_mendeley.csv
```

Train the main Mendeley model:

```powershell
python train_irrigation_model.py --csv prepared_soil_data_mendeley.csv --model-out irrigation_model_mendeley.pkl
```

Evaluate it:

```powershell
python evaluate_irrigation_model.py --csv prepared_soil_data_mendeley.csv --model irrigation_model_mendeley.pkl --out-dir evaluation_outputs_mendeley
```

Run the full Mendeley pipeline in one command:

```powershell
.\run_mendeley_pipeline.ps1
```

## Secondary Mendeley workflow details

For the Mendeley `DailyAverageSensedData1.xlsx` workbook:

```powershell
python prepare_mendeley_daily_average.py --input "C:\Users\kenji\Downloads\mendeley_3w3pf3vnd4-1\unzipped\Agricultural Irrigation Control Data\DailyAverageSensedData1.xlsx" --output prepared_soil_data_mendeley.csv
```

Then train on the prepared file:

```powershell
python train_irrigation_model.py --csv prepared_soil_data_mendeley.csv --model-out irrigation_model_mendeley.pkl
```

## Evaluate and export

Create appendix-ready evaluation outputs after training:

For the primary SCAN hourly model, use:

```powershell
python evaluate_irrigation_model.py --csv prepared_soil_data_scan_hourly.csv --model irrigation_model_scan_hourly.pkl --out-dir evaluation_outputs_scan_hourly
```

For the daily SCAN baseline, use:

```powershell
python evaluate_irrigation_model.py --csv prepared_soil_data_scan.csv --model irrigation_model_scan.pkl --out-dir evaluation_outputs_scan
```

For the Mendeley comparison model, use:

```powershell
python evaluate_irrigation_model.py --csv prepared_soil_data_mendeley.csv --model irrigation_model_mendeley.pkl --out-dir evaluation_outputs_mendeley
```

## Run everything in one command

For the primary SCAN hourly path:

```powershell
.\run_scan_hourly_pipeline.ps1
```

For the daily SCAN baseline path:

```powershell
.\run_scan_pipeline.ps1
```

## Predict

```powershell
python predict_irrigation.py --model irrigation_model_scan_hourly.pkl --moisture 4.0 --temperature 35 --humidity 30 --zone "Walnut Gulch #1, AZ"
python predict_irrigation.py --model irrigation_model_scan_hourly.pkl --moisture 7.5 --temperature 24 --humidity 45 --zone "Walnut Gulch #1, AZ"
python predict_irrigation.py --model irrigation_model_scan_hourly.pkl --moisture 12.0 --temperature 24 --humidity 70 --zone "Walnut Gulch #1, AZ"
```

For the current Walnut Gulch hourly model, representative ranges are:

- around `4` moisture: usually `schedule_soon`
- around `7.5` moisture: usually `schedule_soon`
- around `12` moisture: usually `hold_irrigation`

This is because the SCAN hourly source data uses a much lower moisture range than the demo percentages shown in the mobile UI.

## Run the API server

Install API dependencies:

```powershell
pip install -r api_requirements.txt
```

Start the FastAPI backend:

```powershell
uvicorn ml_api_server:app --host 0.0.0.0 --port 8000
```

Serve the primary SCAN hourly model:

```powershell
$env:IRRIGATION_MODEL_PATH="irrigation_model_scan_hourly.pkl"
uvicorn ml_api_server:app --host 0.0.0.0 --port 8000
```

Or serve the daily SCAN baseline model:

```powershell
$env:IRRIGATION_MODEL_PATH="irrigation_model_scan.pkl"
uvicorn ml_api_server:app --host 0.0.0.0 --port 8000
```

Or serve the Mendeley comparison model:

```powershell
$env:IRRIGATION_MODEL_PATH="irrigation_model_mendeley.pkl"
uvicorn ml_api_server:app --host 0.0.0.0 --port 8000
```

Test the API with the default SCAN payload:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -InFile "sample_predict_request.json"
```

For the Mendeley comparison payload:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -InFile "sample_predict_request_mendeley.json"
```

Health check:

```powershell
curl "http://127.0.0.1:8000/health"
```

## Deploy to Railway

This repo is ready to deploy as a FastAPI service on Railway.

Files included for Railway:

- `requirements.txt`
- `Procfile`
- `railway.json`

The API defaults to the hourly SCAN model:

- `irrigation_model_scan_hourly.pkl`

### Railway steps

1. Push this repo to GitHub.
2. In Railway, create a new project.
3. Choose `Deploy from GitHub repo`.
4. Select this ML repo.
5. Railway should detect the Python project automatically.
6. The service start command is already configured as:

```bash
uvicorn ml_api_server:app --host 0.0.0.0 --port $PORT
```

7. Once deployed, generate a public domain in Railway.
8. Use the deployed HTTPS URL in the mobile app:

```env
EXPO_PUBLIC_IRRIGATION_API_URL=https://your-service.up.railway.app
```

### Optional Railway environment variable

If you want to override the model path explicitly in Railway:

```env
IRRIGATION_MODEL_PATH=irrigation_model_scan_hourly.pkl
```

### Verify deployment

Open:

```text
https://your-service.up.railway.app/health
```

It should return JSON including the active model path.

## For your real capstone dataset

Replace the sample rows with actual field data collected by the app or UGV:

- One row per soil reading event
- Use the measured sensor values
- Set the `recommendation` label based on expert or adviser guidance
- Keep the class names consistent across the dataset

The model will become more useful once you collect many real samples from different days, zones, and weather conditions.
