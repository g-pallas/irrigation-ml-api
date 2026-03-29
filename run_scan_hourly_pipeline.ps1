param(
    [string[]]$InputFiles = @("C:\Users\kenji\Downloads\usdascan\hourly\2026_ALL_YEAR=2024_hourly.csv"),
    [string]$PreparedCsv = "prepared_soil_data_scan_hourly.csv",
    [string]$ModelPath = "irrigation_model_scan_hourly.pkl",
    [string]$EvaluationDir = "evaluation_outputs_scan_hourly",
    [string]$Zone = "Walnut Gulch #1, AZ",
    [ValidateSet("binary", "three_class", "duration", "quantile")]
    [string]$LabelMode = "three_class",
    [double]$DryThreshold = 6.0,
    [double]$WetThreshold = 10.0,
    [double]$InvalidBelow = -50.0,
    [Nullable[double]]$InvalidAbove = $null,
    [string]$DateColumn = "Date",
    [string]$TimeColumn = "Time",
    [string]$MoistureColumn = "SMS.I-1:-2 (pct) (loam)",
    [string]$TemperatureColumn = "STO.I-1:-2 (degC)",
    [string]$HumidityColumn = "RHUM.I-1 (pct)"
)

$ErrorActionPreference = "Stop"

Write-Host "Preparing USDA SCAN hourly data..." -ForegroundColor Cyan

$prepareArgs = @(
    "prepare_usda_scan_hourly.py",
    "--input"
)

$prepareArgs += $InputFiles
$prepareArgs += @(
    "--output", $PreparedCsv,
    "--zone", $Zone,
    "--label-mode", $LabelMode,
    "--dry-threshold", $DryThreshold,
    "--wet-threshold", $WetThreshold,
    "--invalid-below", $InvalidBelow,
    "--date-col", $DateColumn,
    "--time-col", $TimeColumn,
    "--moisture-col", $MoistureColumn,
    "--temperature-col", $TemperatureColumn,
    "--humidity-col", $HumidityColumn
)

if ($null -ne $InvalidAbove) {
    $prepareArgs += @("--invalid-above", $InvalidAbove)
}

python @prepareArgs

Write-Host ""
Write-Host "Training SCAN hourly model..." -ForegroundColor Cyan
python train_irrigation_model.py --csv $PreparedCsv --model-out $ModelPath

Write-Host ""
Write-Host "Evaluating SCAN hourly model..." -ForegroundColor Cyan
python evaluate_irrigation_model.py --csv $PreparedCsv --model $ModelPath --out-dir $EvaluationDir

Write-Host ""
Write-Host "SCAN hourly pipeline complete." -ForegroundColor Green
Write-Host "Prepared CSV: $PreparedCsv"
Write-Host "Model: $ModelPath"
Write-Host "Evaluation outputs: $EvaluationDir"
