param(
    [string]$InputFile = "C:\Users\kenji\Downloads\mendeley_3w3pf3vnd4-1\unzipped\Agricultural Irrigation Control Data\DailyAverageSensedData1.xlsx",
    [string]$Sheet = "SensedData",
    [string]$PreparedCsv = "prepared_soil_data_mendeley.csv",
    [string]$ModelPath = "irrigation_model_mendeley.pkl",
    [string]$EvaluationDir = "evaluation_outputs_mendeley",
    [double]$SoilScale = 0.01,
    [ValidateSet("binary", "three_class")]
    [string]$LabelMode = "binary"
)

$ErrorActionPreference = "Stop"

Write-Host "Preparing Mendeley daily-average data..." -ForegroundColor Cyan
python prepare_mendeley_daily_average.py `
    --input $InputFile `
    --sheet $Sheet `
    --output $PreparedCsv `
    --soil-scale $SoilScale `
    --label-mode $LabelMode

Write-Host ""
Write-Host "Training Mendeley model..." -ForegroundColor Cyan
python train_irrigation_model.py --csv $PreparedCsv --model-out $ModelPath

Write-Host ""
Write-Host "Evaluating Mendeley model..." -ForegroundColor Cyan
python evaluate_irrigation_model.py --csv $PreparedCsv --model $ModelPath --out-dir $EvaluationDir

Write-Host ""
Write-Host "Mendeley pipeline complete." -ForegroundColor Green
Write-Host "Prepared CSV: $PreparedCsv"
Write-Host "Model: $ModelPath"
Write-Host "Evaluation outputs: $EvaluationDir"
