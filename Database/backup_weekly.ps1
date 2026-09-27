# Weekly backup (called by run_daily.bat on Saturdays, after the pipeline run):
# raw.duckdb (clinical-trial version history can't be re-downloaded) and corporate_ids.csv
# (source of truth for the company IDs) -> %USERPROFILE%\Backups\database_weekly\<name>_<date>.<ext>.
# Keeps the last $Keep copies of each file. Run by hand: powershell -File backup_weekly.ps1

param([int]$Keep = 8)

$ErrorActionPreference = 'Stop'
$data = Join-Path $PSScriptRoot 'data'
$dest = Join-Path $env:USERPROFILE 'Backups\database_weekly'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$stamp = Get-Date -Format 'yyyy-MM-dd'

foreach ($file in @('raw.duckdb', 'corporate_ids.csv')) {
    $src = Join-Path $data $file
    if (-not (Test-Path $src)) { Write-Output "backup: $file not found, skipped"; continue }
    $base = [IO.Path]::GetFileNameWithoutExtension($file)
    $ext = [IO.Path]::GetExtension($file)
    $target = Join-Path $dest "${base}_$stamp$ext"
    Copy-Item $src $target -Force
    Write-Output ("backup: {0} -> {1} ({2:N1} MB)" -f $file, $target, ((Get-Item $target).Length / 1MB))
    # keep the newest $Keep copies
    Get-ChildItem $dest -Filter "${base}_*$ext" | Sort-Object Name -Descending |
        Select-Object -Skip $Keep | ForEach-Object { Remove-Item $_.FullName; Write-Output "backup: removed old $($_.Name)" }
}
