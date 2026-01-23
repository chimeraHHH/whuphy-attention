param(
  [int]$SleepSeconds = 30,
  [string]$Python = "python",
  [string]$TrainArgs = "--restart_each_epoch"
)

$trainScript = "train.py"
Write-Host "Starting training runner..."

while ($true) {
  Write-Host "-----------------------------------------------"
  Write-Host ("Time: " + (Get-Date).ToString("HH:mm:ss"))
  Write-Host ("Running: {0} {1} {2}" -f $Python, $trainScript, $TrainArgs)

  & $Python $trainScript $TrainArgs
  $exitCode = $LASTEXITCODE

  if ($exitCode -eq 0) {
    Write-Host "Stage finished."
  }
  else {
    Write-Host ("Process exited with code: " + $exitCode)
    if ($exitCode -eq 130) {
      Write-Host "Interrupted, stopping runner."
      break
    }
  }

  Write-Host ("Sleeping {0} seconds..." -f $SleepSeconds)
  Start-Sleep -Seconds $SleepSeconds
}

