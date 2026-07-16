[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

Push-Location $script:ProjectRoot
try {
    Write-Step "Running environment checks"
    & (Join-Path $PSScriptRoot "check.ps1") -RequireApiKey
    if ($LASTEXITCODE -ne 0) {
        throw "Environment checks failed."
    }

    $python = Get-ProjectPython
    if (-not $python) {
        throw ".venv is missing. Run .\scripts\init.ps1 first."
    }

    Write-Step "Running end-to-end acceptance cases"
    & $python (Join-Path $PSScriptRoot "e2e_acceptance.py")
    if ($LASTEXITCODE -ne 0) {
        throw "One or more end-to-end acceptance cases failed."
    }
}
finally {
    Pop-Location
}
