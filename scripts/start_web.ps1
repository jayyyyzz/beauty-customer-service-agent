[CmdletBinding()]
param(
    [switch]$SkipDockerStart,
    [switch]$SkipCheck
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

Push-Location $script:ProjectRoot
try {
    $python = Get-ProjectPython
    if (-not $python) {
        throw ".venv is missing. Run .\scripts\init.ps1 first."
    }

    if (-not $SkipDockerStart) {
        Write-Step "Starting Elasticsearch"
        & docker compose -f compose.yaml up -d
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to start the Elasticsearch container."
        }
        $esUrl = Get-EsUrl
        if (-not (Wait-Elasticsearch -Url $esUrl -TimeoutSeconds 60)) {
            throw "Elasticsearch did not become healthy within 60 seconds."
        }
    }

    if (-not $SkipCheck) {
        Write-Step "Running preflight checks"
        & (Join-Path $PSScriptRoot "check.ps1") -RequireApiKey
        if ($LASTEXITCODE -ne 0) {
            throw "Preflight checks failed."
        }
    }

    $hostAddress = Get-EnvValue -Name "WEB_HOST" -Default "127.0.0.1"
    $port = Get-EnvValue -Name "WEB_PORT" -Default "8000"
    Write-Step "Starting Web demo at http://${hostAddress}:${port}"
    & $python -m uvicorn web_app:app --host $hostAddress --port $port
    if ($LASTEXITCODE -ne 0) {
        throw "Web demo failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
