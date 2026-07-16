[CmdletBinding()]
param(
    [switch]$RecreateIndex,
    [switch]$SkipInstall,
    [switch]$SkipIngest
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

Push-Location $script:ProjectRoot
try {
    Write-Step "Checking Docker"
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker was not found. Install and start Docker Desktop first."
    }
    & docker version --format "Docker Client={{.Client.Version}} Server={{.Server.Version}}"
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Engine is unavailable. Start Docker Desktop first."
    }
    & docker compose version
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose is unavailable."
    }
    & docker compose -f compose.yaml config --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "compose.yaml validation failed."
    }

    Write-Step "Preparing the Python virtual environment"
    $python = Get-ProjectPython
    if (-not $python) {
        $venvPath = Join-Path $script:ProjectRoot ".venv"
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3 -m venv $venvPath
        }
        elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m venv $venvPath
        }
        else {
            throw "Python 3 was not found."
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the Python virtual environment."
        }
        $python = Get-ProjectPython
    }
    & $python --version

    if (-not $SkipInstall) {
        Write-Step "Installing pinned Python dependencies"
        & $python -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Python dependency installation failed."
        }
        & $python -m pip check
        if ($LASTEXITCODE -ne 0) {
            throw "Python dependency validation failed."
        }
    }

    $envPath = Join-Path $script:ProjectRoot ".env"
    if (-not (Test-Path -LiteralPath $envPath)) {
        Copy-Item -LiteralPath (Join-Path $script:ProjectRoot ".env.example") -Destination $envPath
        Write-Warning "Created .env from .env.example. Set DEEPSEEK_API_KEY before starting the Agent."
    }

    Write-Step "Starting Elasticsearch"
    & docker compose -f compose.yaml up -d
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start the Elasticsearch container."
    }

    $esUrl = Get-EsUrl
    if (-not (Wait-Elasticsearch -Url $esUrl -TimeoutSeconds 60)) {
        throw "Elasticsearch did not become healthy within 60 seconds. Run docker compose logs elasticsearch for details."
    }
    Write-Host "Elasticsearch is ready: $esUrl" -ForegroundColor Green

    if (-not $SkipIngest) {
        Write-Step "Checking the knowledge index"
        $indexName = Get-EnvValue -Name "ES_INDEX" -Default "customer_service_knowledge_v1"
        $expected = Get-ExpectedChunkCount
        $actual = Get-IndexDocumentCount -Url $esUrl -Index $indexName

        if ((-not $RecreateIndex) -and ($null -ne $actual) -and ($actual -eq $expected)) {
            Write-Host "Index $indexName already contains $actual documents. Skipping ingestion." -ForegroundColor Green
        }
        else {
            $ingestArgs = @(
                (Join-Path $script:ProjectRoot "es_store\es_ingest.py"),
                "--url", $esUrl,
                "--index", $indexName,
                "--recreate"
            )
            & $python @ingestArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Elasticsearch index initialization failed."
            }
        }

        $finalCount = Get-IndexDocumentCount -Url $esUrl -Index $indexName
        if ($null -eq $finalCount -or $finalCount -ne $expected) {
            throw "Index count mismatch. Expected $expected, actual $finalCount."
        }
        Write-Host "Knowledge index is valid: $finalCount documents." -ForegroundColor Green
    }

    Write-Host "`nInitialization complete. Run .\scripts\start.ps1 to start the Agent." -ForegroundColor Green
}
finally {
    Pop-Location
}
