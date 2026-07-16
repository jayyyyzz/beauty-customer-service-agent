[CmdletBinding()]
param(
    [switch]$RequireApiKey
)

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "common.ps1")

$failures = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()

function Add-Pass {
    param([string]$Message)
    Write-Host "[OK]   $Message" -ForegroundColor Green
}

function Add-Failure {
    param([string]$Message)
    $script:failures.Add($Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

function Add-Warning {
    param([string]$Message)
    $script:warnings.Add($Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

Push-Location $script:ProjectRoot
try {
    Write-Host "Customer Service Agent environment check`n" -ForegroundColor Cyan

    foreach ($requiredFile in @("requirements.txt", "compose.yaml", ".env.example", "web_app.py", "web/index.html")) {
        if (Test-Path -LiteralPath (Join-Path $script:ProjectRoot $requiredFile)) {
            Add-Pass "$requiredFile exists"
        }
        else {
            Add-Failure "$requiredFile is missing"
        }
    }

    if (Test-Path -LiteralPath (Join-Path $script:ProjectRoot ".env")) {
        Add-Pass ".env exists"
    }
    else {
        Add-Failure ".env is missing. Run .\scripts\init.ps1 first"
    }

    $python = Get-ProjectPython
    if ($python) {
        $pythonVersion = & $python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Add-Pass "Python virtual environment is available ($pythonVersion)"
        }
        else {
            Add-Failure "Python virtual environment cannot run"
        }

        & $python -c "import openai, numpy, dotenv, fastapi, uvicorn" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Add-Pass "Core Python dependencies can be imported"
        }
        else {
            Add-Failure "Core Python dependencies are missing"
        }

        & $python -m pip check *> $null
        if ($LASTEXITCODE -eq 0) {
            Add-Pass "Python dependencies have no conflicts"
        }
        else {
            Add-Failure "Python dependency conflicts were found"
        }
    }
    else {
        Add-Failure ".venv is missing. Run .\scripts\init.ps1 first"
    }

    if (Get-Command docker -ErrorAction SilentlyContinue) {
        & docker version --format "{{.Server.Version}}" *> $null
        if ($LASTEXITCODE -eq 0) {
            Add-Pass "Docker Engine is available"
        }
        else {
            Add-Failure "Docker exists, but Docker Engine is unavailable"
        }

        & docker compose -f compose.yaml config --quiet *> $null
        if ($LASTEXITCODE -eq 0) {
            Add-Pass "compose.yaml is valid"
        }
        else {
            Add-Failure "compose.yaml validation failed"
        }
    }
    else {
        Add-Failure "Docker was not found"
    }

    $esUrl = Get-EsUrl
    if (Wait-Elasticsearch -Url $esUrl -TimeoutSeconds 5) {
        Add-Pass "Elasticsearch is healthy ($esUrl)"

        $indexName = Get-EnvValue -Name "ES_INDEX" -Default "customer_service_knowledge_v1"
        try {
            $expected = Get-ExpectedChunkCount
            $actual = Get-IndexDocumentCount -Url $esUrl -Index $indexName
            if ($null -ne $actual -and $actual -eq $expected) {
                Add-Pass "Index $indexName has the expected document count ($actual)"
            }
            else {
                Add-Failure "Index $indexName count mismatch: expected $expected, actual $actual"
            }
        }
        catch {
            Add-Failure $_.Exception.Message
        }
    }
    else {
        Add-Failure "Elasticsearch is unavailable ($esUrl)"
    }

    if (Test-DeepSeekApiKey) {
        Add-Pass "DEEPSEEK_API_KEY is configured (value hidden)"
    }
    elseif ($RequireApiKey) {
        Add-Failure "DEEPSEEK_API_KEY is missing or still uses the template value"
    }
    else {
        Add-Warning "DEEPSEEK_API_KEY is missing; Elasticsearch works, but the full Agent cannot run"
    }

    Write-Host "`nCheck complete: $($failures.Count) failure(s), $($warnings.Count) warning(s)."
    if ($failures.Count -gt 0) {
        exit 1
    }
    exit 0
}
finally {
    Pop-Location
}
