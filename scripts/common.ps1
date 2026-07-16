Set-StrictMode -Version Latest

$script:ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script:DefaultEsUrl = "http://127.0.0.1:9200"

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Get-ProjectPython {
    $python = Join-Path $script:ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $python) {
        return $python
    }
    return $null
}

function Get-EnvValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string]$Default = ""
    )

    $envPath = Join-Path $script:ProjectRoot ".env"
    if (-not (Test-Path -LiteralPath $envPath)) {
        return $Default
    }

    $pattern = "^\s*" + [regex]::Escape($Name) + "\s*=\s*(.*)$"
    foreach ($line in Get-Content -LiteralPath $envPath) {
        if ($line -match $pattern) {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return $Default
}

function Get-EsUrl {
    $url = Get-EnvValue -Name "ES_URL" -Default $script:DefaultEsUrl
    if ([string]::IsNullOrWhiteSpace($url)) {
        $url = $script:DefaultEsUrl
    }
    return $url.TrimEnd('/')
}

function Test-DeepSeekApiKey {
    $value = Get-EnvValue -Name "DEEPSEEK_API_KEY"
    return (-not [string]::IsNullOrWhiteSpace($value)) -and
        ($value -ne "your_deepseek_api_key_here")
}

function Wait-Elasticsearch {
    param(
        [string]$Url = $script:DefaultEsUrl,
        [int]$TimeoutSeconds = 60
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        try {
            $health = Invoke-RestMethod -Uri "$($Url.TrimEnd('/'))/_cluster/health" -TimeoutSec 5
            if ($health.status -in @("green", "yellow")) {
                return $true
            }
        }
        catch {
            # Elasticsearch may still be starting.
        }
        Start-Sleep -Seconds 2
    } while ([DateTime]::UtcNow -lt $deadline)

    return $false
}

function Get-ExpectedChunkCount {
    $processedDir = $null
    foreach ($directory in Get-ChildItem -LiteralPath $script:ProjectRoot -Directory) {
        $candidate = Join-Path $directory.FullName "processed"
        if (Test-Path -LiteralPath (Join-Path $candidate "product_knowledge.csv")) {
            $processedDir = $candidate
            break
        }
    }
    if (-not $processedDir) {
        throw "Processed knowledge directory was not found."
    }

    $sources = @(
        @{ Path = (Join-Path $script:ProjectRoot "chunk\qa_topic_chunks.jsonl"); HasHeader = $false },
        @{ Path = (Join-Path $processedDir "product_knowledge.csv"); HasHeader = $true },
        @{ Path = (Join-Path $processedDir "faq_knowledge.csv"); HasHeader = $true },
        @{ Path = (Join-Path $processedDir "policy_knowledge.csv"); HasHeader = $true },
        @{ Path = (Join-Path $processedDir "shipping_rules.csv"); HasHeader = $true }
    )

    $total = 0
    foreach ($source in $sources) {
        $path = $source.Path
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Knowledge file not found: $path"
        }
        if ($source.HasHeader) {
            $count = @(Import-Csv -LiteralPath $path).Count
        }
        else {
            $count = 0
            foreach ($line in [System.IO.File]::ReadLines($path)) {
                if (-not [string]::IsNullOrWhiteSpace($line)) {
                    $count++
                }
            }
        }
        $total += $count
    }
    return $total
}

function Get-IndexDocumentCount {
    param(
        [string]$Url = $script:DefaultEsUrl,
        [string]$Index = "qa_topic_chunks"
    )

    try {
        $response = Invoke-RestMethod -Uri "$($Url.TrimEnd('/'))/$Index/_count" -TimeoutSec 10
        return [int]$response.count
    }
    catch {
        return $null
    }
}
