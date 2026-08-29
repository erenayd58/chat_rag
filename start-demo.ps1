<#
.SYNOPSIS
    Start the demo: the chat_rag product and the chunk Viewer v2, in one go.

.DESCRIPTION
    Starts two separate servers as background processes, waits until each one
    answers its health endpoint, prints the addresses and opens the product in
    the default browser. Run it again and it recognises servers that are
    already up instead of starting duplicates. Stop everything it started with
    .\stop-demo.ps1.

      chat_rag product   ->  venv\Scripts\python.exe app.py           (Flask, FLASK_PORT)
      chunk Viewer v2    ->  py -3.11 -m amsc.viewer_server ...       (stdlib server, --port)

    The chunk repository is found next to this one (..\chunk) unless -ChunkPath
    or the CHUNK_REPO environment variable says otherwise. Logs go to
    .demo\logs\ under this repository (git-ignored); the started process ids go
    to .demo\state.json for stop-demo.ps1. Nothing from .env is printed.

.PARAMETER ChunkPath
    Path of the chunk repository (the Viewer v2 sources and artifacts).
.PARAMETER ProductPort
    Port for chat_rag (default 5005, the application's own default).
.PARAMETER ViewerPort
    Port for the Viewer v2 server (default 8765, its own default).
.PARAMETER NoBrowser
    Do not open a browser when the demo is ready.
.PARAMETER OpenViewer
    Also open the Viewer in a second tab (the product is always opened first).
.PARAMETER Lexical
    Run the Viewer's chat with BM25 only (no embedding provider needed).
.PARAMETER TimeoutSeconds
    How long to wait for each server to become ready (default 180).

.EXAMPLE
    .\start-demo.ps1
.EXAMPLE
    .\start-demo.ps1 -ChunkPath D:\work\chunk -OpenViewer
#>
[CmdletBinding()]
param(
    [string]$ChunkPath,
    [int]$ProductPort = 5005,
    [int]$ViewerPort = 8765,
    [switch]$NoBrowser,
    [switch]$OpenViewer,
    [switch]$Lexical,
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root = $PSScriptRoot
$DemoDir = Join-Path $Root '.demo'
$LogDir = Join-Path $DemoDir 'logs'
$StatePath = Join-Path $DemoDir 'state.json'
$ProductUrl = "http://127.0.0.1:$ProductPort"
$ViewerUrl = "http://127.0.0.1:$ViewerPort"

# ----------------------------------------------------------------- output
function Write-Line {
    param([string]$Mark, [string]$Name, [string]$Detail, [ConsoleColor]$Color = 'Gray')
    $label = $Name.PadRight(18)
    Write-Host "[" -NoNewline
    Write-Host $Mark -NoNewline -ForegroundColor $Color
    Write-Host "] $label $Detail"
}
function Ok   { param($Name, $Detail) Write-Line -Mark ([char]0x2713) -Name $Name -Detail $Detail -Color Green }
function Fail { param($Name, $Detail) Write-Line -Mark ([char]0x2717) -Name $Name -Detail $Detail -Color Red }
function Info { param($Name, $Detail) Write-Line -Mark ([char]0x2022) -Name $Name -Detail $Detail -Color DarkGray }
function Warn { param($Name, $Detail) Write-Line -Mark '!' -Name $Name -Detail $Detail -Color Yellow }

function Tail-Log {
    param([string]$Path, [int]$Lines = 12)
    if (Test-Path $Path) {
        $text = Get-Content $Path -Tail $Lines -ErrorAction SilentlyContinue
        if ($text) {
            Write-Host "      last lines of $(Split-Path $Path -Leaf):" -ForegroundColor DarkGray
            foreach ($line in $text) { Write-Host "      | $line" -ForegroundColor DarkGray }
        }
    }
}

# ------------------------------------------------------------- discovery
function Resolve-ChunkRepo {
    param([string]$Given)
    $candidates = @()
    if ($Given) { $candidates += $Given }
    if ($env:CHUNK_REPO) { $candidates += $env:CHUNK_REPO }
    $candidates += (Join-Path (Split-Path $Root -Parent) 'chunk')
    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate 'src\amsc\viewer_server.py')) {
            return (Resolve-Path $candidate).Path
        }
    }
    return $null
}

function Resolve-ProductPython {
    foreach ($rel in @('venv\Scripts\python.exe', '.venv\Scripts\python.exe')) {
        $path = Join-Path $Root $rel
        if (Test-Path $path) { return $path }
    }
    return $null
}

function Resolve-ViewerPython {
    param([string]$Repo)
    # The chunk package is an editable install; whichever interpreter imports
    # it *from that repository* is the right one. Try the repo's own venv
    # first, then the 3.11 launcher the project documents, then plain python.
    $attempts = @()
    foreach ($rel in @('.venv\Scripts\python.exe', 'venv\Scripts\python.exe')) {
        $path = Join-Path $Repo $rel
        if (Test-Path $path) { $attempts += ,@($path, @()) }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) { $attempts += ,@('py', @('-3.11')) }
    if (Get-Command python -ErrorAction SilentlyContinue) { $attempts += ,@('python', @()) }
    foreach ($attempt in $attempts) {
        $exe = $attempt[0]; $pre = $attempt[1]
        try {
            $probe = & $exe @pre -c "import amsc, os; print(os.path.dirname(os.path.dirname(os.path.dirname(amsc.__file__))))" 2>$null
            if ($LASTEXITCODE -eq 0 -and $probe) {
                $where = ($probe | Select-Object -Last 1).Trim()
                if ((Resolve-Path $where -ErrorAction SilentlyContinue).Path -eq $Repo) {
                    return @{ Exe = $exe; Pre = $pre }
                }
            }
        } catch { }
    }
    return $null
}

# --------------------------------------------------------------- probing
function Get-PortOwner {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) { return [int]$conn.OwningProcess }
    return $null
}

function Get-Health {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$Url/api/health" -TimeoutSec 4 -ErrorAction Stop
        if ($response.StatusCode -eq 200) { return ($response.Content | ConvertFrom-Json) }
    } catch { }
    return $null
}

function Test-ProductHealth { param($Health) return ($null -ne $Health -and $Health.PSObject.Properties.Name -contains 'llm_provider') }
function Test-ViewerHealth  { param($Health) return ($null -ne $Health -and (($Health.PSObject.Properties.Name -contains 'documents') -or ($Health.PSObject.Properties.Name -contains 'arms'))) }

function Describe-Process {
    param([int]$ProcessId)
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($proc) { return "$($proc.Name) (pid $ProcessId): $($proc.CommandLine)" }
    return "pid $ProcessId"
}

function Get-ParentPid {
    param([int]$ProcessId)
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($proc) { return [int]$proc.ParentProcessId }
    return $null
}

# A server this launcher started earlier is still "ours" on a re-run. The
# recorded pid is the process Start-Process returned; on Windows a venv's
# python.exe and the py launcher are stubs whose child holds the port, so the
# listener may be the recorded pid or its child.
$PreviousState = $null
if (Test-Path $StatePath) {
    try { $PreviousState = Get-Content $StatePath -Raw | ConvertFrom-Json } catch { $PreviousState = $null }
}
function Get-PreviousLaunch {
    param([string]$Name, [int]$ListenerPid)
    if (-not $PreviousState) { return $null }
    foreach ($service in @($PreviousState.services)) {
        if ([string]$service.name -ne $Name -or -not $service.started_by_launcher) { continue }
        $recorded = [int]$service.pid
        if ($recorded -eq $ListenerPid -or (Get-ParentPid $ListenerPid) -eq $recorded) {
            return $service
        }
    }
    return $null
}

function Wait-Ready {
    param([string]$Name, [string]$Url, [scriptblock]$Recognise, $Process, [string]$ErrLog, [string]$OutLog, [int]$Timeout)
    $deadline = (Get-Date).AddSeconds($Timeout)
    $spinner = '|/-\'
    $tick = 0
    while ((Get-Date) -lt $deadline) {
        if ($Process -and $Process.HasExited) {
            Write-Host "`r" -NoNewline
            Fail $Name "exited early (code $($Process.ExitCode)) - see $ErrLog"
            Tail-Log $ErrLog
            Tail-Log $OutLog
            return $false
        }
        $health = Get-Health $Url
        if (& $Recognise $health) {
            Write-Host "`r" -NoNewline
            return $true
        }
        $remaining = [int]($deadline - (Get-Date)).TotalSeconds
        Write-Host ("`r[{0}] {1} starting... ({2}s left)   " -f $spinner[$tick % 4], $Name.PadRight(18), $remaining) -NoNewline
        $tick++
        Start-Sleep -Milliseconds 800
    }
    Write-Host "`r" -NoNewline
    Fail $Name "not ready after ${Timeout}s - see $ErrLog"
    Tail-Log $ErrLog
    Tail-Log $OutLog
    return $false
}

# ------------------------------------------------------------------ main
Write-Host ""
Write-Host "chat_rag demo launcher" -ForegroundColor Cyan
Write-Host "----------------------" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$chunkRepo = Resolve-ChunkRepo $ChunkPath
if (-not $chunkRepo) {
    Fail 'chunk repo' "not found. Looked at -ChunkPath, CHUNK_REPO and $(Join-Path (Split-Path $Root -Parent) 'chunk')"
    exit 1
}
$viewerHtml = Join-Path $chunkRepo 'artifacts\viewer-v2\index.html'
$viewerConfig = Join-Path $chunkRepo 'configs\rag-poc.yaml'
Info 'chunk repo' $chunkRepo

$productPython = Resolve-ProductPython
if (-not $productPython) {
    Fail 'chat_rag python' "no venv found under $Root (expected venv\Scripts\python.exe). Create it: python -m venv venv; .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}
Info 'chat_rag python' $productPython

$viewerPython = Resolve-ViewerPython $chunkRepo
if (-not $viewerPython) {
    Fail 'viewer python' "no interpreter imports amsc from $chunkRepo. Install it there: py -3.11 -m pip install -e `".[benchmark]`""
    exit 1
}
$viewerPythonLabel = if ($viewerPython.Pre.Count) { "$($viewerPython.Exe) $($viewerPython.Pre -join ' ')" } else { $viewerPython.Exe }
Info 'viewer python' $viewerPythonLabel

if (-not (Test-Path $viewerHtml)) {
    Fail 'viewer page' "$viewerHtml is missing. Build it in the chunk repo (see docs/viewer-v2-poc.md: python -m amsc.viewer_v2 ...)"
    exit 1
}

# Environment for the children. Saved and restored so the caller's session
# is left exactly as it was. Values are never echoed.
$saved = @{}
foreach ($name in @('FLASK_DEBUG', 'FLASK_PORT', 'PYTHONIOENCODING', 'PYTHONUTF8', 'OPENROUTER_API_KEY', 'VIEWER_URL')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

$state = @{ started_at = (Get-Date).ToString('s'); services = @() }
$allReady = $true

try {
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'

    # The Viewer's chat reads the provider key from the environment at
    # request time (configs/rag-poc.yaml names OPENROUTER_API_KEY). chat_rag
    # keeps that key in its .env; hand it to the child process only.
    if (-not $env:OPENROUTER_API_KEY) {
        $dotenv = Join-Path $Root '.env'
        if (Test-Path $dotenv) {
            foreach ($line in Get-Content $dotenv) {
                if ($line -match '^\s*OPENROUTER_API_KEY\s*=\s*(.+?)\s*$') {
                    $env:OPENROUTER_API_KEY = $Matches[1].Trim('"', "'")
                    break
                }
            }
        }
    }
    $keyNote = if ($env:OPENROUTER_API_KEY) { 'provider key present (from environment or .env; not shown)' } else { 'no OPENROUTER_API_KEY - Viewer chat runs BM25-only, no answers' }
    Info 'provider key' $keyNote

    # ---------------------------------------------------- chunk Viewer v2
    $viewerOwner = Get-PortOwner $ViewerPort
    if ($viewerOwner) {
        if (Test-ViewerHealth (Get-Health $ViewerUrl)) {
            $previous = Get-PreviousLaunch 'viewer' $viewerOwner
            if ($previous) {
                Ok 'chunk Viewer' "$ViewerUrl  (already running, started by start-demo earlier, pid $($previous.pid))"
                $state.services += @{ name = 'viewer'; pid = [int]$previous.pid; port = $ViewerPort; url = $ViewerUrl; started_by_launcher = $true; log = $previous.log; err = $previous.err; command = $previous.command }
            } else {
                Ok 'chunk Viewer' "$ViewerUrl  (already running, pid $viewerOwner)"
                $state.services += @{ name = 'viewer'; pid = $viewerOwner; port = $ViewerPort; url = $ViewerUrl; started_by_launcher = $false }
            }
        } else {
            Fail 'chunk Viewer' "port $ViewerPort is taken by something else: $(Describe-Process $viewerOwner). Stop it or use -ViewerPort."
            $allReady = $false
        }
    } else {
        $vOut = Join-Path $LogDir 'viewer.out.log'
        $vErr = Join-Path $LogDir 'viewer.err.log'
        $viewerArgs = @() + $viewerPython.Pre + @('-m', 'amsc.viewer_server', '--viewer', $viewerHtml, '--config', $viewerConfig, '--root', $chunkRepo, '--host', '127.0.0.1', '--port', "$ViewerPort")
        if ($Lexical -or -not $env:OPENROUTER_API_KEY) { $viewerArgs += '--lexical' }
        if (-not $env:OPENROUTER_API_KEY) { $viewerArgs += '--no-answer' }
        $viewerProc = Start-Process -FilePath $viewerPython.Exe -ArgumentList $viewerArgs -WorkingDirectory $chunkRepo `
            -RedirectStandardOutput $vOut -RedirectStandardError $vErr -WindowStyle Hidden -PassThru
        if (Wait-Ready -Name 'chunk Viewer' -Url $ViewerUrl -Recognise ${function:Test-ViewerHealth} -Process $viewerProc -ErrLog $vErr -OutLog $vOut -Timeout $TimeoutSeconds) {
            Ok 'chunk Viewer' "$ViewerUrl  (pid $($viewerProc.Id))"
            $state.services += @{ name = 'viewer'; pid = $viewerProc.Id; port = $ViewerPort; url = $ViewerUrl; started_by_launcher = $true; log = $vOut; err = $vErr; command = "$viewerPythonLabel $($viewerArgs -join ' ')" }
        } else {
            $allReady = $false
            if ($viewerProc -and -not $viewerProc.HasExited) {
                $state.services += @{ name = 'viewer'; pid = $viewerProc.Id; port = $ViewerPort; url = $ViewerUrl; started_by_launcher = $true; log = $vOut; err = $vErr; failed = $true }
            }
        }
    }

    # --------------------------------------------------- chat_rag product
    $productOwner = Get-PortOwner $ProductPort
    if ($productOwner) {
        if (Test-ProductHealth (Get-Health $ProductUrl)) {
            $previous = Get-PreviousLaunch 'product' $productOwner
            if ($previous) {
                Ok 'chat_rag' "$ProductUrl  (already running, started by start-demo earlier, pid $($previous.pid))"
                $state.services += @{ name = 'product'; pid = [int]$previous.pid; port = $ProductPort; url = $ProductUrl; started_by_launcher = $true; log = $previous.log; err = $previous.err; command = $previous.command }
            } else {
                Ok 'chat_rag' "$ProductUrl  (already running, pid $productOwner)"
                $state.services += @{ name = 'product'; pid = $productOwner; port = $ProductPort; url = $ProductUrl; started_by_launcher = $false }
            }
        } else {
            Fail 'chat_rag' "port $ProductPort is taken by something else: $(Describe-Process $productOwner). Stop it or use -ProductPort."
            $allReady = $false
        }
    } else {
        $pOut = Join-Path $LogDir 'product.out.log'
        $pErr = Join-Path $LogDir 'product.err.log'
        # No reloader: it would build the pipeline twice and hide the real pid.
        $env:FLASK_DEBUG = 'false'
        $env:FLASK_PORT = "$ProductPort"
        $env:VIEWER_URL = "$ViewerUrl/"
        $productProc = Start-Process -FilePath $productPython -ArgumentList @('app.py') -WorkingDirectory $Root `
            -RedirectStandardOutput $pOut -RedirectStandardError $pErr -WindowStyle Hidden -PassThru
        if (Wait-Ready -Name 'chat_rag' -Url $ProductUrl -Recognise ${function:Test-ProductHealth} -Process $productProc -ErrLog $pErr -OutLog $pOut -Timeout $TimeoutSeconds) {
            Ok 'chat_rag' "$ProductUrl  (pid $($productProc.Id))"
            $state.services += @{ name = 'product'; pid = $productProc.Id; port = $ProductPort; url = $ProductUrl; started_by_launcher = $true; log = $pOut; err = $pErr; command = "$productPython app.py" }
        } else {
            $allReady = $false
            if ($productProc -and -not $productProc.HasExited) {
                $state.services += @{ name = 'product'; pid = $productProc.Id; port = $ProductPort; url = $ProductUrl; started_by_launcher = $true; log = $pOut; err = $pErr; failed = $true }
            }
        }
    }
}
finally {
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}

# Record what was started so stop-demo.ps1 can stop exactly that.
$state | ConvertTo-Json -Depth 4 | Set-Content -Path $StatePath -Encoding UTF8

Write-Host ""
if ($allReady) {
    Write-Host "Demo ready." -ForegroundColor Green
    Write-Host ""
    Write-Host "Product:" -ForegroundColor Cyan
    Write-Host "  $ProductUrl"
    Write-Host "Viewer:" -ForegroundColor Cyan
    Write-Host "  $ViewerUrl"
    Write-Host ""
    Write-Host "In the product, Tools > Agentic Chunking Viewer opens the Viewer in a new tab."
    Write-Host "Logs: $LogDir    Stop: .\stop-demo.ps1" -ForegroundColor DarkGray
    if (-not $NoBrowser) {
        Start-Process $ProductUrl
        if ($OpenViewer) { Start-Sleep -Milliseconds 1200; Start-Process $ViewerUrl }
    }
    exit 0
} else {
    Write-Host "Demo is NOT fully ready." -ForegroundColor Red
    Write-Host "Logs: $LogDir    Clean up what did start: .\stop-demo.ps1" -ForegroundColor DarkGray
    exit 1
}
