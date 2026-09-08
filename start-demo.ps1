<#
.SYNOPSIS
    Start the demo: the chat_rag backend and the Next.js console, in one go.

.DESCRIPTION
    Starts two processes, waits until each one answers, prints the addresses
    and opens the console in the default browser. Run it again and it
    recognises servers that are already up instead of starting duplicates.
    Stop everything it started with .\stop-demo.ps1.

      backend   ->  venv\Scripts\python.exe -m asgi   (uvicorn, FLASK_PORT)
      console   ->  npm run dev                       (Next.js, --port)

    There is no third process, and there is no second backend. The Viewer used
    to be one -- a server in the chunk repository on :8765, serving its own
    page and relaying this console over /api/demo -- and since Step 12 it is a
    screen of the console at /viewer, reading /api/v1 like every other screen;
    Step 13 removed the relay it used and the Flask console beside it, so the
    backend below is the one contract and nothing else.

    The browser only ever talks to the console's own origin: next.config.mjs
    rewrites /api/v1/* to the backend, so there is no CORS grant and one place
    (CHAT_RAG_API_URL) knows the backend's address. Logs go to .demo\logs\
    under this repository (git-ignored); the started process ids go to
    .demo\state.json for stop-demo.ps1. Nothing from .env is printed.

.PARAMETER ProductPort
    Port for the chat_rag backend (default 5005, the application's own default).
.PARAMETER ConsolePort
    Port for the Next.js console (default 3000, its own default).
.PARAMETER NoBrowser
    Do not open a browser when the demo is ready.
.PARAMETER NoInstall
    Do not run `npm install` even when frontend\node_modules is missing.
.PARAMETER TimeoutSeconds
    How long to wait for each server to become ready (default 180).

.EXAMPLE
    .\start-demo.ps1
.EXAMPLE
    .\start-demo.ps1 -ConsolePort 3001 -NoBrowser
#>
[CmdletBinding()]
param(
    [int]$ProductPort = 5005,
    [int]$ConsolePort = 3000,
    [switch]$NoBrowser,
    [switch]$NoInstall,
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root = $PSScriptRoot
$Frontend = Join-Path $Root 'frontend'
$DemoDir = Join-Path $Root '.demo'
$LogDir = Join-Path $DemoDir 'logs'
$StatePath = Join-Path $DemoDir 'state.json'
$ProductUrl = "http://127.0.0.1:$ProductPort"
$ConsoleUrl = "http://127.0.0.1:$ConsolePort"

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
function Resolve-ProductPython {
    foreach ($rel in @('venv\Scripts\python.exe', '.venv\Scripts\python.exe')) {
        $path = Join-Path $Root $rel
        if (Test-Path $path) { return $path }
    }
    return $null
}

function Resolve-Npm {
    # npm on Windows is npm.cmd; Start-Process needs the resolved path.
    $command = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $command) { $command = Get-Command npm -ErrorAction SilentlyContinue }
    if ($command) { return $command.Source }
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
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$Url/api/v1/health" -TimeoutSec 4 -ErrorAction Stop
        if ($response.StatusCode -eq 200) { return ($response.Content | ConvertFrom-Json) }
    } catch { }
    return $null
}

# /api/v1/health answers the resource itself: state, ready, reasons, capacity.
# `ready` is what says the backend can be sent traffic; it stays true while the
# service calls itself degraded, which is exactly when the launcher should
# still print the address rather than time out.
function Test-ProductHealth { param($Health) return ($null -ne $Health -and $Health.PSObject.Properties.Name -contains 'ready') }

# The console has no health endpoint of its own -- it is a front end. What
# "ready" means for it is that it serves the Viewer route, which is also the
# one that proves the rewrite to /api/v1 is wired.
function Test-ConsoleReady {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$Url/viewer" -TimeoutSec 6 -ErrorAction Stop
        return ($response.StatusCode -eq 200)
    } catch { }
    return $false
}

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
# python.exe and npm.cmd are stubs whose child holds the port, so the listener
# may be the recorded pid or its child.
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
    param([string]$Name, [scriptblock]$Probe, $Process, [string]$ErrLog, [string]$OutLog, [int]$Timeout)
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
        if (& $Probe) {
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

$productPython = Resolve-ProductPython
if (-not $productPython) {
    Fail 'chat_rag python' "no venv found under $Root (expected venv\Scripts\python.exe). Create it: python -m venv venv; .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}
Info 'chat_rag python' $productPython

$npm = Resolve-Npm
if (-not $npm) {
    Fail 'npm' "not on PATH. The console is a Next.js application; install Node.js 18+ and try again."
    exit 1
}
Info 'npm' $npm

if (-not (Test-Path (Join-Path $Frontend 'node_modules'))) {
    if ($NoInstall) {
        Fail 'console deps' "frontend\node_modules is missing and -NoInstall was given. Run: npm install --prefix frontend"
        exit 1
    }
    Info 'console deps' 'frontend\node_modules missing - running npm install (once)'
    $install = Start-Process -FilePath $npm -ArgumentList @('install') -WorkingDirectory $Frontend -NoNewWindow -Wait -PassThru
    if ($install.ExitCode -ne 0) {
        Fail 'console deps' "npm install failed (exit $($install.ExitCode)). Run it by hand in $Frontend"
        exit 1
    }
    Ok 'console deps' 'installed'
}

# Environment for the children. Saved and restored so the caller's session is
# left exactly as it was. Values are never echoed.
$saved = @{}
foreach ($name in @('FLASK_PORT', 'PYTHONIOENCODING', 'PYTHONUTF8', 'CHAT_RAG_API_URL')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

$state = @{ started_at = (Get-Date).ToString('s'); services = @() }
$allReady = $true

try {
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'

    # --------------------------------------------------- chat_rag backend
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
        $env:FLASK_PORT = "$ProductPort"
        $productProc = Start-Process -FilePath $productPython -ArgumentList @('-m', 'asgi') -WorkingDirectory $Root `
            -RedirectStandardOutput $pOut -RedirectStandardError $pErr -WindowStyle Hidden -PassThru
        $probe = { Test-ProductHealth (Get-Health $ProductUrl) }.GetNewClosure()
        if (Wait-Ready -Name 'chat_rag' -Probe $probe -Process $productProc -ErrLog $pErr -OutLog $pOut -Timeout $TimeoutSeconds) {
            Ok 'chat_rag' "$ProductUrl  (pid $($productProc.Id))"
            $state.services += @{ name = 'product'; pid = $productProc.Id; port = $ProductPort; url = $ProductUrl; started_by_launcher = $true; log = $pOut; err = $pErr; command = "$productPython -m asgi" }
        } else {
            $allReady = $false
            if ($productProc -and -not $productProc.HasExited) {
                $state.services += @{ name = 'product'; pid = $productProc.Id; port = $ProductPort; url = $ProductUrl; started_by_launcher = $true; log = $pOut; err = $pErr; failed = $true }
            }
        }
    }

    # ------------------------------------------------- Next.js console
    $consoleOwner = Get-PortOwner $ConsolePort
    if ($consoleOwner) {
        if (Test-ConsoleReady $ConsoleUrl) {
            $previous = Get-PreviousLaunch 'console' $consoleOwner
            if ($previous) {
                Ok 'console' "$ConsoleUrl  (already running, started by start-demo earlier, pid $($previous.pid))"
                $state.services += @{ name = 'console'; pid = [int]$previous.pid; port = $ConsolePort; url = $ConsoleUrl; started_by_launcher = $true; log = $previous.log; err = $previous.err; command = $previous.command }
            } else {
                Ok 'console' "$ConsoleUrl  (already running, pid $consoleOwner)"
                $state.services += @{ name = 'console'; pid = $consoleOwner; port = $ConsolePort; url = $ConsoleUrl; started_by_launcher = $false }
            }
        } else {
            Fail 'console' "port $ConsolePort is taken by something else: $(Describe-Process $consoleOwner). Stop it or use -ConsolePort."
            $allReady = $false
        }
    } else {
        $cOut = Join-Path $LogDir 'console.out.log'
        $cErr = Join-Path $LogDir 'console.err.log'
        # The one place that knows where the backend is; the browser never
        # learns it, because every /api/v1 call goes to the console's origin.
        $env:CHAT_RAG_API_URL = $ProductUrl
        $consoleArgs = @('run', 'dev', '--', '--port', "$ConsolePort")
        $consoleProc = Start-Process -FilePath $npm -ArgumentList $consoleArgs -WorkingDirectory $Frontend `
            -RedirectStandardOutput $cOut -RedirectStandardError $cErr -WindowStyle Hidden -PassThru
        $probe = { Test-ConsoleReady $ConsoleUrl }.GetNewClosure()
        if (Wait-Ready -Name 'console' -Probe $probe -Process $consoleProc -ErrLog $cErr -OutLog $cOut -Timeout $TimeoutSeconds) {
            Ok 'console' "$ConsoleUrl  (pid $($consoleProc.Id))"
            $state.services += @{ name = 'console'; pid = $consoleProc.Id; port = $ConsolePort; url = $ConsoleUrl; started_by_launcher = $true; log = $cOut; err = $cErr; command = "npm run dev -- --port $ConsolePort" }
        } else {
            $allReady = $false
            if ($consoleProc -and -not $consoleProc.HasExited) {
                $state.services += @{ name = 'console'; pid = $consoleProc.Id; port = $ConsolePort; url = $ConsoleUrl; started_by_launcher = $true; log = $cOut; err = $cErr; failed = $true }
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
    Write-Host "Console:" -ForegroundColor Cyan
    Write-Host "  $ConsoleUrl            knowledge bases, documents, chat, search, analysis"
    Write-Host "  $ConsoleUrl/viewer     the Viewer: Genel / Incele / Sorgu / Debug / Benchmark"
    Write-Host "Backend:" -ForegroundColor Cyan
    Write-Host "  $ProductUrl/api/v1     the contract the console speaks"
    Write-Host ""
    Write-Host "Logs: $LogDir    Stop: .\stop-demo.ps1" -ForegroundColor DarkGray
    if (-not $NoBrowser) { Start-Process $ConsoleUrl }
    exit 0
} else {
    Write-Host "Demo is NOT fully ready." -ForegroundColor Red
    Write-Host "Logs: $LogDir    Clean up what did start: .\stop-demo.ps1" -ForegroundColor DarkGray
    exit 1
}
