<#
.SYNOPSIS
    Stop the demo servers that start-demo.ps1 started.

.DESCRIPTION
    Reads .demo\state.json, and for every server the launcher itself started
    checks that the recorded process id still belongs to that server (its
    command line names asgi or next) before stopping it. A server that was
    already running when the launcher ran is left alone unless -All is given,
    and no unrelated process is ever touched.

    Two servers, not three: since Step 12 the Viewer is a screen of the
    console rather than its own process, so there is nothing on :8765 to stop.

.PARAMETER All
    Also stop a chat_rag backend / console on the demo ports that this
    launcher did not start, provided its command line identifies it as ours.
.PARAMETER ProductPort
    Port to inspect with -All (default 5005).
.PARAMETER ConsolePort
    Port to inspect with -All (default 3000).

.EXAMPLE
    .\stop-demo.ps1
.EXAMPLE
    .\stop-demo.ps1 -All
#>
[CmdletBinding()]
param(
    [switch]$All,
    [int]$ProductPort = 5005,
    [int]$ConsolePort = 3000
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root = $PSScriptRoot
$StatePath = Join-Path $Root '.demo\state.json'

function Write-Line {
    param([string]$Mark, [string]$Name, [string]$Detail, [ConsoleColor]$Color = 'Gray')
    Write-Host "[" -NoNewline
    Write-Host $Mark -NoNewline -ForegroundColor $Color
    Write-Host "] $($Name.PadRight(18)) $Detail"
}
function Ok   { param($Name, $Detail) Write-Line -Mark ([char]0x2713) -Name $Name -Detail $Detail -Color Green }
function Fail { param($Name, $Detail) Write-Line -Mark ([char]0x2717) -Name $Name -Detail $Detail -Color Red }
function Info { param($Name, $Detail) Write-Line -Mark ([char]0x2022) -Name $Name -Detail $Detail -Color DarkGray }
function Warn { param($Name, $Detail) Write-Line -Mark '!' -Name $Name -Detail $Detail -Color Yellow }

# What a process's command line has to name before this script will stop it.
# The console's listener is the `next` worker npm.cmd spawned, so the pattern
# is the framework's own module rather than the npm script that started it.
$Signatures = @{ product = 'asgi'; console = 'next' }

function Get-CommandLine {
    param([int]$ProcessId)
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($proc) { return [string]$proc.CommandLine }
    return $null
}

function Stop-Tree {
    param([int]$ProcessId)
    # Children first (a launcher stub's child, should one ever exist), then the server.
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) { Stop-Tree ([int]$child.ProcessId) }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Stop-Service {
    param([string]$Name, [int]$ProcessId, [string]$Signature)
    $cmd = Get-CommandLine $ProcessId
    if (-not $cmd) {
        Info $Name "pid $ProcessId is not running"
        return
    }
    if ($cmd -notmatch [regex]::Escape($Signature)) {
        Warn $Name "pid $ProcessId is now a different program ($($cmd.Substring(0, [Math]::Min(60, $cmd.Length)))...) - left alone"
        return
    }
    Stop-Tree $ProcessId
    Start-Sleep -Milliseconds 400
    if (Get-CommandLine $ProcessId) {
        Fail $Name "pid $ProcessId did not stop"
    } else {
        Ok $Name "stopped (pid $ProcessId)"
    }
}

Write-Host ""
Write-Host "chat_rag demo: stopping" -ForegroundColor Cyan
Write-Host "-----------------------" -ForegroundColor Cyan

$handled = @{}
if (Test-Path $StatePath) {
    $state = Get-Content $StatePath -Raw | ConvertFrom-Json
    foreach ($service in @($state.services)) {
        $name = [string]$service.name
        $label = if ($name -eq 'console') { 'console' } else { 'chat_rag' }
        $handled[$name] = $true
        if ($service.started_by_launcher) {
            Stop-Service -Name $label -ProcessId ([int]$service.pid) -Signature $Signatures[$name]
        } elseif ($All) {
            Stop-Service -Name $label -ProcessId ([int]$service.pid) -Signature $Signatures[$name]
        } else {
            Info $label "left running (pid $($service.pid); it was not started by start-demo - use -All to stop it)"
        }
    }
    Remove-Item $StatePath -Force -ErrorAction SilentlyContinue
} else {
    Info 'state' "no .demo\state.json - nothing was recorded as started by start-demo"
}

if ($All) {
    foreach ($entry in @(@{ name = 'product'; port = $ProductPort; label = 'chat_rag' }, @{ name = 'console'; port = $ConsolePort; label = 'console' })) {
        if ($handled[$entry.name]) { continue }
        $conn = Get-NetTCPConnection -LocalPort $entry.port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($conn) {
            Stop-Service -Name $entry.label -ProcessId ([int]$conn.OwningProcess) -Signature $Signatures[$entry.name]
        }
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
