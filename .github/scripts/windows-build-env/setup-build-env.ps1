<#
.SYNOPSIS
  Checks the tools the Windows ROCm PyTorch build needs and, unless -CheckOnly is
  given, installs the ones that are missing.

.DESCRIPTION
  Intended for two uses:
    1. Run once when the Windows VM image (template) is built.
    2. Run at the start of a workflow as a conditional step: if every tool is
       already there it changes nothing.

  Tools (see Windows_Build_Environment_Requirements.md for why each is needed):
    - Git for Windows (provides git and bash; the workflows use bash as default shell)
    - Python 3.12 (optional: the workflow also gets Python from actions/setup-python)
    - Visual Studio 2022 Build Tools with the C++ workload (MSVC, Windows SDK, CMake)
    - CMake (the one bundled with the Visual Studio C++ workload is enough)
    - Windows long paths (recommended)
  ninja is not installed here: the build installs it from PyPI.

  Without -CheckOnly the script needs an administrator session to install. It never
  reboots the machine.

  Exit code: 0 when every required tool is present at the end, 1 otherwise.

.PARAMETER CheckOnly
  Only report. Install nothing and change no PATH or registry setting.

.PARAMETER SkipPython
  Do not require or install Python (use when the workflow runs actions/setup-python).
#>
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$SkipPython,
    [string]$GitInstallerUrl = 'https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe',
    [string]$PythonVersion = '3.12.10',
    [string]$VsBuildToolsUrl = 'https://aka.ms/vs/17/release/vs_buildtools.exe',
    [string]$WorkDir = (Join-Path ([System.IO.Path]::GetTempPath()) 'build-env-setup')
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$script:Results = New-Object System.Collections.ArrayList
$script:IsAdmin = $false

function Write-Section([string]$Text) {
    Write-Host ''
    Write-Host "== $Text =="
}

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-DiskGB {
    $c = Get-PSDrive -Name C
    return ('C: used {0} GB, free {1} GB' -f [math]::Round($c.Used / 1GB, 1), [math]::Round($c.Free / 1GB, 1))
}

function Add-Result([string]$Tool, [string]$Status, [string]$Detail, [bool]$Required = $true) {
    [void]$script:Results.Add([pscustomobject]@{
            Tool     = $Tool
            Status   = $Status
            Required = $Required
            Detail   = $Detail
        })
    Write-Host ('{0,-14} {1,-12} {2}' -f $Tool, $Status, $Detail)
}

function Find-Tool([string]$Name) {
    $c = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return $c.Source }
    return $null
}

# Put a folder on PATH for this process, for the following steps of a GitHub job
# (GITHUB_PATH), and, in an administrator session, for the whole machine.
function Add-ToPath([string]$Dir) {
    if (-not (Test-Path $Dir)) { return }
    if (($env:PATH -split ';') -notcontains $Dir) { $env:PATH = "$Dir;$env:PATH" }
    if ($env:GITHUB_PATH) { Add-Content -Path $env:GITHUB_PATH -Value $Dir }
    if ($script:IsAdmin) {
        $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
        if (($machine -split ';') -notcontains $Dir) {
            [Environment]::SetEnvironmentVariable('Path', "$Dir;$machine", 'Machine')
        }
    }
}

function Get-File([string]$Url, [string]$OutFile) {
    New-Item -ItemType Directory -Force -Path (Split-Path $OutFile) | Out-Null
    for ($i = 1; $i -le 3; $i++) {
        try {
            Write-Host "Downloading $Url (attempt $i)"
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $OutFile
            if ((Get-Item $OutFile).Length -gt 0) { return }
        }
        catch {
            Write-Host "  failed: $($_.Exception.Message)"
            Start-Sleep -Seconds (5 * $i)
        }
    }
    throw "Could not download $Url"
}

function Invoke-Installer([string]$File, [string[]]$Arguments, [int[]]$OkExitCodes = @(0)) {
    Write-Host "Running: $File $($Arguments -join ' ')"
    $p = Start-Process -FilePath $File -ArgumentList $Arguments -Wait -PassThru
    if ($OkExitCodes -notcontains $p.ExitCode) {
        throw "$File exited with code $($p.ExitCode)"
    }
    if ($p.ExitCode -eq 3010) { Write-Host '  (installer asks for a reboot; not rebooting)' }
}

# ---------------------------------------------------------------- Git for Windows
function Ensure-Git {
    $gitDir = Join-Path $env:ProgramFiles 'Git\bin'
    $gitBash = Join-Path $gitDir 'bash.exe'

    if (-not (Test-Path $gitBash)) {
        if ($CheckOnly) {
            Add-Result 'git+bash' 'MISSING' 'Git for Windows not installed'
            return
        }
        $installer = Join-Path $WorkDir 'Git-installer.exe'
        Get-File $GitInstallerUrl $installer
        Invoke-Installer $installer @('/VERYSILENT', '/NORESTART', '/NOCANCEL', '/SP-', '/SUPPRESSMSGBOXES')
    }

    if (-not (Test-Path $gitBash)) {
        Add-Result 'git+bash' 'MISSING' "expected $gitBash"
        return
    }
    if (-not $CheckOnly) { Add-ToPath $gitDir }

    # `bash` on PATH may be the WSL launcher in System32; the build needs Git's bash.
    $onPath = Find-Tool 'bash'
    $version = (& $gitBash --version | Select-Object -First 1)
    if ($CheckOnly -and $onPath -and ($onPath -ne $gitBash)) {
        Add-Result 'git+bash' 'WARN' "Git bash exists but PATH resolves bash to $onPath ($version)"
    }
    else {
        Add-Result 'git+bash' 'OK' "$gitBash ($version)"
    }
}

# ---------------------------------------------------------------- Python
function Ensure-Python {
    if ($SkipPython) {
        Add-Result 'python' 'SKIPPED' 'workflow uses actions/setup-python' $false
        return
    }
    $parts = $PythonVersion.Split('.')
    $dirName = 'Python' + $parts[0] + $parts[1]
    $pyDir = Join-Path $env:ProgramFiles $dirName

    $py = Find-Tool 'python'
    # The Microsoft Store alias in WindowsApps is a stub, not a real Python.
    if ($py -and ($py -like '*\WindowsApps\*')) { $py = $null }

    if (-not $py -and (Test-Path (Join-Path $pyDir 'python.exe'))) {
        if (-not $CheckOnly) { Add-ToPath $pyDir; Add-ToPath (Join-Path $pyDir 'Scripts') }
        $py = Join-Path $pyDir 'python.exe'
    }
    if (-not $py) {
        if ($CheckOnly) {
            Add-Result 'python' 'MISSING' "Python $PythonVersion not installed"
            return
        }
        $installer = Join-Path $WorkDir 'python-installer.exe'
        Get-File "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe" $installer
        Invoke-Installer $installer @('/quiet', 'InstallAllUsers=1', 'PrependPath=1', 'Include_test=0', 'Include_launcher=1')
        Add-ToPath $pyDir
        Add-ToPath (Join-Path $pyDir 'Scripts')
        $py = Join-Path $pyDir 'python.exe'
    }
    if (-not (Test-Path $py)) {
        Add-Result 'python' 'MISSING' "expected $py"
        return
    }
    $version = (& $py --version 2>&1 | Select-Object -First 1)
    $want = $parts[0] + '.' + $parts[1]
    if ("$version" -like "*$want.*") {
        Add-Result 'python' 'OK' "$py ($version)"
    }
    else {
        Add-Result 'python' 'WARN' "$py is $version, expected $want" $false
    }
}

# ---------------------------------------------------------------- Visual Studio Build Tools
function Get-VsInstallPath {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $null }
    $path = & $vswhere -products '*' -version '[17.0,18.0)' -requires 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' -property installationPath | Select-Object -First 1
    if ($path) { return $path }
    return $null
}

function Ensure-VisualStudio {
    $vs = Get-VsInstallPath
    if (-not $vs) {
        if ($CheckOnly) {
            Add-Result 'vs2022-cpp' 'MISSING' 'Visual Studio 2022 C++ build tools not installed'
            return
        }
        Write-Host "Free disk before the Visual Studio install: $(Get-DiskGB)"
        $installer = Join-Path $WorkDir 'vs_buildtools.exe'
        Get-File $VsBuildToolsUrl $installer
        # 3010 = success, reboot recommended.
        Invoke-Installer $installer @('--quiet', '--wait', '--norestart', '--nocache',
            '--add', 'Microsoft.VisualStudio.Workload.VCTools', '--includeRecommended') @(0, 3010)
        $vs = Get-VsInstallPath
    }
    if (-not $vs) {
        Add-Result 'vs2022-cpp' 'MISSING' 'installer finished but vswhere finds no C++ tools'
        return
    }
    $toolsets = @()
    $msvcDir = Join-Path $vs 'VC\Tools\MSVC'
    if (Test-Path $msvcDir) { $toolsets = (Get-ChildItem $msvcDir -Directory | Select-Object -ExpandProperty Name) }
    Add-Result 'vs2022-cpp' 'OK' "$vs (MSVC toolsets: $($toolsets -join ', '))"
}

# ---------------------------------------------------------------- CMake
function Ensure-CMake {
    $cmake = Find-Tool 'cmake'
    if (-not $cmake) {
        $vs = Get-VsInstallPath
        if ($vs) {
            $dir = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
            if (Test-Path (Join-Path $dir 'cmake.exe')) {
                if (-not $CheckOnly) { Add-ToPath $dir }
                $cmake = Join-Path $dir 'cmake.exe'
            }
        }
    }
    if (-not $cmake) {
        Add-Result 'cmake' 'MISSING' 'not on PATH and not in the Visual Studio install'
        return
    }
    $version = (& $cmake --version | Select-Object -First 1)
    Add-Result 'cmake' 'OK' "$cmake ($version)"
}

# ---------------------------------------------------------------- Long paths
function Ensure-LongPaths {
    $key = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem'
    $value = (Get-ItemProperty -Path $key -Name 'LongPathsEnabled' -ErrorAction SilentlyContinue).LongPathsEnabled
    if ($value -eq 1) {
        Add-Result 'long-paths' 'OK' 'LongPathsEnabled=1' $false
    }
    elseif ($CheckOnly) {
        Add-Result 'long-paths' 'MISSING' 'LongPathsEnabled is not 1' $false
    }
    elseif ($script:IsAdmin) {
        Set-ItemProperty -Path $key -Name 'LongPathsEnabled' -Value 1
        Add-Result 'long-paths' 'OK' 'set LongPathsEnabled=1' $false
    }
    else {
        Add-Result 'long-paths' 'WARN' 'needs an administrator session to enable' $false
    }
}

# ---------------------------------------------------------------- main
$script:IsAdmin = Test-IsAdmin
Write-Section 'Windows build environment'
Write-Host "Mode:    $(if ($CheckOnly) { 'check only' } else { 'check and install missing tools' })"
Write-Host "Admin:   $($script:IsAdmin)"
Write-Host "Machine: $env:COMPUTERNAME"
Write-Host "Disk:    $(Get-DiskGB)"

if (-not $CheckOnly -and -not $script:IsAdmin) {
    Write-Host 'WARNING: not an administrator session; installs may fail.'
}

Write-Section 'Tools'
foreach ($step in @('Ensure-Git', 'Ensure-Python', 'Ensure-VisualStudio', 'Ensure-CMake', 'Ensure-LongPaths')) {
    try { & $step }
    catch {
        Add-Result ($step -replace '^Ensure-', '') 'ERROR' $_.Exception.Message
    }
}

Write-Section 'Summary'
Write-Host "Disk: $(Get-DiskGB)"
$bad = @($script:Results | Where-Object { $_.Required -and ($_.Status -ne 'OK') -and ($_.Status -ne 'SKIPPED') })
if ($bad.Count -eq 0) {
    Write-Host 'All required tools are present.'
    exit 0
}
Write-Host ('Missing or failed: ' + (($bad | ForEach-Object { $_.Tool }) -join ', '))
exit 1
