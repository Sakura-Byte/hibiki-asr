# Install hibiki-asr on Windows: uv (if it is missing), the engine as a uv tool, and the runtime that fits this
# machine's hardware. Needs no administrator rights, and is safe to run again (it reinstalls the engine and
# re-applies the runtime).
#
#   irm https://raw.githubusercontent.com/Sakura-Byte/hibiki-asr/main/install.ps1 | iex
#
# Environment:
#   HIBIKI_ASR_REF              branch, tag or commit to install (default: the repository's default branch)
#   HIBIKI_ASR_INSTALL_VARIANT  runtime variant for `hibiki-asr setup` (default: auto, which follows the hardware)
#
# Errors are thrown rather than reported with `exit`, so running this through `iex` never closes your window.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoUrl = "https://github.com/Sakura-Byte/hibiki-asr"
$UvInstallerUrl = "https://astral.sh/uv/install.ps1"
$Ref = $env:HIBIKI_ASR_REF
$Variant = if ($env:HIBIKI_ASR_INSTALL_VARIANT) { $env:HIBIKI_ASR_INSTALL_VARIANT } else { "auto" }
# These belong to this script; the engine must not see them (HIBIKI_ASR_* variables configure it).
Remove-Item Env:HIBIKI_ASR_REF, Env:HIBIKI_ASR_INSTALL_VARIANT -ErrorAction SilentlyContinue

# Windows PowerShell 5.1 does not offer TLS 1.2 by default, which the download hosts require.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# Runs a native command and returns its exit code. PowerShell does not stop when one fails, and Windows
# PowerShell 5.1 would turn what a tool writes to stderr (uv prints its progress there) into errors under "Stop".
function Invoke-Native {
    param([string]$Command, [string[]]$Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Command @Arguments | Out-Host
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command "uv")) {
    Write-Host "Installing uv (https://docs.astral.sh/uv/) ..."
    Invoke-RestMethod $UvInstallerUrl | Invoke-Expression
    # uv's installer puts it in %USERPROFILE%\.local\bin, which this session does not know about yet
    $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
    if (-not (Test-Command "uv")) {
        throw "uv was installed but is not on PATH; open a new PowerShell window and run this script again"
    }
}

# [runtime] is the CPU baseline, so the engine works even if the GPU step below fails; `setup` then swaps in the
# pinned runtime for this machine.
$Spec = "hibiki-asr[runtime] @ git+$RepoUrl"
if ($Ref) { $Spec = "$Spec@$Ref" }
$RefName = if ($Ref) { $Ref } else { "default branch" }
Write-Host "Installing hibiki-asr ($RefName) ..."
$Code = Invoke-Native "uv" @("tool", "install", "--force", $Spec)
if ($Code -ne 0) {
    throw "uv tool install failed with exit code $Code"
}

$ToolBin = (& uv tool dir --bin | Out-String).Trim()
$env:Path = "$ToolBin;$env:Path"
if (-not (Test-Command "hibiki-asr")) {
    throw "hibiki-asr was installed but is not in $ToolBin"
}

Write-Host ""
Write-Host "Choosing the runtime for this machine ..."
$Code = Invoke-Native "hibiki-asr" @("setup", "--variant", $Variant, "--yes")
if ($Code -ne 0) {
    Write-Host ""
    Write-Host "hibiki-asr is installed and runs on the CPU, but the runtime step reported a problem (see above)."
    throw "Fix it and run:  hibiki-asr setup --variant $Variant"
}

Write-Host @"

hibiki-asr is installed.

Next steps:
  hibiki-asr models sources                    # can Hugging Face be reached? which mirror is fastest?
  hibiki-asr models download chickenrice@v2    # the translate model, about 3 GB, with the VAD it needs
  hibiki-asr serve                             # listens on http://127.0.0.1:8001

Optional:
  hibiki-asr service install                   # start it automatically when you log on (Task Scheduler)
  hibiki-asr update                            # upgrade later

If ``hibiki-asr`` is not found in a new window, run:  uv tool update-shell
"@
