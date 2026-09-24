param(
    [switch]$DebugBuild
)

$ErrorActionPreference = "Stop"
$nativeDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDirectory = Split-Path -Parent $nativeDirectory
$arguments = @("build", "--manifest-path", (Join-Path $nativeDirectory "Cargo.toml"))
$configuration = "debug"
if (-not $DebugBuild) {
    $arguments += "--release"
    $configuration = "release"
}

& cargo @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Cargo failed to build the native Clarabel landing solver."
}

$source = Join-Path $nativeDirectory "target\$configuration\clarabel_hover.dll"
$destination = Join-Path $projectDirectory "clarabel_hover.dll"
Copy-Item -LiteralPath $source -Destination $destination -Force
Write-Host "Built $destination"
