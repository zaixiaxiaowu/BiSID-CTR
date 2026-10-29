param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("gift", "game", "music")]
    [string]$Domain,
    [switch]$SkipLlm,
    [switch]$HashEmbed
)

$root = Split-Path $PSScriptRoot -Parent
$dataDir = Join-Path $root "data\$Domain"
Set-Location $dataDir
python process.py

Set-Location $root
$profileArgs = @("data/build_profiles.py", "--data_dir", $dataDir)
if ($SkipLlm) { $profileArgs += "--skip_llm" }
python @profileArgs

$sidArgs = @("data/build_sids.py", "--data_dir", $dataDir)
if ($HashEmbed) { $sidArgs += "--hash_embed" }
python @sidArgs

python data/prepare_bisid.py --data_dir $dataDir
