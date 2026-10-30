param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("gift", "game", "music", "ml1m")]
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

$bisidArgs = @("data/prepare_bisid.py", "--data_dir", $dataDir)
if ($Domain -eq "ml1m") {
    $bisidArgs += @("--user_col", "UserID", "--item_col", "MovieID", "--time_col", "Timestamp")
}
python @bisidArgs
