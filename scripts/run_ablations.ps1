param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("gift", "game", "music", "ml1m")]
    [string]$Domain,
    [int]$Gpu = 0
)

Set-Location (Join-Path $PSScriptRoot "..\ctr")
$variants = @("BiSIDCTR", "woUS", "woIS", "woXA", "woTA", "AID")
foreach ($variant in $variants) {
    $expid = if ($variant -eq "BiSIDCTR") { "BiSIDCTR_$Domain" } else { "BiSIDCTR_${Domain}_$variant" }
    python run_expid.py --config ./config --expid $expid --gpu $Gpu
}
