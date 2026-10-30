param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("gift", "game", "music", "ml1m")]
    [string]$Domain,
    [int]$Gpu = 0
)

Set-Location (Join-Path $PSScriptRoot "..\ctr")
$models = @("DeepFM", "DIN", "BST", "TIN", "SIDFeat", "SIDCTR", "BiSIDCTR")
foreach ($model in $models) {
    python run_expid.py --config ./config --expid "${model}_${Domain}" --gpu $Gpu
}
