param(
    [Parameter(Mandatory=$true)][string]$Template,
    [Parameter(Mandatory=$true)][string]$ReviewReceipt,
    [Parameter(Mandatory=$true)][string]$PilotProof,
    [Parameter(Mandatory=$true)][string]$RuntimeProbe,
    [Parameter(Mandatory=$true)][string]$CompilationDirectory,
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [switch]$Compile,
    [switch]$Start,
    [string]$ReviewedPlanSha256
)
$ErrorActionPreference = 'Stop'
if ($Compile -and $Start) { throw 'Compile for review first, then start that exact reviewed plan separately.' }
if (-not $Start -and -not $Compile) {
    [pscustomobject]@{status='prepared_only';started=$false;message='No resource check, adoption or model work. Use -Compile to prepare a timed plan for review; -Start requires its reviewed hash.'}
    return
}
$preparation = Get-Content -LiteralPath (Join-Path $Template 'preparation.json') -Raw | ConvertFrom-Json
if ($preparation.status -ne 'blocked_no_launch' -or $preparation.adopted -ne $false) {
    throw 'A prepared, unstarted template is required.'
}
$compileAbsolute = [IO.Path]::GetFullPath($CompilationDirectory)
$runAbsolute = [IO.Path]::GetFullPath($RunDirectory)
foreach ($studyPath in @($compileAbsolute, $runAbsolute)) {
    if ($studyPath.Contains('"') -or $studyPath.Contains("`n") -or $studyPath.Contains("`r")) {
        throw 'Choose output directories with ordinary path characters.'
    }
}
if ($compileAbsolute -eq $runAbsolute) { throw 'Compilation and execution need different output directories.' }
if (Test-Path -LiteralPath $runAbsolute) { throw 'Execution requires a NEW run directory.' }
if ($Compile -and (Test-Path -LiteralPath $compileAbsolute)) { throw 'Preserve old compilation receipts and choose a NEW directory.' }
$admission = $compileAbsolute + '.admission.json'
if ($Compile -and (Test-Path -LiteralPath $admission)) { throw 'Admission receipt already exists; preserve it and choose a new output name.' }
Push-Location -LiteralPath $preparation.source_root
try {
    if ($Compile) {
    # One local admission attempt. A failed observation never loops or closes apps.
    & $preparation.python -B -m haltiq_tau_hotpot.planning resources --config $preparation.template.path --output $admission
    if ($LASTEXITCODE -ne 0) { throw 'Resource admission failed; nothing was adopted.' }
    $admissionResult = Get-Content -LiteralPath $admission -Raw | ConvertFrom-Json
    if ($admissionResult.status -ne 'PASS_RESOURCE_ADMISSION') { throw 'Resource reserves did not pass; nothing was adopted.' }
    $startTime = (Get-Date).ToString('o')
    & $preparation.python -B -m haltiq_tau_hotpot.planning compile-main --template $Template --output $compileAbsolute `
        --start-time $startTime --review-receipt $ReviewReceipt --pilot-proof $PilotProof `
        --resource-receipt $admission --runtime-probe $RuntimeProbe
    if ($LASTEXITCODE -ne 0) { throw 'Plan compilation failed; nothing was adopted.' }
    $compiled = Get-Content -LiteralPath (Join-Path $compileAbsolute 'compilation.json') -Raw | ConvertFrom-Json
    if ($compiled.status -ne 'compiled_not_adopted') { throw 'The measured full route is blocked; preserve its forecast and receipts.' }
    [pscustomobject]@{status='compiled_awaiting_review';started=$false;plan=$compiled.plan.path;plan_sha256=$compiled.plan.sha256;selected_questions=$compiled.selected_n}
    return
    }
    $compiled = Get-Content -LiteralPath (Join-Path $compileAbsolute 'compilation.json') -Raw | ConvertFrom-Json
    if ($compiled.status -ne 'compiled_not_adopted' -or $ReviewedPlanSha256 -notmatch '^[a-f0-9]{64}$') {
        throw 'Review a compiled plan and supply -ReviewedPlanSha256 before starting.'
    }
    $planPath = $compiled.plan.path
    if ((Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ReviewedPlanSha256 -or $compiled.plan.sha256 -ne $ReviewedPlanSha256) {
        throw 'The plan differs from the exact reviewed hash; nothing was adopted.'
    }
    $plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    $elapsed = ((Get-Date).ToUniversalTime() - [datetime]::Parse($plan.start_time).ToUniversalTime()).TotalSeconds
    if ($elapsed -lt 0 -or $elapsed -gt 300) { throw 'Timed review window expired; preserve it and compile a new plan.' }
    $inspectionText = & $preparation.python -B -m haltiq_supervisor inspect --plan $planPath
    if ($LASTEXITCODE -ne 0) { throw 'Final plan inspection failed; nothing was adopted.' }
    $inspection = ($inspectionText -join [Environment]::NewLine) | ConvertFrom-Json
    if ($inspection.status -ne 'ready_not_started') { throw 'Final readiness is blocked; nothing was adopted.' }
    $adoptionText = & $preparation.python -B -m haltiq_supervisor prepare --plan $planPath --run-dir $runAbsolute --allow-model-calls
    if ($LASTEXITCODE -ne 0) { throw 'Source adoption failed; no supervisor was started.' }
    $adoptionText | Set-Content -LiteralPath (Join-Path $compileAbsolute 'adoption.json') -Encoding utf8
    $studyArgs = @('-B', '-m', 'haltiq_supervisor', 'run', '--run-dir', ('"' + $runAbsolute + '"'), '--allow-model-calls')
    $studyProcess = Start-Process -FilePath $plan.python -ArgumentList $studyArgs -WindowStyle Hidden -PassThru `
        -WorkingDirectory (Join-Path $runAbsolute 'source') `
        -RedirectStandardOutput (Join-Path $runAbsolute 'supervisor.stdout.log') `
        -RedirectStandardError (Join-Path $runAbsolute 'supervisor.stderr.log')
    [pscustomobject]@{status='local_study_dispatched';pid_snapshot=$studyProcess.Id;run_directory=$runAbsolute;codex_callbacks=$false}
} finally {
    Pop-Location
}
