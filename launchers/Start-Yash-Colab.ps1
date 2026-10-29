param([ValidateSet('Both','HotpotQA','Tau2')][string]$Environment = 'Both')
$ErrorActionPreference = 'Stop'
$yashHotpotNotebook = 'https://colab.research.google.com/drive/161BXIOvt_kKGgmOPC0Uv5dXQLaWS9Mpb'
$yashTauNotebook = 'https://colab.research.google.com/drive/1vjkuOGxrOinXnnXMzCZEeX-RxTzCcEQv?authuser=1'
if ($Environment -in @('Both','HotpotQA')) { Start-Process $yashHotpotNotebook }
if ($Environment -in @('Both','Tau2')) { Start-Process $yashTauNotebook }
Write-Output "Opened Yash's new Colab notebook(s). Use the corresponding provisioned account."
