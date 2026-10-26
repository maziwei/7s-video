# 榜单视频生成器 - 启动本地桥（LLM / 媒体库随机背景与 BGM / 飞书）
# 自动探测 ComfyUI Desktop 环境；未运行时顺带拉起 ComfyUI；全程只监听本机
$ErrorActionPreference = "SilentlyContinue"
$dir = $PSScriptRoot

function Test-Port($p) { [bool](Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue) }

# ---- 探测 ComfyUI venv python（通配符路径，避开中文目录名编码问题） ----
$py = $null
$comfyDir = $null
$installs = Join-Path $env:LOCALAPPDATA "Comfy-Desktop\ComfyUI-Installs"
if (Test-Path $installs) {
  $cand = Get-ChildItem -Path $installs -Recurse -Depth 5 -Filter python.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match '\\\.venv\\Scripts\\python\.exe$' } |
    Select-Object -First 1
  if ($cand) {
    $py = $cand.FullName
    # .venv\Scripts\python.exe -> 上溯三级 = ComfyUI 程序目录
    $comfyDir = (Split-Path $py -Parent | Split-Path -Parent | Split-Path -Parent)
  }
}
if (-not $py) {
  Write-Host "[ERROR] ComfyUI python not found. Install ComfyUI Desktop first."
  Write-Host "        (找不到 ComfyUI 环境；不启动本桥也能正常出片，只是不能使用媒体库随机背景/BGM、LLM 与飞书等桥能力)"
  Read-Host "Press Enter to exit"
  exit 1
}
Write-Host "python: $py"

# ---- 1. ComfyUI 未运行则拉起 ----
if (-not (Test-Port 8188)) {
  if (Test-Path (Join-Path $comfyDir "main.py")) {
    Write-Host "[1/3] Starting ComfyUI (8188)..."
    $env:PYTHONUTF8 = "1"
    Start-Process -FilePath $py -ArgumentList "`"$(Join-Path $comfyDir 'main.py')`"","--port","8188","--listen","127.0.0.1" `
      -WorkingDirectory $comfyDir -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $dir "comfyui_out.log") -RedirectStandardError (Join-Path $dir "comfyui_err.log")
    for ($i = 0; $i -lt 60; $i++) {
      Start-Sleep -Seconds 2
      if (Test-Port 8188) { break }
    }
  }
}
if (Test-Port 8188) { Write-Host "      ComfyUI OK (8188)" }
else { Write-Host "      WARN: ComfyUI 8188 not up; LLM/媒体库/飞书 may fail until ComfyUI Desktop is running." }

# ---- 2. 启动桥（8677）----
if (Test-Port 8677) {
  Write-Host "[2/3] Bridge already running: http://127.0.0.1:8677"
} else {
  Write-Host "[2/3] Starting bridge (8677)..."
  Start-Process -FilePath $py -ArgumentList "`"$(Join-Path $dir 'comfy_bridge.py')`"" `
    -WorkingDirectory $dir -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $dir "bridge_out.log") -RedirectStandardError (Join-Path $dir "bridge_err.log")
  Start-Sleep -Seconds 4
}
if (Test-Port 8677) { Write-Host "      Bridge OK: http://127.0.0.1:8677" }
else {
  Write-Host "      ERROR: bridge not listening. Check bridge_err.log"
  Read-Host "Press Enter to exit"
  exit 1
}

# ---- 3. 打开生成器 ----
Write-Host "[3/3] Opening generator..."
Start-Process (Join-Path $dir "榜单视频生成器.html")
Write-Host "Done. 首次调用本地 LLM 需加载模型约 1~2 分钟，属正常。"
