@echo off
title EASE Qwen3.8-27B Server [32k Context - Zero Spill - OpenAI API]

echo ===============================================================================
echo   SERVIDOR DE INFERENCIA EASE - QWEN 3.8 27B + MTP SPECULATIVE ENGINE
echo ===============================================================================
echo   Contexto Ativo:     32k (32768 Tokens)
echo   Precisao KV:        Q4 (4-bit Quantized KV Cache - 100%% VRAM Puro)
echo   Consumo EASE VRAM:  ~9.6 GB (Deixa >2.3 GB livres para Windows/Apps)
echo   API Endpoint:       http://localhost:8000/v1
echo   Compatibilidade:    OpenAI API (/v1/chat/completions, /v1/models)
echo   Integracoes:        Pi Agent, OpenWebUI, SillyTavern, VS Code Continue
echo ===============================================================================
echo.

:: Configuracoes de Ambiente Otimizadas para RTX 3060 (12GB)
set "MAX_CONTEXT_TOKENS=32768"
set "CACHE_BITS=4"
set "HOST=0.0.0.0"
set "PORT=8000"
set "MODEL_DIR=models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
set "DEVICE=cuda:0"

echo [INICIANDO SERVIDOR FASTAPI / UVICORN...]
echo.

.venv\Scripts\python.exe serve.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O servidor encerrou com codigo de erro %ERRORLEVEL%.
    pause
)
