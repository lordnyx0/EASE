@echo off
title EASE Qwen3.8-27B Server [64k Context - OpenAI API]

echo ===============================================================================
echo   SERVIDOR DE INFERENCIA EASE - QWEN 3.8 27B + MTP SPECULATIVE ENGINE
echo ===============================================================================
echo   Contexto Ativo:     64k (65536 Tokens) via 4-bit Quantized KV Cache
echo   API Endpoint:       http://localhost:8000/v1
echo   Compatibilidade:    OpenAI API (/v1/chat/completions, /v1/models)
echo   Integracoes:        OpenWebUI, SillyTavern, VS Code (Continue/Cline/Roo)
echo ===============================================================================
echo.

:: Configuracoes de Ambiente
set "MAX_CONTEXT_TOKENS=65536"
set "CACHE_BITS=4"
set "HOST=0.0.0.0"
set "PORT=8000"
set "MODEL_DIR=models/Qwen3.8-27B-exl3_SC_3.00bpw_H4"
set "DEVICE=cuda:0"

:: Instrucoes para OpenWebUI
echo [INFO] Para conectar no OpenWebUI:
echo        1. Acesse Configuracoes ^> Conexoes ^> OpenAI API
echo        2. URL Base: http://localhost:8000/v1  (ou http://127.0.0.1:8000/v1)
echo        3. Chave API: sk-ease (ou qualquer texto)
echo        4. Clique em Salvar e selecione o modelo "qwen3.8-27b-ease"
echo.
echo [INICIANDO SERVIDOR FASTAPI / UVICORN...]
echo.

.venv\Scripts\python.exe serve.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O servidor encerrou com codigo de erro %ERRORLEVEL%.
    pause
)
