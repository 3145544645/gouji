@echo off
REM Teach a new pattern without fighting shell quoting.
REM
REM The reply is read from a file, so JSON quotes survive intact. Passing the
REM reply directly on the command line lost the quotation marks and stored a
REM truncated tool call -- a silent corruption that only showed up when the
REM tool was actually invoked.
REM
REM usage:  teach.bat "play some jazz" reply.txt

setlocal
cd /d "%~dp0\.."
if "%~1"=="" ( echo usage: teach.bat "PROMPT" REPLYFILE & exit /b 1 )
build\canticle.exe am-learn --teach-file "%~1" "%~2"
endlocal
