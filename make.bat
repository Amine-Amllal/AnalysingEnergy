@echo off
REM Windows batch file for Sphinx documentation building
REM AnalysingEnergy Documentation Builder

setlocal

set SPHINXBUILD=sphinx-build
set SOURCEDIR=docs
set BUILDDIR=docs\_build

if "%1"=="help" (
    %SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%
    goto end
)

if "%1"=="clean" (
    echo Cleaning build directory...
    if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
    echo Done.
    goto end
)

if "%1"=="html" (
    echo Building HTML documentation...
    %SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR%
    echo.
    echo Build finished. The HTML pages are in %BUILDDIR%\html
    goto end
)

if "%1"=="linkcheck" (
    echo Checking external links...
    %SPHINXBUILD% -b linkcheck %SOURCEDIR% %BUILDDIR%\linkcheck
    goto end
)

if "%1"=="install-deps" (
    echo Installing documentation dependencies...
    pip install -r docs\requirements.txt
    goto end
)

if "%1"=="" (
    echo Usage: make.bat [target]
    echo.
    echo Available targets:
    echo   help        Show Sphinx help
    echo   html        Build HTML documentation
    echo   clean       Clean build directory
    echo   linkcheck   Check external links
    echo   install-deps Install documentation dependencies
    echo.
    goto end
)

REM Default: route to Sphinx
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR%

:end
