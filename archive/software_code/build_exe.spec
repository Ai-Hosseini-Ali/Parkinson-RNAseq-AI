# PyInstaller spec file for Parkinson AI Predictor
#
# Run with:   pyinstaller build_exe.spec
#
# This bundles the model files, gene list, and SVG assets as data files
# (they are NOT Python code, so PyInstaller won't find them automatically),
# and pulls in the PySide6 SVG plugin explicitly since it's easy to miss.

import os

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models', 'models'),           # final_*.pkl + final_genes.txt
        ('gui/assets', 'gui/assets'),   # logo_dark.svg
    ],
    hiddenimports=[
        'sklearn.utils._weight_vector',
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'PySide6.QtSvg',
        'PySide6.QtSvgWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # <- onedir mode: keep binaries out of the exe itself
    name='ParkinsonAIPredictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,         # no terminal window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='gui/assets/app_icon.ico',  # optional — remove this line if you skip the icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ParkinsonAIPredictor',   # <- this becomes the dist\ParkinsonAIPredictor\ folder
)