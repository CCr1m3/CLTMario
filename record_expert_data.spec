# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['record_expert_data.py'],
    pathex=[],
    binaries=[],
    datas=[('config.yaml', '.'), ('utils', 'utils'), ('.venv/Lib/site-packages/nes_py/lib_nes_env.cp311-win_amd64.pyd', 'nes_py'), ('.venv\\Lib\\site-packages\\gym_super_mario_bros\\_roms\\super-mario-bros.nes', 'gym_super_mario_bros\\_roms')],
    hiddenimports=['gym.envs', 'pygame.freetype', 'tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='record_expert_data',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
