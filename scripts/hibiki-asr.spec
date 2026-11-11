# Build on each target OS; every archive includes its interpreter and native runtime.
import json
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
variant = "mlx" if sys.platform == "darwin" else "cpu"
metadata_file = root / "build" / "bundle.json"
metadata_file.parent.mkdir(exist_ok=True)
metadata_file.write_text(json.dumps({"variant": variant}))
datas = collect_data_files("hibiki_asr") + copy_metadata("hibiki-asr", recursive=True)
datas += [(str(metadata_file), "hibiki_asr")]
binaries, hiddenimports = [], collect_submodules("tiktoken_ext")
packages = ["faster_whisper", "ctranslate2", "onnxruntime", "av", "tiktoken"]
if variant == "mlx":
    packages += ["mlx", "mlx_whisper"]
    datas += copy_metadata("mlx")
for package in packages:
    data, native, modules = collect_all(package)
    datas += data
    binaries += native
    hiddenimports += modules
analysis = Analysis([str(root / "scripts" / "standalone_entry.py")],
    pathex=[str(root / "src")], binaries=binaries, datas=datas,
    hiddenimports=hiddenimports, excludes=["torch", "tensorflow", "pytest", "IPython"])
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="hibiki-asr", console=True)
coll = COLLECT(exe, analysis.binaries, analysis.datas, name="hibiki-asr")
