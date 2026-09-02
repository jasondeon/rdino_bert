"""Register the 3D-Speaker submodule in the active Python environment."""

from pathlib import Path
import sysconfig


project_root = Path(__file__).resolve().parents[1]
source_root = project_root / "vendor" / "3D-Speaker"
if not (source_root / "speakerlab").is_dir():
    raise FileNotFoundError(
        "3D-Speaker submodule is missing. Run: git submodule update --init --recursive"
    )

site_packages = Path(sysconfig.get_paths()["purelib"])
pth_file = site_packages / "rdino_bert_speakerlab.pth"
pth_file.write_text(f"{source_root}\n", encoding="utf-8")
print(f"Registered SpeakerLab in {pth_file}")
