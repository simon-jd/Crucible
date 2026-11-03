"""从 Go 官方站下载并核对锁定工具链，仅写入项目 .tools。"""

import hashlib
import json
import platform
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "tools.lock.json").read_text())["go"]
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise SystemExit("锁定包仅适用于 linux-amd64；其他平台请另行锁定官方 SHA256")
    destination = ROOT / ".tools"
    binary = destination / "go/bin/go"
    version_file = destination / "go/VERSION"
    if binary.is_file() and version_file.read_text().splitlines()[0] == lock["version"]:
        return
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix="go-download-") as temp:
        archive = Path(temp) / lock["filename"]
        with (
            urllib.request.urlopen(lock["url"], timeout=120) as response,
            archive.open("wb") as output,
        ):
            while block := response.read(1024 * 1024):
                output.write(block)
        with archive.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != lock["sha256"]:
            raise SystemExit(f"Go SHA256 不匹配: {actual}")
        if (destination / "go").exists():
            raise SystemExit("已有其他 Go 版本；请先人工移走 .tools/go 后重试")
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")
    print(f"installed {lock['version']} (SHA256 verified)")


if __name__ == "__main__":
    main()
