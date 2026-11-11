"""Exercise a standalone archive from a directory outside the source checkout."""
import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("binary", type=Path)
args = parser.parse_args()
binary = str(args.binary.resolve())
with tempfile.TemporaryDirectory() as directory:
    env = dict(os.environ, HIBIKI_ASR_DATA_DIR=directory, HIBIKI_ASR_TOKEN="binary-ci")
    subprocess.run([binary, "--self-test"], cwd=directory, env=env, check=True, timeout=120)
    process = subprocess.Popen([binary, "--host", "127.0.0.1", "--port", "18001", "serve"], cwd=directory, env=env)
    try:
        for _ in range(60):
            assert process.poll() is None, "standalone server exited"
            try:
                with urllib.request.urlopen("http://127.0.0.1:18001/v1/healthz", timeout=2) as response:
                    assert response.status == 200
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("standalone server did not become healthy")
        request = urllib.request.Request("http://127.0.0.1:18001/v1/diagnostics", headers={"Authorization": "Bearer binary-ci"})
        with urllib.request.urlopen(request, timeout=90) as response:
            report = json.load(response)
        assert report["runtime"]["probe_ok"], report
    finally:
        process.terminate()
        process.wait(timeout=30)
