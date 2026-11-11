# hibiki-asr

A local speech recognition engine for [Hibiki / KikoeruGo](https://github.com/Sakura-Byte/KikoeruGo).
It runs Whisper (through CTranslate2 or MLX) behind a small HTTP API with **cancellable jobs**, **automatic
CPU / NVIDIA CUDA / AMD ROCm / Apple Silicon detection**, and **model and version management**.

Run it natively on **macOS, Linux or Windows**, or in **Docker** with the CPU or CUDA image. Apple GPU acceleration
uses MLX and requires native macOS; Docker on a Mac uses the CPU. ROCm remains experimental and unverified.

If a GPU is expected but the engine ends up on the CPU, it tells you why and how to fix it, in the log, in
`hibiki-asr doctor`, and in every job result.

It knows nothing about Hibiki: any client can use the [API](#api). Hibiki uses it for two things:

| Model | Task | What it does |
|---|---|---|
| `chickenrice` | translate | Japanese audio straight to Chinese subtitles (the "海南鸡" model, 5000 h) |
| `whisper-ja` | transcribe | Japanese audio to Japanese text (`whisper-ja-1.5B`, bf16), for translation by an LLM |

Both use an ASMR-tuned voice activity detector, smart 30 s chunking and repetition cleanup, ported from
[Faster-Whisper-TransWithAI-ChickenRice](https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice)
(MIT, see [NOTICE](NOTICE)).

## Quick start

```bash
# Linux
curl -LsSf https://raw.githubusercontent.com/Sakura-Byte/hibiki-asr/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/Sakura-Byte/hibiki-asr/main/install.ps1 | iex
```

The script installs [uv](https://docs.astral.sh/uv/) if it is missing, installs the engine as a uv tool, runs
`hibiki-asr setup --variant auto` (which picks the runtime for your hardware and says why) and prints the next steps.
No sudo. Run it again any time; `HIBIKI_ASR_REF=v0.1.0` pins a tag, branch or commit.

On macOS, or by hand in any Python environment:

```bash
uv tool install "hibiki-asr[runtime] @ git+https://github.com/Sakura-Byte/hibiki-asr"
hibiki-asr setup                       # installs the runtime for this machine into the same environment
```

Then:

```bash
hibiki-asr doctor                      # what hardware was found, which device will be used, and why
hibiki-asr models sources              # can Hugging Face be reached directly? which mirror is best?
hibiki-asr models download chickenrice@v2
hibiki-asr serve                       # listens on http://127.0.0.1:8001
```

Then in Hibiki: *Admin → AI → Local engine*, endpoint `http://127.0.0.1:8001`.
On loopback no token is needed. Listening on any other address **requires** a token
(`HIBIKI_ASR_TOKEN`), and the engine refuses to start without one.

**Start it automatically:** `hibiki-asr service install` (also `uninstall`, `status`). On Linux it writes a systemd
*user* unit (`~/.config/systemd/user/hibiki-asr.service`) and enables it; run `loginctl enable-linger $USER` to keep it
running when you are logged out. On Windows it creates a Task Scheduler task that starts the engine when you log on
and does not restart it if it crashes. macOS has no service support yet: run `hibiki-asr serve`.

**Update:** `hibiki-asr update` upgrades the engine (`uv tool upgrade`, or `pip install --upgrade` from where it was
installed; a source checkout is refused), re-applies the runtime when its pinned versions changed, and tells you to
restart the engine. `hibiki-asr setup --dry-run` and `service install --dry-run` print what they would do.

## Runtime variants

| Environment | CPU | NVIDIA GPU | AMD GPU | Apple GPU |
|---|---|---|---|---|
| Native macOS (Apple Silicon, macOS 14+) | yes | — | unsupported | MLX |
| Native Linux x86_64 | yes | CUDA 12 | ROCm, experimental | — |
| Native Windows x86_64 | yes | CUDA 12 | ROCm, experimental | — |
| Docker (Linux containers) | amd64 / arm64 | amd64, host GPU access required | amd64, Linux GPU devices required; experimental | — |

These are the runtime routes; GPU hardware validation is listed below. On a Mac, Docker can call a native MLX
engine instead of running the engine inside the container.

`hibiki-asr setup` installs one pinned, hash-checked set of packages
([`provision/lockfiles`](src/hibiki_asr/provision/lockfiles), compiled by `scripts/compile_lockfiles.py`) with
`uv pip install` (or `python -m pip` when uv is missing), records the choice, and prints the `doctor` report so you see
at once whether the GPU is usable. `--variant auto` (the default) follows the hardware; an *experimental* variant is
never chosen for you and needs `--allow-experimental`.

| Variant | For | Status |
|---|---|---|
| `cpu` | any machine (macOS: only Apple Silicon on macOS 14+, the pinned onnxruntime has no other wheel) | works; installed for real on Linux with Python 3.10 to 3.13 |
| `mlx` | Apple Silicon Macs, on the GPU | the CPU runtime plus [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper); see below |
| `cuda12` | NVIDIA, driver 525+, Pascal (RTX 10 series) to Ada (RTX 40) | pins installed for real on Linux (Python 3.11, 3.12) and the CUDA libraries load; **never run on a GPU** |
| `cuda12-blackwell` | NVIDIA RTX 50 | experimental: the CTranslate2 wheel has no Blackwell kernels (only PTX for `compute_86`), so it relies on the driver's JIT |
| `cuda11` | older NVIDIA | **no pins**: faster-whisper 1.x needs ctranslate2 4.x, which on PyPI is built for CUDA 12 |
| `rocm-linux`, `rocm-win-gfx103x`, `rocm-win-gfx110x`, `rocm-win-gfx120x` | AMD RDNA2 / RDNA3 / RDNA4 on Linux or Windows x86_64 | experimental: official CT2 4.9.0 archives are pinned and SHA256-checked; needs a matching ROCm runtime; **GPU inference not verified** |
| `rocm-win-gfx101x` | AMD RDNA1 | **no pins**: the official 4.9.0 wheels do not include these gfx targets |

**Apple Silicon.** `--variant auto` picks `mlx` on an Apple Silicon Mac. The model then runs on the GPU through
MLX; the voice activity detector stays on the CPU. It reads the same CTranslate2 model files as every other variant
(`worker/ct2_weights.py`), so there is nothing else to download. Two differences from CTranslate2: there is no beam
search (`beam_size` above 1 is decoded greedily; the default is 1), and the only precisions are `float16` and
`float32`. This needs the engine to run on macOS itself: a container on a Mac is a Linux virtual machine without
access to the GPU, so Hibiki in Docker reaches a native engine at `http://host.docker.internal:8001`. Measured once,
on an M4 with 16 GB and one 3 minute recording through the current pipeline: 27.2 s on MLX 0.32.4 against 60.1 s on
the CPU (CT2 4.9.0, int8), each returning 61 segments. MLX 0.32.4 corrected two obvious translation errors seen
with MLX 0.29.3 in an earlier comparison. Output is not guaranteed to match across runtime versions or backends.

**AMD.** After installing the AMD driver and matching ROCm/HIP userspace libraries, opt in with
`hibiki-asr setup --variant auto --allow-experimental` (or explicitly choose a matching variant).
Setup installs the CPU baseline, downloads the official CT2 4.9.0 archive, checks its SHA256 and replaces only
the matching CTranslate2 wheel, even when the version is unchanged. Linux needs ROCm 7.2 libraries on the loader
path and access to `/dev/kfd` and `/dev/dri`. Windows needs `amdhip64_7.dll` and `hipblas.dll`, found through
`HIP_PATH` or the `rocm_sdk` packages. The ROCm Docker image supplies the Linux userspace libraries. AMD GPUs
on macOS are unsupported. These packages have not been tested for inference on AMD hardware.

What was checked while building this, and what was not:

* **Checked:** the pins resolve, and every wheel exists, for Linux x86_64 and Windows on Python 3.10 to 3.13; real
  installs, and switches cpu ↔ cuda12, through uv and through the pip fallback; that the CUDA libraries installed by pip
  are found only once `hibiki-asr` puts them on `LD_LIBRARY_PATH`/`PATH` (it does, for its own worker); `install.sh`
  end to end from GitHub; `update` on a uv tool; the systemd unit with `systemd-analyze verify`; the Dockerfiles with
  hadolint, the compose file with `docker compose config`, the workflows with actionlint.
* **Not checked:** any NVIDIA or AMD GPU, CUDA or ROCm (there was none); Windows (`install.ps1`, the scheduled task,
  CUDA DLL lookup); macOS other than one Apple Silicon machine (M4, macOS 26, Python 3.12).
* **Checked later:** the test suite on Windows, Linux and that Mac; CPU images on amd64 and arm64, and CUDA/ROCm
  images on amd64, including startup, authentication, doctor, runtime settings and audio decoding; the `mlx`
  variant installed from its lockfile on that Mac and a job run through
  the HTTP API, including runtime changes and disabling CPU fallback.

## Docker

```bash
export HIBIKI_ASR_TOKEN="$(openssl rand -hex 24)"
docker compose -f docker/compose.example.yml --profile cuda up -d      # or --profile cpu
docker compose -f docker/compose.example.yml exec hibiki-asr-cuda hibiki-asr models download chickenrice@v2
```

`docker/compose.example.yml` has `cpu`, `cuda` and `rocm` profiles, the required token, a named volume for `/data` and
port 8001. Images are `ghcr.io/sakura-byte/hibiki-asr:<version>-cpu` and `-cuda` (plus `latest-*`), published by the
release workflow when a `v*` tag is pushed; until then build them: `docker build -f docker/Dockerfile.cuda -t hibiki-asr:cuda .`.
Each image runs the same `hibiki-asr setup --variant <id> --yes` as a bare-metal install, as a non-root user, with
models in `/data`. An NVIDIA image needs the NVIDIA Container Toolkit on the host (`doctor` says so when the GPU was
not passed in).

The experimental `rocm` profile uses `-rocm` / `latest-rocm`, with the required device mappings in the compose
file. Docker GPU access depends on the host: Linux can pass NVIDIA/AMD devices through; Windows needs a
GPU-capable WSL2/Docker setup and compatible drivers. Docker on macOS has no access to the Apple GPU; use a native
MLX engine. Building an image does not validate GPU inference on that hardware.

## Choosing a download source (Hugging Face mirror)

Models come from Hugging Face, which is slow or blocked in some regions.

```bash
hibiki-asr models sources                                  # tests huggingface.co and every mirror
hibiki-asr models download whisper-ja@1.5b --endpoint https://hf-mirror.com
hibiki-asr models download whisper-ja@1.5b --endpoint https://my-mirror.example --no-fallback

hibiki-asr config set hf_endpoint https://hf-mirror.com    # make a mirror the default
```

* The first endpoint that works is used; the others are fallbacks. An endpoint that keeps failing is
  skipped for the rest of the download instead of being retried for every file.
* Large files are fetched as parallel 16 MiB blocks (`download_threads`, default 4) and resume where they
  stopped after an interruption or a cancel.
* Every file is checked against the size and sha256 pinned in the catalog before it is installed.
* `HF_ENDPOINT` and `HF_TOKEN` are honoured; `HTTPS_PROXY` too.

## Models and versions

A model has one or more **versions**; several can be installed side by side and one is *active* (the one used
when a job does not name a version).

```bash
hibiki-asr models list
hibiki-asr models download chickenrice@v2      # also downloads the VAD and feature extractor it needs
hibiki-asr models use chickenrice@v2           # choose the default version
hibiki-asr models verify chickenrice@v2        # re-hash the files
hibiki-asr models delete chickenrice@v2
hibiki-asr models refresh                      # fetch a newer catalog (new model versions)
```

Model files you already have (say, the `models/` folder of the upstream project) need not be downloaded again:

```bash
hibiki-asr models import ~/upstream/models/whisper-large-v2-translate-zh-v0.2-st-ct2 --model chickenrice@v2
hibiki-asr models import ~/upstream/models/Whisper-Vad-EncDec-ASMR-onnx --model vad-asr@1     # a component
```

Every file the catalog lists for that version is checked (size and sha256) before anything is installed; one wrong file
rejects the import and installs nothing. The files are copied (`--move` takes them out of the folder instead), and a
later `models download` fetches only what is still missing.

The catalog pins every file to a Hugging Face commit and sha256. To use your own model, add it to
`catalog.local.toml` next to the config file (see `hibiki-asr config path`):

```toml
schema = 1

[[entry]]
id = "my-finetune"
kind = "model"
display_name = "My fine-tune"
task = "transcribe"            # or "translate"
source_languages = ["ja"]
output_languages = ["ja"]
requires = ["vad-asr@1", "whisper-base-fe@1"]

[[entry.versions]]
version = "1"
repo = "me/my-finetune-ct2"
revision = "main"
files = [{ path = "config.json" }, { path = "model.bin" }, { path = "tokenizer.json" }, { path = "vocabulary.json" }]
```

## Which device is used, and why not the GPU

`device = auto` (the default) uses a GPU when the runtime can see one. `hibiki-asr doctor` shows the whole picture:

```
Device   cpu (int8), VAD on cpu   <-- a GPU was expected, running on the CPU instead

Findings
  [WARNING] NO_GPU_VISIBLE: No NVIDIA or AMD GPU is visible to the engine, although a CUDA/ROCm runtime is installed.
      -> Pass the GPU into the container. NVIDIA (needs the NVIDIA Container Toolkit on the host): `gpus: all` ...
```

If loading on the GPU fails (out of memory, a driver problem) the engine retries **once** on the CPU and
records `GPU_LOAD_FAILED_FALLBACK_CPU` in the job result. Set `allow_cpu_fallback = false` to fail instead.

| Finding | Meaning |
|---|---|
| `CPU_ONLY` | No GPU was found. Expected on such a machine. |
| `NO_GPU_VISIBLE` | A GPU runtime is installed, or `device=cuda` is set, but no GPU is visible (typically a container started without `gpus: all` / `/dev/kfd`). |
| `NVIDIA_DRIVER_MISSING`, `NVIDIA_DRIVER_TOO_OLD` | Install or update the NVIDIA driver. |
| `CT2_NO_GPU_SUPPORT`, `AMD_HIP_WHEEL_MISSING` | A GPU and driver exist but the installed CTranslate2 is a CPU build. |
| `CT2_IMPORT_FAILED` | The GPU runtime is broken, usually missing cuDNN/cuBLAS. |
| `AMD_DEVICE_NOT_VISIBLE`, `AMD_IGPU_UNSUPPORTED`, `AMD_RDNA2_RUNTIME_CRASH` | AMD specifics, each with the fix. |
| `BLACKWELL_NEEDS_CU128` | RTX 50 series needs the CUDA 12.8 runtime. |
| `CUDA_DISABLED_BY_ENV` | `CUDA_VISIBLE_DEVICES=-1` (or empty) hides the GPU. |
| `VRAM_LOW`, `COMPUTE_TYPE_DOWNGRADED`, `VAD_ON_CPU` | Informational. |
| `GPU_LOAD_FAILED_FALLBACK_CPU` | Loading on the GPU failed; the CPU was used. |

## Settings

`~/.config/hibiki-asr/hibiki-asr.toml` (Windows: `%APPDATA%\hibiki-asr`), overridden by `HIBIKI_ASR_*`
environment variables (`HIBIKI_ASR_VAD__THRESHOLD=0.4` for nested values), overridden by command line flags.
`hibiki-asr config show`, `config set KEY VALUE`, `config path`.

| Key | Default | |
|---|---|---|
| `host`, `port` | `127.0.0.1`, `8001` | |
| `token` | none | Bearer token; **required** off loopback |
| `device` | `auto` | `auto`, `cpu`, `cuda`, `amd`, `mlx` (ROCm is `cuda` to CTranslate2 too) |
| `compute_type` | `auto` | e.g. `float16`, `int8_float16`, `int8` |
| `allow_cpu_fallback` | `true` | retry once on the CPU if the GPU cannot load the model |
| `idle_unload_seconds` | `600` | free the model and its VRAM when idle (0 = never) |
| `cancel_grace_seconds` | `10` | after a cancel, how long before the worker is killed |
| `download_threads` | `4` | parallel connections per large file |
| `hf_endpoint`, `hf_mirrors` | huggingface.co, hf-mirror.com | download sources, tried in order |
| `data_dir`, `models_dir` | per OS | where models live |
| `vad.*`, `merge.*`, `chunk_target_s`, `generation.*` | upstream defaults | tuning |

Runtime defaults can also be read and changed through the authenticated API, in native and Docker deployments:

```bash
# Run the same hardware/runtime checks as doctor, without using the diagnostic cache
curl -H "Authorization: Bearer $HIBIKI_ASR_TOKEN" 'http://127.0.0.1:8001/v1/diagnostics?refresh=true'

curl -H "Authorization: Bearer $HIBIKI_ASR_TOKEN" http://127.0.0.1:8001/v1/settings
curl -X PATCH -H "Authorization: Bearer $HIBIKI_ASR_TOKEN" -H 'Content-Type: application/json' \
  -d '{"device":"mlx","allow_cpu_fallback":false,"vad":{"threshold":0.4},"generation":{"beam_size":1}}' \
  http://127.0.0.1:8001/v1/settings
```

Only supplied fields change. `vad` and `merge` fields are merged; `generation` replaces the whole map (`{}` resets
it to engine defaults). Changes apply to future submissions; running and queued jobs keep their original settings.
They last until the service restarts; use the config file or environment for persistent defaults. Host, port, tokens,
paths and package installation are not writable through this endpoint. With `allow_cpu_fallback=false`, an expected
but unavailable GPU or a GPU load failure fails the job instead of running it on the CPU. An explicitly selected
CPU, or `auto` on a machine with no expected GPU, still runs on the CPU.

## Cancelling

`DELETE /v1/jobs/{id}` asks the job to stop; it notices between VAD windows and between decoded segments.
If it has not stopped after `cancel_grace_seconds`, the inference process is killed and a fresh one starts for
the next job (which has to load the model again). A cancel is never ignored.

## API

The contract is [`openapi.json`](openapi.json) (also served at `/openapi.json` and `/docs`). It only ever grows
within an `api_version`; `GET /v1/version` lists `capabilities` clients can rely on.

| | |
|---|---|
| `GET /v1/healthz` | up? (no token) |
| `GET /v1/version`, `GET /v1/diagnostics` | version and capabilities; hardware, device in use, findings |
| `GET /v1/settings`, `PATCH /v1/settings` | read or change runtime defaults, including `allow_cpu_fallback` |
| `GET /v1/models`, `POST /v1/models/{id}/versions/{v}/download`, `PUT /v1/models/{id}/active`, ... | list, install (with an optional `endpoint`), switch, verify, delete |
| `GET /v1/models/sources`, `POST /v1/models/sources/probe` | is Hugging Face reachable directly, which mirror is best, test another one |
| `POST /v1/jobs` (multipart: `params` + `audio`) | submit audio |
| `GET /v1/jobs/{id}`, `DELETE /v1/jobs/{id}` | progress and result; cancel |

## Development

```bash
uv venv && uv pip install -e ".[runtime,dev]"
.venv/bin/pytest                          # ~500 tests; MLX tests skip without MLX, no model downloads or installs
.venv/bin/ruff check . && .venv/bin/mypy
python scripts/export_openapi.py          # after changing the API; commit openapi.json
python scripts/build_catalog.py           # after adding a model version; pins revision + sha256 from Hugging Face
python scripts/compile_lockfiles.py       # after changing requirements/<variant>.in; needs uv and network access
python scripts/compile_lockfiles.py --upgrade  # refresh all pins from PyPI to the newest compatible versions
```

Adding a runtime variant is a row in `provision/variants.toml`, a `requirements/<id>.in` and its lockfile (ROCm shares
the CPU baseline and adds a pinned official wheel archive). CI (ruff,
mypy, pytest on Linux, Windows and macOS, `openapi.json` up to date, the Docker images) is in `.github/workflows`.

## License

MIT. No model weights are distributed; see [NOTICE](NOTICE) for the licenses of the models it can download.
