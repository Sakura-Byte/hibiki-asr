# hibiki-asr

A local speech recognition engine for [Hibiki / KikoeruGo](https://github.com/Sakura-Byte/KikoeruGo).
It runs Whisper (through CTranslate2) behind a small HTTP API with **cancellable jobs**, **automatic
CPU / NVIDIA CUDA / AMD ROCm detection**, and **model and version management**.

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
# CPU runtime (any machine). GPU runtimes are covered below.
uv tool install "hibiki-asr[runtime] @ git+https://github.com/Sakura-Byte/hibiki-asr"

hibiki-asr doctor                      # what hardware was found, which device will be used, and why
hibiki-asr models sources              # can Hugging Face be reached directly? which mirror is best?
hibiki-asr models download chickenrice@v2
hibiki-asr serve                       # listens on http://127.0.0.1:8001
```

Then in Hibiki: *Admin → AI → Local engine*, endpoint `http://127.0.0.1:8001`.
On loopback no token is needed. Listening on any other address **requires** a token
(`HIBIKI_ASR_TOKEN`), and the engine refuses to start without one.

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
| `device` | `auto` | `auto`, `cpu`, `cuda`, `amd` (ROCm is `cuda` to CTranslate2 too) |
| `compute_type` | `auto` | e.g. `float16`, `int8_float16`, `int8` |
| `allow_cpu_fallback` | `true` | retry once on the CPU if the GPU cannot load the model |
| `idle_unload_seconds` | `600` | free the model and its VRAM when idle (0 = never) |
| `cancel_grace_seconds` | `10` | after a cancel, how long before the worker is killed |
| `download_threads` | `4` | parallel connections per large file |
| `hf_endpoint`, `hf_mirrors` | huggingface.co, hf-mirror.com | download sources, tried in order |
| `data_dir`, `models_dir` | per OS | where models live |
| `vad.*`, `merge.*`, `chunk_target_s`, `generation.*` | upstream defaults | tuning |

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
| `GET /v1/models`, `POST /v1/models/{id}/versions/{v}/download`, `PUT /v1/models/{id}/active`, ... | list, install (with an optional `endpoint`), switch, verify, delete |
| `GET /v1/models/sources`, `POST /v1/models/sources/probe` | is Hugging Face reachable directly, which mirror is best, test another one |
| `POST /v1/jobs` (multipart: `params` + `audio`) | submit audio |
| `GET /v1/jobs/{id}`, `DELETE /v1/jobs/{id}` | progress and result; cancel |

## Development

```bash
uv venv && uv pip install -e ".[runtime,dev]"
.venv/bin/pytest                          # ~250 tests, none needs a GPU or a model
.venv/bin/ruff check . && .venv/bin/mypy
python scripts/export_openapi.py          # after changing the API; commit openapi.json
python scripts/build_catalog.py           # after adding a model version; pins revision + sha256 from Hugging Face
```

## License

MIT. No model weights are distributed; see [NOTICE](NOTICE) for the licenses of the models it can download.
