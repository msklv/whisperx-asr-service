"""
Shared ASR pipeline stage functions.

Extracts the 3-stage WhisperX pipeline (transcribe -> align -> diarize) into
reusable functions consumed by both the legacy FastAPI endpoints and the
Ray Serve deployments.
"""

import copy
import gc
import json
import logging
import math
import os
import threading
import time
import warnings
from typing import Any, Dict, Optional, Tuple

# Suppress pyannote's torchcodec warning -- we decode audio via whisperx.load_audio (ffmpeg),
# not pyannote's built-in decoder, so the missing torchcodec is irrelevant.
warnings.filterwarnings("ignore", message=".*torchcodec.*")

import numpy as np  # noqa: E402 — must come after warning suppression
import torch  # noqa: E402
import whisperx  # noqa: E402
from whisperx.diarize import DiarizationPipeline  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at import time, same as before)
# ---------------------------------------------------------------------------
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16" if DEVICE == "cuda" else "2"))
# Device for the Wav2Vec2 alignment stage. Defaults to DEVICE; set
# ALIGN_DEVICE=cpu to keep alignment off the GPU and reduce VRAM at the cost
# of slower word timestamps (issue #32).
ALIGN_DEVICE = os.getenv("ALIGN_DEVICE", "").strip().lower() or DEVICE
HF_TOKEN = os.getenv("HF_TOKEN", None)
CACHE_DIR = os.getenv("CACHE_DIR", "/.cache")
DEFAULT_MODEL = os.getenv("PRELOAD_MODEL", "large-v3")

# Idle model eviction. Set MODEL_KEEP_ALIVE_SECONDS > 0 to unload Whisper,
# alignment and diarization models that have not been used in that many
# seconds. Floor of 30s on the sweep interval to avoid pegging a thread on
# tight loops.
MODEL_KEEP_ALIVE_SECONDS = int(os.getenv("MODEL_KEEP_ALIVE_SECONDS", "0"))
MODEL_EVICTION_INTERVAL_SECONDS = max(
    30, int(os.getenv("MODEL_EVICTION_INTERVAL_SECONDS", "60"))
)

# Diarization hyperparameter tuning (pyannote community-1).
# All unset by default -> the pipeline runs with the model's published defaults,
# so behaviour is unchanged unless you opt in.
#
#   DIARIZE_CLUSTERING_THRESHOLD: the main lever for merged/missed speakers.
#       community-1 default is 0.6. Lower it (e.g. 0.5) to split voices more
#       aggressively when distinct speakers share one label; raise it to merge
#       more (fewer phantom speakers). Useful range ~0.4-0.8.
#   DIARIZE_MIN_DURATION_OFF: non-speech gaps shorter than this (seconds) are
#       filled, MERGING the turns on either side. community-1 default is 0.0.
#       RAISE it (e.g. 0.1-0.5) to suppress over-segmentation; lowering below
#       0.0 is not possible, so it does not help recover rapid turns -- use the
#       clustering threshold for that.
#   DIARIZE_PARAM_OVERRIDES: escape hatch -- a JSON object deep-merged into the
#       pipeline's instantiated parameters, for any key the two vars above don't
#       cover. The exact schema is logged at pipeline load (see logs).
def _env_or_none(name: str) -> Optional[str]:
    """Read an env var, treating unset OR empty/whitespace as None.

    Compose forwards optional vars as `${VAR:-}`, which sets them to an empty
    string rather than leaving them unset, so a plain os.getenv() would return
    "" and downstream float() would crash. Normalize that to None here.
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


DIARIZE_CLUSTERING_THRESHOLD = _env_or_none("DIARIZE_CLUSTERING_THRESHOLD")
DIARIZE_MIN_DURATION_OFF = _env_or_none("DIARIZE_MIN_DURATION_OFF")
DIARIZE_PARAM_OVERRIDES = _env_or_none("DIARIZE_PARAM_OVERRIDES")

# When True, words/segments that fall outside every diarization turn are assigned
# the *nearest* speaker instead of being left unlabeled. Fixes "orphan" segments
# (e.g. a closing line with no speaker tag) at the cost of occasionally labeling
# a long silence. Default False preserves the prior behaviour.
DIARIZE_FILL_NEAREST = os.getenv("DIARIZE_FILL_NEAREST", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

# ASR backend selection. "whisper" (default) is the faster-whisper/CTranslate2
# path. "qwen3" transcribes with Qwen3-ASR and aligns with the Qwen3 forced
# aligner (language-agnostic, useful for code-switched audio); it ignores the
# requested Whisper model name and does not support task=translate.
ASR_BACKEND = (_env_or_none("ASR_BACKEND") or "whisper").lower()

# When True, segments are rebuilt at speaker-change boundaries after
# diarization, so rapid turns are not merged into one speaker's segment.
# Always applied on the qwen3 backend (its pre-diarization segments come from
# silence gaps alone); opt-in for the whisper backend because it changes the
# segment shape existing users are accustomed to.
RESEGMENT_BY_SPEAKER = (
    os.getenv("RESEGMENT_BY_SPEAKER", "false").strip().lower()
    in ("1", "true", "yes", "on")
)


def get_canonical_models() -> list:
    """
    Canonical model names accepted by the underlying faster-whisper engine.

    Sourced from faster_whisper.available_models() so this list stays in sync
    with whatever version of faster-whisper is installed, instead of being
    hardcoded here.
    """
    try:
        from faster_whisper import available_models
        return list(available_models())
    except Exception:
        # Defensive fallback if the import surface ever changes upstream.
        return [
            "tiny.en", "tiny", "base.en", "base", "small.en", "small",
            "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large",
            "distil-large-v2", "distil-medium.en", "distil-small.en",
            "distil-large-v3", "distil-large-v3.5", "large-v3-turbo", "turbo",
        ]


# OpenAI-style aliases → canonical faster-whisper names. These are kept for
# backwards compatibility on the request path; new clients should use the
# canonical names returned by /v1/models.
_MODEL_ALIASES = {
    "whisper-1": os.getenv("OPENAI_WHISPER1_MODEL", DEFAULT_MODEL),
    "whisper-large-v3": "large-v3",
    "whisper-large-v2": "large-v2",
    "whisper-medium": "medium",
    "whisper-small": "small",
    "whisper-base": "base",
    "whisper-tiny": "tiny",
}


def resolve_model_name(model: str) -> str:
    """
    Resolve a user-supplied model identifier to a canonical faster-whisper name.

    Accepts canonical names (tiny, large-v3, distil-medium.en, ...) as-is and
    maps OpenAI-style aliases (whisper-tiny, whisper-large-v3, ...) to their
    canonical equivalents. Unknown values are returned unchanged so the engine
    can produce its own validation error.
    """
    if not model:
        return DEFAULT_MODEL
    canonical = set(get_canonical_models())
    if model in canonical:
        return model
    if model in _MODEL_ALIASES:
        return _MODEL_ALIASES[model]
    if model.startswith("whisper-"):
        stripped = model[len("whisper-"):]
        if stripped in canonical:
            return stripped
    return model


_model_load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Model caches
# ---------------------------------------------------------------------------
_whisper_models: Dict[str, Any] = {}
_whisper_models_last_used: Dict[str, float] = {}
_align_models: Dict[str, Tuple[Any, Any]] = {}
_align_models_last_used: Dict[str, float] = {}
_diarize_pipeline: Optional[DiarizationPipeline] = None
_diarize_last_used: Optional[float] = None

_eviction_thread_lock = threading.Lock()
_eviction_thread_started = False


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def clear_gpu_memory():
    """Clear GPU memory cache to prevent VRAM buildup."""
    if DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")


# ---------------------------------------------------------------------------
# Stage 0 -- model loading
# ---------------------------------------------------------------------------
def load_whisper_model(model_name: str):
    """Load WhisperX model with caching (thread-safe)."""
    if model_name not in _whisper_models:
        with _model_load_lock:
            if model_name not in _whisper_models:
                logger.info(f"Loading WhisperX model: {model_name}")
                model = whisperx.load_model(
                    model_name,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                    download_root=CACHE_DIR,
                )
                _whisper_models[model_name] = model
                logger.info(f"Model {model_name} loaded successfully")
                # Pre-register the eviction counter time series for this model
                # so the row appears in /metrics with value 0 from the moment
                # the model is loaded, instead of only after the first eviction.
                try:
                    from app import metrics as prom_metrics
                    prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=model_name)
                except Exception:
                    pass
    _whisper_models_last_used[model_name] = time.time()
    _ensure_eviction_thread()
    return _whisper_models[model_name]


def _ensure_eviction_thread():
    """Lazily start the idle-model eviction daemon (no-op if disabled)."""
    global _eviction_thread_started
    if MODEL_KEEP_ALIVE_SECONDS <= 0 or _eviction_thread_started:
        return
    with _eviction_thread_lock:
        if _eviction_thread_started:
            return
        t = threading.Thread(
            target=_eviction_loop, daemon=True, name="model-evictor"
        )
        t.start()
        _eviction_thread_started = True
        logger.info(
            f"Idle model eviction enabled: unload after "
            f"{MODEL_KEEP_ALIVE_SECONDS}s idle, sweep every "
            f"{MODEL_EVICTION_INTERVAL_SECONDS}s"
        )


def _evict_from_cache(
    cache: dict,
    last_used: dict,
    label: str,
    now: float,
    with_metrics: bool = False,
) -> bool:
    """Evict entries idle longer than MODEL_KEEP_ALIVE_SECONDS from a model cache.

    Snapshots last_used under the lock, then re-checks inside the lock before
    deleting to avoid racing against concurrent loaders.
    Returns True if at least one entry was evicted.
    """
    with _model_load_lock:
        snapshot = list(last_used.items())
    candidates = [k for k, last in snapshot
                  if now - last > MODEL_KEEP_ALIVE_SECONDS and k in cache]
    evicted_any = False
    for key in candidates:
        with _model_load_lock:
            last = last_used.get(key, 0)
            if key in cache and now - last > MODEL_KEEP_ALIVE_SECONDS:
                logger.info(f"Evicting idle {label} {key}")
                del cache[key]
                last_used.pop(key, None)
                evicted_any = True
                if with_metrics:
                    try:
                        from app import metrics as prom_metrics
                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=key).inc()
                    except Exception:
                        pass
    return evicted_any


def _eviction_loop():
    global _diarize_pipeline, _diarize_last_used
    while True:
        time.sleep(MODEL_EVICTION_INTERVAL_SECONDS)
        if MODEL_KEEP_ALIVE_SECONDS <= 0:
            continue
        now = time.time()
        evicted_any = _evict_from_cache(
            _whisper_models, _whisper_models_last_used, "model", now, with_metrics=True
        )
        evicted_any |= _evict_from_cache(
            _align_models, _align_models_last_used, "alignment model for language", now
        )

        # Sweep idle diarization pipeline (singleton).
        if (
            _diarize_last_used is not None
            and now - _diarize_last_used > MODEL_KEEP_ALIVE_SECONDS
        ):
            with _model_load_lock:
                if (
                    _diarize_last_used is not None
                    and now - _diarize_last_used > MODEL_KEEP_ALIVE_SECONDS
                    and _diarize_pipeline is not None
                ):
                    logger.info("Evicting idle diarization pipeline")
                    _diarize_pipeline = None
                    _diarize_last_used = None
                    evicted_any = True

        if evicted_any:
            clear_gpu_memory()


def load_align_model(language_code: str):
    """Load alignment model with per-language caching (thread-safe)."""
    if language_code not in _align_models:
        with _model_load_lock:
            if language_code not in _align_models:
                logger.info(
                    f"Loading alignment model for language: {language_code} "
                    f"on {ALIGN_DEVICE}"
                )
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=ALIGN_DEVICE,
                    model_dir=CACHE_DIR,
                )
                _align_models[language_code] = (model_a, metadata)
                logger.info(f"Alignment model for {language_code} loaded")
    with _model_load_lock:
        _align_models_last_used[language_code] = time.time()
    _ensure_eviction_thread()
    return _align_models[language_code]


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into `base` in place."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _set_scoped(params: dict, section: str, key: str, value: float) -> bool:
    """Set params[section][key]=value only if that exact path already exists."""
    sec = params.get(section)
    if isinstance(sec, dict) and key in sec:
        sec[key] = value
        return True
    return False


def _apply_diarize_tuning(pipeline_wrapper: DiarizationPipeline) -> None:
    """
    Apply env-configured hyperparameter overrides to the underlying pyannote
    pipeline. No-op unless at least one DIARIZE_* tuning var is set.

    Reads the pipeline's *actual* instantiated parameters and merges overrides
    into them, so this stays correct regardless of community-1's internal
    parameter schema. Any failure is logged and swallowed -- diarization then
    runs with published defaults rather than breaking.
    """
    if not any([DIARIZE_CLUSTERING_THRESHOLD, DIARIZE_MIN_DURATION_OFF, DIARIZE_PARAM_OVERRIDES]):
        return

    pyannote_pipeline = getattr(pipeline_wrapper, "model", None)
    if pyannote_pipeline is None:
        logger.warning("Diarization tuning requested but underlying pyannote pipeline not accessible; using defaults")
        return

    try:
        current = pyannote_pipeline.parameters(instantiated=True)
        params = copy.deepcopy(dict(current))
    except Exception as e:
        logger.warning(f"Could not read diarization pipeline parameters for tuning ({e}); using defaults")
        return

    logger.info(f"Diarization default hyperparameters: {params}")

    applied = []
    if DIARIZE_CLUSTERING_THRESHOLD is not None:
        try:
            val = float(DIARIZE_CLUSTERING_THRESHOLD)
        except ValueError:
            logger.warning(f"DIARIZE_CLUSTERING_THRESHOLD={DIARIZE_CLUSTERING_THRESHOLD!r} is not a number; ignoring")
        else:
            if _set_scoped(params, "clustering", "threshold", val):
                applied.append(f"clustering.threshold={val}")
            else:
                logger.warning(
                    "DIARIZE_CLUSTERING_THRESHOLD set but no clustering.threshold in pipeline "
                    "params (see logged schema above); ignoring"
                )
    if DIARIZE_MIN_DURATION_OFF is not None:
        try:
            val = float(DIARIZE_MIN_DURATION_OFF)
        except ValueError:
            logger.warning(f"DIARIZE_MIN_DURATION_OFF={DIARIZE_MIN_DURATION_OFF!r} is not a number; ignoring")
        else:
            if _set_scoped(params, "segmentation", "min_duration_off", val):
                applied.append(f"segmentation.min_duration_off={val}")
            else:
                logger.warning(
                    "DIARIZE_MIN_DURATION_OFF set but no segmentation.min_duration_off in pipeline "
                    "params (see logged schema above); ignoring"
                )
    if DIARIZE_PARAM_OVERRIDES:
        try:
            overrides = json.loads(DIARIZE_PARAM_OVERRIDES)
            if not isinstance(overrides, dict):
                raise ValueError("DIARIZE_PARAM_OVERRIDES must be a JSON object")
            _deep_merge(params, overrides)
            applied.append(f"json_overrides={overrides}")
        except Exception as e:
            logger.warning(f"DIARIZE_PARAM_OVERRIDES could not be applied ({e}); ignoring")

    if not applied:
        return

    try:
        pyannote_pipeline.instantiate(params)
        logger.info(f"Applied diarization hyperparameter overrides: {', '.join(applied)}")
    except Exception as e:
        logger.warning(f"Failed to apply diarization hyperparameter overrides ({e}); using defaults")


def load_diarize_pipeline() -> DiarizationPipeline:
    """Load diarization pipeline (singleton, thread-safe)."""
    global _diarize_pipeline, _diarize_last_used
    if _diarize_pipeline is None:
        with _model_load_lock:
            if _diarize_pipeline is None:
                logger.info("Loading diarization pipeline: pyannote/speaker-diarization-community-1")
                pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-community-1",
                    use_auth_token=HF_TOKEN,
                    device=torch.device(DEVICE),
                )
                _apply_diarize_tuning(pipeline)
                _diarize_pipeline = pipeline
                logger.info("Diarization pipeline loaded")
    with _model_load_lock:
        _diarize_last_used = time.time()
    _ensure_eviction_thread()
    return _diarize_pipeline


# ---------------------------------------------------------------------------
# Stage 1 -- Transcription
# ---------------------------------------------------------------------------
def transcribe(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    task: str = "transcribe",
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
) -> dict:
    """Run WhisperX transcription and return raw result dict."""
    if ASR_BACKEND in ("qwen3", "external"):
        if task == "transcribe":
            if ASR_BACKEND == "qwen3":
                from app import qwen3_backend as backend_mod
            else:
                from app import external_backend as backend_mod

            context = " ".join(p for p in (initial_prompt, hotwords) if p) or None
            logger.info(f"Starting transcription ({ASR_BACKEND} backend)...")
            result = backend_mod.transcribe(audio, language=language, context=context)
            logger.info(
                f"Transcription complete. Detected language: {result.get('language')}"
            )
            clear_gpu_memory()
            return result
        logger.warning(
            f"ASR_BACKEND={ASR_BACKEND} does not support task=translate; "
            "using whisper backend for this request"
        )

    whisper_model = load_whisper_model(model_name)

    # Set per-request options on the model's transcription options.
    # The model is cached/shared, so we must reset after transcription.
    if hotwords is not None:
        whisper_model.options.hotwords = hotwords
    if initial_prompt is not None:
        whisper_model.options.initial_prompt = initial_prompt

    transcribe_options: Dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "language": language,
        "task": task,
    }

    logger.info("Starting transcription...")
    try:
        result = whisper_model.transcribe(audio, **transcribe_options)
    finally:
        if hotwords is not None:
            whisper_model.options.hotwords = None
        if initial_prompt is not None:
            whisper_model.options.initial_prompt = None

    detected_language = result.get("language", language or "en")
    logger.info(f"Transcription complete. Detected language: {detected_language}")

    clear_gpu_memory()
    return result


# ---------------------------------------------------------------------------
# Stage 2 -- Alignment
# ---------------------------------------------------------------------------
def align(audio: np.ndarray, result: dict) -> dict:
    """Run alignment to get word-level timestamps (Wav2Vec2, or the Qwen3
    forced aligner for qwen3-backend results and external results without
    provider timestamps)."""
    if result.get("_asr_backend") == "qwen3" or result.get("_qwen_align"):
        from app import qwen3_backend

        logger.info("Aligning timestamps (qwen3 forced aligner)...")
        try:
            result = qwen3_backend.align(audio, result)
            logger.info("Timestamp alignment complete")
            clear_gpu_memory()
        except Exception as e:
            logger.warning(
                f"Timestamp alignment failed: {e}, continuing without word-level timestamps"
            )
        return result

    detected_language = result.get("language", "en")
    logger.info("Aligning timestamps...")
    try:
        model_a, metadata = load_align_model(detected_language)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            ALIGN_DEVICE,
            return_char_alignments=False,
        )
        with _model_load_lock:
            _align_models_last_used[detected_language] = time.time()
        logger.info("Timestamp alignment complete")
        clear_gpu_memory()
    except Exception as e:
        logger.warning(f"Timestamp alignment failed: {e}, continuing without word-level timestamps")
    return result


# ---------------------------------------------------------------------------
# Stage 3 -- Diarization
# ---------------------------------------------------------------------------
def diarize(
    audio: np.ndarray,
    result: dict,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_speaker_embeddings: bool = False,
) -> Tuple[dict, Optional[dict]]:
    """
    Run pyannote speaker diarization and assign speakers to segments.

    Returns (result_with_speakers, speaker_embeddings_or_None).
    """
    global _diarize_last_used

    if not HF_TOKEN:
        logger.warning("Speaker diarization requested but HF_TOKEN not set")
        return result, None

    logger.info("Starting speaker diarization...")
    speaker_embeddings = None
    try:
        diarize_model = load_diarize_pipeline()

        diarize_params: Dict[str, Any] = {}
        if num_speakers is not None:
            diarize_params["num_speakers"] = num_speakers
            logger.info(f"Diarization with exact speaker count: {num_speakers}")
        else:
            if min_speakers is not None:
                diarize_params["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_params["max_speakers"] = max_speakers
            logger.info(f"Diarization with speaker range: {min_speakers}-{max_speakers}")

        if return_speaker_embeddings:
            diarize_params["return_embeddings"] = True
            logger.info("Speaker embeddings will be returned")

        diarize_output = diarize_model(audio, **diarize_params)

        if return_speaker_embeddings and isinstance(diarize_output, tuple):
            diarize_segments, speaker_embeddings = diarize_output
            logger.info(f"Received speaker embeddings for {len(speaker_embeddings)} speakers")
        else:
            diarize_segments = diarize_output

        if hasattr(diarize_segments, "exclusive_speaker_diarization"):
            diarize_segments = diarize_segments.exclusive_speaker_diarization
            logger.info("Using exclusive speaker diarization for better timestamp reconciliation")

        result = whisperx.assign_word_speakers(
            diarize_segments, result, fill_nearest=DIARIZE_FILL_NEAREST
        )
        # qwen3 and text-only external segments are built from silence gaps
        # or chunk spans alone, so a segment can span several speakers'
        # turns. Now that words carry speaker labels, rebuild the segments
        # so each one holds a single speaker's turn. Opt-in for the whisper
        # backend via RESEGMENT_BY_SPEAKER.
        if (
            result.get("_asr_backend") == "qwen3"
            or result.get("_qwen_align")
            or RESEGMENT_BY_SPEAKER
        ):
            from app import qwen3_backend

            result = qwen3_backend.resegment_by_speaker(result)
        with _model_load_lock:
            _diarize_last_used = time.time()
        logger.info("Speaker diarization complete")
        clear_gpu_memory()
    except Exception as e:
        logger.warning(f"Speaker diarization failed: {e}, continuing without diarization")

    return result, speaker_embeddings


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------
def sanitize_float_values(obj):
    """Recursively sanitize float values for JSON compliance (NaN/Inf -> None)."""
    if isinstance(obj, dict):
        return {key: sanitize_float_values(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_float_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_float_values(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return obj


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------
def run_pipeline(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    task: str = "transcribe",
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
    word_timestamps: bool = True,
    should_diarize: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_speaker_embeddings: bool = False,
) -> Tuple[dict, Optional[dict]]:
    """
    Run the full 3-stage pipeline: transcribe -> align -> diarize.

    Returns (result, speaker_embeddings_or_None).
    """
    result = transcribe(
        audio,
        model_name=model_name,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )

    # The qwen3 backend and text-only external results produce coarse
    # chunk-level segments, so speaker assignment needs word-level
    # timestamps even if the caller did not ask for them; the whisper
    # backend keeps its original behaviour.
    internally_aligned = (
        result.get("_asr_backend") == "qwen3" or result.get("_qwen_align")
    )
    needs_align = word_timestamps or (should_diarize and internally_aligned)
    if needs_align:
        result = align(audio, result)

    speaker_embeddings = None
    if should_diarize:
        result, speaker_embeddings = diarize(
            audio,
            result,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
        )

    # Non-whisper backends may align even when the caller did not ask for
    # word timestamps (speaker assignment needs them); strip the word-level
    # data from the response in that case, matching the whisper backend's
    # response shape.
    if not word_timestamps and internally_aligned:
        result.pop("word_segments", None)
        for seg in result.get("segments", []):
            seg.pop("words", None)

    result.pop("_asr_backend", None)
    result.pop("_language_name", None)
    result.pop("_qwen_align", None)
    return result, speaker_embeddings
