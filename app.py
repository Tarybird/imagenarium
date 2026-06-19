from __future__ import annotations

import base64
import io
import mimetypes
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from flask import Flask, flash, g, redirect, render_template, request, send_from_directory, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image


load_dotenv()


# Hard ceilings on paid-API inputs so a public demo cannot be abused into
# expensive calls (very large images, huge upscales, absurd step counts).
MAX_DIMENSION = 1536
MAX_STEPS = 40
MAX_UPSCALE_MEGAPIXELS = 4

# Quality presets for /generate: "standard" keeps the requested size/steps as-is,
# "economy" cuts resolution and step count to make draft generations cheaper,
# "test" is a minimal-cost preset for quickly sanity-checking a prompt before
# spending a full-price generation on it.
QUALITY_PRESETS = {
    "standard": {"dimension_factor": 1.0, "steps_factor": 1.0},
    "economy": {"dimension_factor": 0.5, "steps_factor": 0.6},
    "test": {"dimension_factor": 0.25, "steps_factor": 0.4},
}

# Runware quantizes width/height to multiples of 64 and enforces a minimum of
# 128px per side (https://runware.ai docs, imageInference parameters). The
# "test" preset above can shrink a 1024px default down past that floor, so
# the absolute minimum is clamped separately from the public-demo ceiling.
MIN_DIMENSION = 128
RUNWARE_DIMENSION_STEP = 64

# Aspect ratio presets for the generation form. Each maps to an explicit
# (width, height) pair at "standard" quality, already rounded to a multiple
# of 64 and capped at MAX_DIMENSION so no extra runtime rounding is needed.
ASPECT_RATIO_PRESETS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
}

# Vector icon styles: each combines the user's free-text subject with a
# hand-tuned FLUX prompt template tuned for that visual style, plus
# generation params suited to clean icon artwork (square, lower CFG for
# simpler/cleaner shapes than a typical photoreal CFG of 3.5-5).
ICON_STYLE_PRESETS = {
    "flat": {
        "label": "Flat design",
        "prompt_template": (
            "flat design icon of {subject}, vector style, solid flat colors, "
            "no gradients, no shadows, simple geometric shapes, clean bold "
            "outlines, centered composition, white background"
        ),
        "negative_prompt": "photo, realistic, 3d render, gradient, texture, shadow, clutter, text, watermark",
        "cfg_scale": 2.5,
        "size": 1024,
    },
    "line": {
        "label": "Line art / outline",
        "prompt_template": (
            "minimalist line art icon of {subject}, thin outline style, "
            "single weight stroke, no fill, black lines on white background, "
            "monochrome, simple, centered"
        ),
        "negative_prompt": "photo, realistic, color fill, gradient, shadow, texture, clutter, text, watermark",
        "cfg_scale": 2.5,
        "size": 1024,
    },
    "isometric": {
        "label": "Isometric",
        "prompt_template": (
            "isometric icon of {subject}, 3d isometric illustration, clean "
            "vector style, soft shadows, pastel color palette, centered, "
            "white background"
        ),
        "negative_prompt": "photo, realistic, flat 2d, text, watermark, clutter",
        "cfg_scale": 3.0,
        "size": 1024,
    },
    "gradient": {
        "label": "Gradient modern",
        "prompt_template": (
            "modern gradient icon of {subject}, vibrant smooth gradient "
            "colors, rounded shapes, soft glow, contemporary app icon style, "
            "centered, white background"
        ),
        "negative_prompt": "photo, realistic, flat color only, sketch, text, watermark, clutter",
        "cfg_scale": 3.0,
        "size": 1024,
    },
}

# FLUX.1 [dev] (runware:101@1) is guidance-distilled: unlike SDXL it does not
# use classifier-free guidance in the traditional sense and has a much
# narrower effective CFGScale range (roughly 2-5; values above ~5 add little
# and can start to look over-cooked, values near 1 barely follow the prompt).
# The previous code never sent CFGScale at all, leaving it to an undocumented
# API-side default - on a model this guidance-sensitive that produced
# generations that drifted from the prompt. Pin an explicit, known-good
# default and keep it overridable via env/form for experimentation.
DEFAULT_CFG_SCALE = 3.5
MIN_CFG_SCALE = 1.0
MAX_CFG_SCALE = 10.0


def clamp_cfg_scale(value: float) -> float:
    return max(MIN_CFG_SCALE, min(MAX_CFG_SCALE, value))


# Demo mode caps request volume on a public deployment. A holder of
# DEMO_ACCESS_TOKEN (sent as header X-Demo-Token or form field access_token)
# is treated as a trusted/own user and exempted from the stricter public
# rate limits below.
DEMO_ACCESS_TOKEN = os.environ.get("DEMO_ACCESS_TOKEN", "").strip()


def is_trusted_caller() -> bool:
    if not DEMO_ACCESS_TOKEN:
        return False
    supplied = (request.headers.get("X-Demo-Token") or request.form.get("access_token") or "").strip()
    return bool(supplied) and supplied == DEMO_ACCESS_TOKEN


def clamp_dimension(value: int) -> int:
    return max(MIN_DIMENSION, min(MAX_DIMENSION, value))


def quantize_dimension(value: int) -> int:
    """Round to the nearest multiple of RUNWARE_DIMENSION_STEP, then clamp.

    Runware rejects/auto-adjusts width/height that aren't multiples of 64;
    rounding client-side avoids silent server-side resizing surprises.
    """
    quantized = max(RUNWARE_DIMENSION_STEP, round(value / RUNWARE_DIMENSION_STEP) * RUNWARE_DIMENSION_STEP)
    return clamp_dimension(quantized)


def clamp_steps(value: int) -> int:
    return max(1, min(MAX_STEPS, value))


def clamp_megapixels(value: int) -> int:
    return max(1, min(MAX_UPSCALE_MEGAPIXELS, value))


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
UPLOAD_DIR = INSTANCE_DIR / "uploads"
GENERATED_DIR = BASE_DIR / "generated"
EDITED_DIR = BASE_DIR / "edited"
UPSCALED_DIR = BASE_DIR / "upscaled"
VECTORS_DIR = BASE_DIR / "vectors"
DATABASE_PATH = INSTANCE_DIR / "app.db"


def create_app() -> Flask:
    app = Flask(__name__, instance_path=str(INSTANCE_DIR))
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "imagenarium-dev-secret")
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

    for directory in [INSTANCE_DIR, UPLOAD_DIR, GENERATED_DIR, EDITED_DIR, UPSCALED_DIR, VECTORS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    init_db(app)

    # Per-IP rate limiting protects the paid Runware API from abuse on a
    # public demo. Holders of DEMO_ACCESS_TOKEN (see is_trusted_caller) are
    # exempt, since they are treated as the trusted operator, not the public.
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri="memory://",
        default_limits=[],
    )
    app.extensions["paid_api_exempt"] = is_trusted_caller

    @app.errorhandler(429)
    def rate_limit_exceeded(_exc):
        flash(
            "Достигнут демо-лимит запросов для вашего IP (не более 5 платных "
            "запросов к Runware в сутки). Попробуйте снова завтра.",
            "error",
        )
        return redirect(url_for("index")), 429

    @app.teardown_appcontext
    def close_db(_exception: Exception | None) -> None:
        database = g.pop("db", None)
        if database is not None:
            database.close()

    @app.context_processor
    def inject_now() -> dict[str, str]:
        return {"now_iso": datetime.utcnow().isoformat(timespec="seconds")}

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            jobs=list_jobs(limit=12),
            source_jobs=list_jobs(limit=20),
            defaults=get_defaults(),
        )

    @app.get("/jobs/<int:job_id>")
    def job_detail(job_id: int):
        job = get_job(job_id)
        if job is None:
            flash("Задание не найдено.", "error")
            return redirect(url_for("index"))
        return render_template(
            "result.html",
            job=job,
            jobs=list_jobs(limit=12),
            source_jobs=list_jobs(limit=20),
            defaults=get_defaults(),
        )

    @app.get("/media/<int:job_id>")
    def media(job_id: int):
        job = get_job(job_id)
        if job is None or not job["result_file"]:
            return ("Not found", 404)
        path = Path(job["result_file"])
        if not path.exists():
            return ("File missing", 404)
        return send_from_directory(path.parent, path.name)

    @app.post("/generate")
    @limiter.limit("5 per day", exempt_when=is_trusted_caller, scope="paid_api_daily")
    def generate():
        prompt = request.form.get("prompt", "").strip()
        negative_prompt = request.form.get("negative_prompt", "").strip()
        quality = request.form.get("quality", "standard").strip().lower()
        if quality not in QUALITY_PRESETS:
            quality = "standard"
        preset = QUALITY_PRESETS[quality]

        aspect_ratio = request.form.get("aspect_ratio", "").strip()
        if aspect_ratio in ASPECT_RATIO_PRESETS:
            requested_width, requested_height = ASPECT_RATIO_PRESETS[aspect_ratio]
        else:
            requested_width = parse_int(request.form.get("width"), get_default_int("DEFAULT_WIDTH", 1024))
            requested_height = parse_int(request.form.get("height"), get_default_int("DEFAULT_HEIGHT", 1024))
        requested_steps = parse_int(request.form.get("steps"), 30)
        requested_cfg_scale = parse_optional_float(request.form.get("cfg_scale"))
        cfg_scale = clamp_cfg_scale(
            requested_cfg_scale if requested_cfg_scale is not None else get_default_float("DEFAULT_CFG_SCALE", DEFAULT_CFG_SCALE)
        )

        width = quantize_dimension(int(requested_width * preset["dimension_factor"]))
        height = quantize_dimension(int(requested_height * preset["dimension_factor"]))
        steps = clamp_steps(int(requested_steps * preset["steps_factor"]))
        seed = parse_optional_int(request.form.get("seed"))
        model = os.environ.get("RUNWARE_TEXT_MODEL", "runware:101@1")

        if not prompt:
            flash("Промпт обязателен.", "error")
            return redirect(url_for("index"))

        try:
            result = runware_generate(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                model=model,
            )
            job_id = create_job(
                job_type="generate",
                status="done",
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                model=model,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                quality=quality,
                seed=seed,
                cost_usd=result.cost,
                result_file=result.local_path,
            )
            flash("Изображение сгенерировано.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="generate",
                status="error",
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                model=model,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                quality=quality,
                seed=seed,
                error_message=str(exc),
            )
            flash(f"Не удалось сгенерировать изображение: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/remove-background")
    @limiter.limit("5 per day", exempt_when=is_trusted_caller, scope="paid_api_daily")
    def remove_background():
        source = get_upload_or_source("image", "source_job_id")
        if source is None:
            flash("Загрузите изображение или выберите предыдущий результат.", "error")
            return redirect(url_for("index"))

        model = os.environ.get("RUNWARE_REMOVE_BG_MODEL", "runware:109@1")
        try:
            uploaded = runware_upload_image(source.source_uri)
            result = runware_remove_background(uploaded.remote_uuid, model=model, return_only_mask=False)
            job_id = create_job(
                job_type="remove_bg",
                status="done",
                source_file=source.saved_path,
                model=model,
                cost_usd=result.cost,
                result_file=result.local_path,
            )
            flash("Фон удалён.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="remove_bg",
                status="error",
                source_file=source.saved_path,
                model=model,
                error_message=str(exc),
            )
            flash(f"Не удалось удалить фон: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/replace-background-prompt")
    @limiter.limit("5 per day", exempt_when=is_trusted_caller, scope="paid_api_daily")
    def replace_background_prompt():
        source = get_upload_or_source("image", "source_job_id")
        prompt = request.form.get("background_prompt", "").strip()
        width = clamp_dimension(parse_int(request.form.get("width"), get_default_int("DEFAULT_WIDTH", 1024)))
        height = clamp_dimension(parse_int(request.form.get("height"), get_default_int("DEFAULT_HEIGHT", 1024)))
        if source is None or not prompt:
            flash("Загрузите изображение и опишите фон.", "error")
            return redirect(url_for("index"))

        inpaint_model = os.environ.get("RUNWARE_INPAINT_MODEL", "runware:102@1")
        try:
            uploaded = runware_upload_image(source.source_uri)
            mask_result = runware_remove_background(
                uploaded.remote_uuid,
                model=os.environ.get("RUNWARE_REMOVE_BG_MODEL", "runware:109@1"),
                return_only_mask=True,
            )
            mask_upload = runware_upload_image(mask_result.remote_url)
            inpaint = runware_inpaint(
                prompt=prompt,
                seed_image_uuid=uploaded.remote_uuid,
                mask_image_uuid=mask_upload.remote_uuid,
                width=width,
                height=height,
                model=inpaint_model,
            )
            job_id = create_job(
                job_type="replace_bg_prompt",
                status="done",
                source_file=source.saved_path,
                prompt=prompt,
                model=inpaint_model,
                width=width,
                height=height,
                cost_usd=(mask_result.cost or 0.0) + (inpaint.cost or 0.0),
                result_file=inpaint.local_path,
            )
            flash("Фон заменён методом inpainting.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="replace_bg_prompt",
                status="error",
                source_file=source.saved_path,
                prompt=prompt,
                model=inpaint_model,
                width=width,
                height=height,
                error_message=str(exc),
            )
            flash(f"Не удалось заменить фон: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/replace-background-image")
    @limiter.limit("5 per day", exempt_when=is_trusted_caller, scope="paid_api_daily")
    def replace_background_image():
        source = get_upload_or_source("image", "source_job_id")
        background_source = get_upload_or_source("background_image", "background_job_id")
        if source is None or background_source is None:
            flash("Загрузите оба изображения: исходное и фон.", "error")
            return redirect(url_for("index"))

        try:
            uploaded = runware_upload_image(source.source_uri)
            removed = runware_remove_background(
                uploaded.remote_uuid,
                model=os.environ.get("RUNWARE_REMOVE_BG_MODEL", "runware:109@1"),
                return_only_mask=False,
            )
            foreground = Image.open(removed.local_path).convert("RGBA")
            background_raw = Image.open(background_source.saved_path).convert("RGBA")
            background = resize_cover(background_raw, *foreground.size)
            composited = Image.alpha_composite(background, foreground)
            composited_path = save_image(composited, EDITED_DIR / f"{uuid.uuid4().hex}_composited.png")
            job_id = create_job(
                job_type="replace_bg_image",
                status="done",
                source_file=source.saved_path,
                background_file=background_source.saved_path,
                result_file=composited_path,
                cost_usd=removed.cost,
            )
            flash("Фон заменён из второго изображения.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="replace_bg_image",
                status="error",
                source_file=source.saved_path,
                background_file=background_source.saved_path,
                error_message=str(exc),
            )
            flash(f"Не удалось заменить фон: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/inpaint")
    @limiter.limit("5 per day", exempt_when=is_trusted_caller, scope="paid_api_daily")
    def inpaint():
        source = get_upload_or_source("image", "source_job_id")
        prompt = request.form.get("inpaint_prompt", "").strip()
        mask_data_url = request.form.get("mask_data_url", "").strip()
        if source is None or not prompt or not mask_data_url:
            flash("Загрузите изображение, выделите область на маске и опишите, что должно там быть.", "error")
            return redirect(url_for("index"))

        try:
            source_image = Image.open(source.saved_path)
            mask_path = save_mask_data_uri(mask_data_url, source_image.size)
        except Exception as exc:
            flash(f"Не удалось обработать маску: {exc}", "error")
            return redirect(url_for("index"))

        width = clamp_dimension(source_image.width)
        height = clamp_dimension(source_image.height)
        inpaint_model = os.environ.get("RUNWARE_INPAINT_MODEL", "runware:102@1")
        try:
            uploaded = runware_upload_image(source.source_uri)
            mask_uploaded = runware_upload_image(path_to_data_uri(mask_path))
            result = runware_inpaint(
                prompt=prompt,
                seed_image_uuid=uploaded.remote_uuid,
                mask_image_uuid=mask_uploaded.remote_uuid,
                width=width,
                height=height,
                model=inpaint_model,
            )
            job_id = create_job(
                job_type="inpaint",
                status="done",
                source_file=source.saved_path,
                prompt=prompt,
                model=inpaint_model,
                width=width,
                height=height,
                cost_usd=result.cost,
                result_file=result.local_path,
            )
            flash("Область перегенерирована.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="inpaint",
                status="error",
                source_file=source.saved_path,
                prompt=prompt,
                model=inpaint_model,
                width=width,
                height=height,
                error_message=str(exc),
            )
            flash(f"Не удалось перегенерировать область: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/generate-icon")
    @limiter.limit("5 per day", exempt_when=is_trusted_caller, scope="paid_api_daily")
    def generate_icon():
        subject = request.form.get("icon_subject", "").strip()
        style = request.form.get("icon_style", "").strip().lower()
        if style not in ICON_STYLE_PRESETS:
            style = "flat"
        if not subject:
            flash("Опишите, какая иконка нужна.", "error")
            return redirect(url_for("index"))

        style_preset = ICON_STYLE_PRESETS[style]
        prompt = style_preset["prompt_template"].format(subject=subject)
        negative_prompt = style_preset.get("negative_prompt", "")
        cfg_scale = style_preset.get("cfg_scale", DEFAULT_CFG_SCALE)
        size = style_preset.get("size", 1024)
        model = os.environ.get("RUNWARE_TEXT_MODEL", "runware:101@1")

        try:
            result = runware_generate(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=size,
                height=size,
                steps=30,
                cfg_scale=cfg_scale,
                model=model,
            )
            job_id = create_job(
                job_type="icon",
                status="done",
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                model=model,
                width=size,
                height=size,
                cfg_scale=cfg_scale,
                cost_usd=result.cost,
                result_file=result.local_path,
            )
            flash("Иконка сгенерирована.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="icon",
                status="error",
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                model=model,
                width=size,
                height=size,
                cfg_scale=cfg_scale,
                error_message=str(exc),
            )
            flash(f"Не удалось сгенерировать иконку: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/upscale")
    @limiter.limit("5 per day", exempt_when=is_trusted_caller, scope="paid_api_daily")
    def upscale():
        source = get_upload_or_source("image", "source_job_id")
        if source is None:
            flash("Загрузите изображение или выберите предыдущий результат.", "error")
            return redirect(url_for("index"))

        model = os.environ.get("RUNWARE_UPSCALE_MODEL", "prunaai:p-image@upscale")
        target_megapixels = clamp_megapixels(parse_int(request.form.get("target_megapixels"), 4))
        try:
            uploaded = runware_upload_image(source.source_uri)
            result = runware_upscale(uploaded.remote_uuid, model=model, target_megapixels=target_megapixels)
            job_id = create_job(
                job_type="upscale",
                status="done",
                source_file=source.saved_path,
                model=model,
                cost_usd=result.cost,
                result_file=result.local_path,
            )
            flash("Изображение увеличено.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="upscale",
                status="error",
                source_file=source.saved_path,
                model=model,
                error_message=str(exc),
            )
            flash(f"Не удалось увеличить изображение: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/vectorize")
    @limiter.limit("5 per day", exempt_when=is_trusted_caller, scope="paid_api_daily")
    def vectorize():
        source = get_upload_or_source("image", "source_job_id")
        if source is None:
            flash("Загрузите изображение или выберите предыдущий результат.", "error")
            return redirect(url_for("index"))

        model = os.environ.get("RUNWARE_VECTORIZE_MODEL", "recraft:1@1")
        try:
            uploaded = runware_upload_image(source.source_uri)
            result = runware_vectorize(uploaded.remote_uuid, model=model)
            job_id = create_job(
                job_type="vectorize",
                status="done",
                source_file=source.saved_path,
                model=model,
                cost_usd=result.cost,
                result_file=result.local_path,
            )
            flash("Векторное изображение создано.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="vectorize",
                status="error",
                source_file=source.saved_path,
                model=model,
                error_message=str(exc),
            )
            flash(f"Не удалось векторизовать изображение: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.get("/files/<path:filename>")
    def files(filename: str):
        for directory in [UPLOAD_DIR, GENERATED_DIR, EDITED_DIR, UPSCALED_DIR, VECTORS_DIR]:
            candidate = directory / filename
            if candidate.exists():
                return send_from_directory(directory, filename)
        return ("Not found", 404)

    @app.get("/download/<int:job_id>")
    def download(job_id: int):
        job = get_job(job_id)
        if job is None or not job["result_file"]:
            return ("Not found", 404)
        path = Path(job["result_file"])
        if not path.exists():
            return ("File missing", 404)
        return send_from_directory(path.parent, path.name, as_attachment=True)

    return app


@dataclass
class StoredSource:
    saved_path: str
    source_uri: str


@dataclass
class RunwareResult:
    remote_url: str
    local_path: str
    cost: float | None


@dataclass
class UploadedImage:
    remote_uuid: str


def init_db(app: Flask) -> None:
    with app.app_context():
        db = get_db()
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                prompt TEXT,
                source_file TEXT,
                background_file TEXT,
                result_file TEXT,
                model TEXT,
                width INTEGER,
                height INTEGER,
                cost_usd REAL,
                created_at TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        # Generation parameters needed to support "regenerate with edits" from
        # history: the original jobs table only stored prompt/model/width/
        # height, which is not enough to repopulate the form faithfully.
        existing_columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
        for column, ddl in (
            ("negative_prompt", "ALTER TABLE jobs ADD COLUMN negative_prompt TEXT"),
            ("steps", "ALTER TABLE jobs ADD COLUMN steps INTEGER"),
            ("cfg_scale", "ALTER TABLE jobs ADD COLUMN cfg_scale REAL"),
            ("quality", "ALTER TABLE jobs ADD COLUMN quality TEXT"),
            ("seed", "ALTER TABLE jobs ADD COLUMN seed INTEGER"),
        ):
            if column not in existing_columns:
                db.execute(ddl)
        db.commit()


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def list_jobs(limit: int = 20) -> list[dict]:
    rows = get_db().execute(
        "SELECT * FROM jobs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_job(job_id: int) -> dict | None:
    row = get_db().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def create_job(
    *,
    job_type: str,
    status: str,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    source_file: str | None = None,
    background_file: str | None = None,
    result_file: str | None = None,
    model: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    quality: str | None = None,
    seed: int | None = None,
    cost_usd: float | None = None,
    error_message: str | None = None,
) -> int:
    cursor = get_db().execute(
        """
        INSERT INTO jobs (
            type, status, prompt, negative_prompt, source_file, background_file,
            result_file, model, width, height, steps, cfg_scale, quality, seed,
            cost_usd, created_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_type,
            status,
            prompt,
            negative_prompt,
            source_file,
            background_file,
            result_file,
            model,
            width,
            height,
            steps,
            cfg_scale,
            quality,
            seed,
            cost_usd,
            datetime.utcnow().isoformat(timespec="seconds"),
            error_message,
        ),
    )
    get_db().commit()
    return int(cursor.lastrowid)


def get_defaults() -> dict[str, int]:
    return {
        "width": get_default_int("DEFAULT_WIDTH", 1024),
        "height": get_default_int("DEFAULT_HEIGHT", 1024),
    }


def get_default_int(name: str, fallback: int) -> int:
    try:
        return int(os.environ.get(name, str(fallback)))
    except ValueError:
        return fallback


def parse_int(value: str | None, fallback: int) -> int:
    try:
        return int(value) if value not in (None, "") else fallback
    except ValueError:
        return fallback


def parse_optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def get_default_float(name: str, fallback: float) -> float:
    try:
        return float(os.environ.get(name, str(fallback)))
    except ValueError:
        return fallback


def get_upload_or_source(file_field: str, job_field: str) -> StoredSource | None:
    uploaded = request.files.get(file_field)
    if uploaded and uploaded.filename:
        saved = save_upload(uploaded)
        return StoredSource(saved_path=str(saved), source_uri=path_to_data_uri(saved))

    job_id = parse_optional_int(request.form.get(job_field))
    if not job_id:
        return None
    job = get_job(job_id)
    if job is None or not job.get("result_file"):
        return None
    path = Path(job["result_file"])
    if not path.exists():
        return None
    return StoredSource(saved_path=str(path), source_uri=path_to_data_uri(path))


def save_upload(file_storage) -> Path:
    filename = secure_stem(file_storage.filename)
    extension = Path(filename).suffix.lower()
    if not extension:
        guessed = mimetypes.guess_extension(file_storage.mimetype or "")
        extension = guessed or ".png"
    unique_name = f"{uuid.uuid4().hex}_{Path(filename).stem}{extension}"
    destination = UPLOAD_DIR / unique_name
    file_storage.save(destination)
    return destination


def secure_stem(filename: str) -> str:
    stem = Path(filename).name.replace("\\", "_").replace("/", "_")
    return "".join(char for char in stem if char.isalnum() or char in {"-", "_", "."}) or "upload.png"


def path_to_data_uri(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"

def save_mask_data_uri(data_uri: str, target_size: tuple[int, int]) -> Path:
    """Decode a canvas-drawn mask (data:image/png;base64,... from the
    inpainting <canvas>) into a black/white PNG matching the source image
    size, and save it to UPLOAD_DIR for upload to Runware.

    The browser canvas is drawn at the displayed (CSS) size, which may not
    match the original image's pixel dimensions, so the mask is resized
    (nearest-neighbor, to keep edges crisp/binary) to the source size before
    saving.
    """
    if "," not in data_uri:
        raise ValueError("Некорректные данные маски")
    header, encoded = data_uri.split(",", 1)
    if "base64" not in header:
        raise ValueError("Маска должна быть в формате base64")
    raw = base64.b64decode(encoded)
    mask_image = Image.open(io.BytesIO(raw)).convert("L")
    if mask_image.size != target_size:
        mask_image = mask_image.resize(target_size, Image.NEAREST)
    # Binarize: anything painted (non-zero alpha-derived gray) becomes pure
    # white "regenerate this" per Runware's maskImage convention; everything
    # else becomes pure black "keep as-is".
    mask_image = mask_image.point(lambda px: 255 if px >= 16 else 0)
    destination = UPLOAD_DIR / f"{uuid.uuid4().hex}_mask.png"
    mask_image.save(destination)
    return destination


def resize_cover(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize ``image`` to exactly (target_width, target_height) using
    "object-fit: cover" semantics: scale to fully cover the target box while
    preserving aspect ratio, then center-crop the overflow. This avoids the
    distortion caused by a naive ``resize((w, h))`` (which behaves like
    "object-fit: fill" and stretches/squishes the image to match the target
    aspect ratio).
    """
    source_width, source_height = image.size
    if source_width == 0 or source_height == 0:
        return image.resize((target_width, target_height), Image.LANCZOS)

    scale = max(target_width / source_width, target_height / source_height)
    scaled_width = max(1, round(source_width * scale))
    scaled_height = max(1, round(source_height * scale))
    scaled = image.resize((scaled_width, scaled_height), Image.LANCZOS)

    left = (scaled_width - target_width) // 2
    top = (scaled_height - target_height) // 2
    return scaled.crop((left, top, left + target_width, top + target_height))


def save_image(image: Image.Image, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return str(destination)


def runware_request(tasks: list[dict]) -> dict | list[dict]:
    api_key = os.environ.get("RUNWARE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("RUNWARE_API_KEY is missing")

    api_url = os.environ.get("RUNWARE_API_URL", "https://api.runware.ai/v1").strip()
    response = requests.post(
        api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=tasks,
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def normalize_runware_item(payload: dict | list) -> dict:
    if isinstance(payload, list):
        if not payload:
            raise RuntimeError("Runware response is empty")
        return payload[0]
    if "data" in payload:
        data = payload["data"]
        if isinstance(data, list):
            if not data:
                raise RuntimeError("Runware response is empty")
            return data[0]
        if isinstance(data, dict):
            return data
    if "imageURL" in payload or "taskType" in payload:
        return payload
    raise RuntimeError(f"Unexpected Runware response: {payload}")


def runware_upload_image(source_uri: str) -> UploadedImage:
    task_uuid = str(uuid.uuid4())
    response = runware_request(
        [
            {
                "taskType": "imageUpload",
                "taskUUID": task_uuid,
                "image": source_uri,
            }
        ]
    )
    item = normalize_runware_item(response)
    image_uuid = item.get("imageUUID")
    if not image_uuid:
        raise RuntimeError(f"Runware did not return an image UUID: {item}")
    return UploadedImage(remote_uuid=str(image_uuid))


def download_remote_image(url: str, destination: Path) -> Path:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination


def file_extension_from_url(url: str, default: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    return suffix if suffix else default


def runware_generate(
    *,
    prompt: str,
    width: int,
    height: int,
    model: str,
    negative_prompt: str | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    output_format: str = "JPG",
) -> RunwareResult:
    task_uuid = str(uuid.uuid4())
    task: dict[str, object] = {
        "taskType": "imageInference",
        "taskUUID": task_uuid,
        "model": model,
        "positivePrompt": prompt,
        "width": width,
        "height": height,
        "outputType": "URL",
        "outputFormat": output_format,
        "includeCost": True,
    }
    if negative_prompt:
        task["negativePrompt"] = negative_prompt
    if steps is not None:
        task["steps"] = steps
    if cfg_scale is not None:
        task["CFGScale"] = cfg_scale
    if seed is not None:
        task["seed"] = seed

    response = runware_request([task])
    item = normalize_runware_item(response)
    image_url = item.get("imageURL")
    if not image_url:
        raise RuntimeError(f"Runware did not return an image URL: {item}")
    extension = file_extension_from_url(str(image_url), ".jpg")
    local_name = f"{uuid.uuid4().hex}{extension}"
    local_path = GENERATED_DIR / local_name
    download_remote_image(str(image_url), local_path)
    return RunwareResult(remote_url=str(image_url), local_path=str(local_path), cost=item.get("cost"))


def runware_inpaint(
    *,
    prompt: str,
    seed_image_uuid: str,
    mask_image_uuid: str,
    width: int,
    height: int,
    model: str,
) -> RunwareResult:
    task_uuid = str(uuid.uuid4())
    task = {
        "taskType": "imageInference",
        "taskUUID": task_uuid,
        "model": model,
        "positivePrompt": prompt,
        "seedImage": seed_image_uuid,
        "maskImage": mask_image_uuid,
        "width": width,
        "height": height,
        "outputType": "URL",
        "includeCost": True,
    }
    response = runware_request([task])
    item = normalize_runware_item(response)
    image_url = item.get("imageURL")
    if not image_url:
        raise RuntimeError(f"Runware did not return an image URL: {item}")
    extension = file_extension_from_url(str(image_url), ".jpg")
    local_name = f"{uuid.uuid4().hex}{extension}"
    local_path = EDITED_DIR / local_name
    download_remote_image(str(image_url), local_path)
    return RunwareResult(remote_url=str(image_url), local_path=str(local_path), cost=item.get("cost"))


def runware_remove_background(source_uri: str, *, model: str, return_only_mask: bool) -> RunwareResult:
    task_uuid = str(uuid.uuid4())
    task = {
        "taskType": "removeBackground",
        "taskUUID": task_uuid,
        "inputImage": source_uri,
        "model": model,
        "outputType": "URL",
        "outputFormat": "PNG",
        "includeCost": True,
        "settings": {
            "rgba": [255, 255, 255, 0],
            "postProcessMask": True,
            "returnOnlyMask": return_only_mask,
            "alphaMatting": True,
            "alphaMattingForegroundThreshold": 240,
            "alphaMattingBackgroundThreshold": 10,
            "alphaMattingErodeSize": 10,
        },
    }
    response = runware_request([task])
    item = normalize_runware_item(response)
    image_url = item.get("imageURL")
    if not image_url:
        raise RuntimeError(f"Runware did not return an image URL: {item}")
    extension = file_extension_from_url(str(image_url), ".png")
    local_name = f"{uuid.uuid4().hex}{extension}"
    local_path = EDITED_DIR / local_name
    download_remote_image(str(image_url), local_path)
    return RunwareResult(remote_url=str(image_url), local_path=str(local_path), cost=item.get("cost"))


def runware_upscale(source_uri: str, *, model: str, target_megapixels: int) -> RunwareResult:
    task_uuid = str(uuid.uuid4())
    task = {
        "taskType": "upscale",
        "taskUUID": task_uuid,
        "model": model,
        "inputs": {
            "image": source_uri,
        },
        "settings": {
            "enhanceDetails": True,
            "realism": True,
        },
        "targetMegapixels": target_megapixels,
        "outputType": "URL",
        "includeCost": True,
    }
    response = runware_request([task])
    item = normalize_runware_item(response)
    image_url = item.get("imageURL")
    if not image_url:
        raise RuntimeError(f"Runware did not return an image URL: {item}")
    extension = file_extension_from_url(str(image_url), ".png")
    local_name = f"{uuid.uuid4().hex}{extension}"
    local_path = UPSCALED_DIR / local_name
    download_remote_image(str(image_url), local_path)
    return RunwareResult(remote_url=str(image_url), local_path=str(local_path), cost=item.get("cost"))


def runware_vectorize(source_uri: str, *, model: str) -> RunwareResult:
    task_uuid = str(uuid.uuid4())
    task = {
        "taskType": "vectorize",
        "taskUUID": task_uuid,
        "model": model,
        "inputs": {
            "image": source_uri,
        },
        "includeCost": True,
    }
    response = runware_request([task])
    item = normalize_runware_item(response)
    image_url = item.get("imageURL")
    if not image_url:
        raise RuntimeError(f"Runware did not return an image URL: {item}")
    extension = file_extension_from_url(str(image_url), ".svg")
    local_name = f"{uuid.uuid4().hex}{extension}"
    local_path = VECTORS_DIR / local_name
    download_remote_image(str(image_url), local_path)
    return RunwareResult(remote_url=str(image_url), local_path=str(local_path), cost=item.get("cost"))


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
