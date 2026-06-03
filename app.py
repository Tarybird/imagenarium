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
from PIL import Image


load_dotenv()


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
            flash("Job not found.", "error")
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
    def generate():
        prompt = request.form.get("prompt", "").strip()
        negative_prompt = request.form.get("negative_prompt", "").strip()
        width = parse_int(request.form.get("width"), get_default_int("DEFAULT_WIDTH", 1024))
        height = parse_int(request.form.get("height"), get_default_int("DEFAULT_HEIGHT", 1024))
        steps = parse_int(request.form.get("steps"), 30)
        seed = parse_optional_int(request.form.get("seed"))
        model = os.environ.get("RUNWARE_TEXT_MODEL", "runware:101@1")

        if not prompt:
            flash("Prompt is required.", "error")
            return redirect(url_for("index"))

        try:
            result = runware_generate(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
                model=model,
            )
            job_id = create_job(
                job_type="generate",
                status="done",
                prompt=prompt,
                model=model,
                width=width,
                height=height,
                cost_usd=result.cost,
                result_file=result.local_path,
            )
            flash("Image generated.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="generate",
                status="error",
                prompt=prompt,
                model=model,
                width=width,
                height=height,
                error_message=str(exc),
            )
            flash(f"Generation failed: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/remove-background")
    def remove_background():
        source = get_upload_or_source("image", "source_job_id")
        if source is None:
            flash("Upload an image or choose a previous result.", "error")
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
            flash("Background removed.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="remove_bg",
                status="error",
                source_file=source.saved_path,
                model=model,
                error_message=str(exc),
            )
            flash(f"Background removal failed: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/replace-background-prompt")
    def replace_background_prompt():
        source = get_upload_or_source("image", "source_job_id")
        prompt = request.form.get("background_prompt", "").strip()
        width = parse_int(request.form.get("width"), get_default_int("DEFAULT_WIDTH", 1024))
        height = parse_int(request.form.get("height"), get_default_int("DEFAULT_HEIGHT", 1024))
        if source is None or not prompt:
            flash("Upload an image and describe the background.", "error")
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
            flash("Background replaced using inpainting.", "success")
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
            flash(f"Background replacement failed: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/replace-background-image")
    def replace_background_image():
        source = get_upload_or_source("image", "source_job_id")
        background_source = get_upload_or_source("background_image", "background_job_id")
        if source is None or background_source is None:
            flash("Upload both source and background images.", "error")
            return redirect(url_for("index"))

        try:
            uploaded = runware_upload_image(source.source_uri)
            removed = runware_remove_background(
                uploaded.remote_uuid,
                model=os.environ.get("RUNWARE_REMOVE_BG_MODEL", "runware:109@1"),
                return_only_mask=False,
            )
            foreground = Image.open(removed.local_path).convert("RGBA")
            background = Image.open(background_source.saved_path).convert("RGBA").resize(foreground.size, Image.LANCZOS)
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
            flash("Background replaced from the second image.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="replace_bg_image",
                status="error",
                source_file=source.saved_path,
                background_file=background_source.saved_path,
                error_message=str(exc),
            )
            flash(f"Background replacement failed: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/upscale")
    def upscale():
        source = get_upload_or_source("image", "source_job_id")
        if source is None:
            flash("Upload an image or choose a previous result.", "error")
            return redirect(url_for("index"))

        model = os.environ.get("RUNWARE_UPSCALE_MODEL", "prunaai:p-image@upscale")
        target_megapixels = parse_int(request.form.get("target_megapixels"), 4)
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
            flash("Image upscaled.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="upscale",
                status="error",
                source_file=source.saved_path,
                model=model,
                error_message=str(exc),
            )
            flash(f"Upscale failed: {exc}", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    @app.post("/vectorize")
    def vectorize():
        source = get_upload_or_source("image", "source_job_id")
        if source is None:
            flash("Upload an image or choose a previous result.", "error")
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
            flash("Vector created.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        except Exception as exc:
            job_id = create_job(
                job_type="vectorize",
                status="error",
                source_file=source.saved_path,
                model=model,
                error_message=str(exc),
            )
            flash(f"Vectorization failed: {exc}", "error")
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


app = create_app()


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
    source_file: str | None = None,
    background_file: str | None = None,
    result_file: str | None = None,
    model: str | None = None,
    width: int | None = None,
    height: int | None = None,
    cost_usd: float | None = None,
    error_message: str | None = None,
) -> int:
    cursor = get_db().execute(
        """
        INSERT INTO jobs (
            type, status, prompt, source_file, background_file, result_file,
            model, width, height, cost_usd, created_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_type,
            status,
            prompt,
            source_file,
            background_file,
            result_file,
            model,
            width,
            height,
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


if __name__ == "__main__":
    app.run(debug=True)
