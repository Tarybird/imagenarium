# Image Generator for Adobe Stock

Single-user web app for generating, editing, upscaling, and vectorizing images through Runware.

The project is designed for one operator who prepares AI-assisted images for Adobe Stock with minimal cost and a simple workflow.

## Goal

Build a practical desktop-friendly tool that lets one user:

- generate image drafts from prompts;
- upload an existing photo and remove or replace its background;
- upscale selected images for Adobe Stock submission;
- convert suitable raster images into SVG vectors;
- keep a local history of prompts, files, statuses, and costs.

## Product Scope

The first version is intentionally simple:

- one local user;
- no accounts and no roles;
- one backend provider: Runware;
- local file storage;
- local SQLite database;
- simple Flask interface.

## Core Features

### 1. Text to image

Use Runware image generation with `FLUX.1 [dev]` for the main generation flow.

Recommended default:

- model: `FLUX.1 [dev]`
- size: `1024x1024`
- output: `JPG`

Why:

- low cost;
- good quality for stock-style drafts;
- simple and fast enough for one user.

### 2. Background removal

Upload an existing image and remove the background through Runware `removeBackground`.

Expected result:

- transparent `PNG`;
- usable for product isolation, cutouts, and later compositing.

### 3. Background replacement by prompt

Upload an image, isolate the subject, then regenerate only the background using inpainting.

Example:

- source image: woman with laptop;
- prompt: `modern bright office interior with soft natural daylight`.

This should use:

- uploaded source image;
- background mask;
- Runware inpainting flow;
- prompt only for the replaced region.

### 4. Background replacement by second file

Upload:

- main image with subject;
- second image to use as new background.

Flow:

- remove background from main image;
- place isolated foreground over second image;
- optionally add a light final edit pass later if blending needs improvement.

For v1, a direct composite is enough.

### 5. Upscale for Adobe Stock

Upscale only selected final images, not every draft.

Recommended flow:

- generate at `1024x1024`;
- select the best result;
- upscale to at least `4 MP`.

Target example:

- `2048x2048` for a square image.

Recommended model:

- Runware `P-Image Upscale`

Reason:

- Adobe Stock requires at least `4 MP` for photos and illustrations;
- upscaling only approved drafts keeps the cost low.

### 6. Vectorization

For vectors, use Runware `vectorize` with `Recraft Vectorize`.

Recommended model:

- `recraft:1@1`

Use cases:

- icons;
- logos;
- isolated flat illustrations;
- simple graphic objects.

Important:

- vectorization is useful for simple clean artwork;
- realistic photos usually do not become high-quality SVGs.

## Recommended v1 Architecture

### Stack

- backend: `Flask`
- database: `SQLite`
- templates: `Jinja2`
- HTTP client: `requests`
- config: `python-dotenv`
- storage: local folders

### Proposed folders

```text
image-generator/
├── app.py
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── instance/
│   └── app.db
├── generated/
├── edited/
├── upscaled/
├── vectors/
└── templates/
    ├── index.html
    ├── result.html
    ├── edit.html
    ├── history.html
    └── vectorize.html
```

### Minimal data model

Use one table, for example `jobs`:

- `id`
- `type` (`generate`, `remove_bg`, `replace_bg_prompt`, `replace_bg_image`, `upscale`, `vectorize`)
- `status`
- `prompt`
- `source_file`
- `background_file`
- `result_file`
- `model`
- `width`
- `height`
- `cost_usd`
- `created_at`
- `error_message`

This is enough for a single-user MVP.

## API Flows

### Generate

1. User enters prompt.
2. Backend sends Runware image generation request.
3. Result image is downloaded and saved locally.
4. Database stores prompt, model, status, and cost.

### Remove background

1. User uploads image.
2. Backend uploads image to Runware.
3. Backend calls `removeBackground`.
4. Result `PNG` is saved locally.

### Replace background by prompt

1. User uploads image.
2. Backend removes background or creates a usable mask.
3. Backend runs inpainting with a new background prompt.
4. Result is saved locally.

### Replace background by second image

1. User uploads source image.
2. User uploads second background image.
3. Backend removes background from source image.
4. Backend composites foreground over the second image.
5. Result is saved locally.

### Upscale

1. User selects one existing result.
2. Backend sends it to Runware upscale.
3. Upscaled image is stored locally.
4. Database stores the extra cost.

### Vectorize

1. User uploads image or chooses an existing result.
2. Backend uploads the raster image to Runware.
3. Backend calls `taskType: vectorize` with `model: recraft:1@1`.
4. Returned `SVG` is downloaded and stored locally.

## Cost Model

Working assumptions used for planning:

- generation `1024x1024` with `FLUX.1 [dev]`: about `$0.0038`
- upscale to `1-4 MP`: about `$0.005`
- vectorization with `Recraft Vectorize`: about `$0.01`

### Cost per final stock image

If one final stock image usually takes:

- `3` generations + `1` upscale: about `$0.0164`
- `5` generations + `1` upscale: about `$0.024`
- `10` generations + `1` upscale: about `$0.043`

### Batch estimate

At `5` generations per accepted image:

- `100` final stock images: about `$2.40`
- `500` final stock images: about `$12.00`
- `1000` final stock images: about `$24.00`

Vectorization is separate and should only be used when the result truly benefits from SVG output.

## Adobe Stock Notes

- Final files should meet Adobe Stock minimum size requirements.
- Generated or AI-edited content should be labeled appropriately during submission.
- For stock workflow, keep source prompt history and final exported file paths.

## What Not To Build In v1

Do not add this yet:

- multi-user auth;
- role system;
- queue workers;
- remote storage;
- advanced admin panel;
- hybrid provider logic;
- automatic keyword generation.

The first version should stay focused and cheap.

## GitHub Status

Current status:

- local Git repository is initialized;
- no GitHub remote is configured yet;
- nothing has been pushed from this folder yet;
- GitHub CLI is not installed on this machine.

## Suggested Next Steps

1. Initialize a local Git repository.
2. Add `.gitignore`, `.env.example`, and `requirements.txt`.
3. Implement a Flask MVP with:
   - generation,
   - background removal,
   - background replacement,
   - upscale,
   - vectorize,
   - local history.
4. Connect the repository to GitHub and push the first commit.
