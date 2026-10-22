# mogzilla

A Python-based portrait assessment app with a looksmaxxing-inspired 1–10 score, 478 facial landmarks and an interactive breakdown of eight facial regions. Upload a portrait or use the camera, inspect measurements and formulas, and explore example results from **sub 3** to **true adam**.

The interface uses a light editorial design with a custom animated-reaction mascot. Image inference runs locally; the UI is currently in Russian.

## What is looksmaxxing?

**Looksmaxxing** is an online culture focused on improving appearance through grooming, styling and other changes. Its male beauty discussions often favour a model-like aesthetic and use their own vocabulary for facial proportions and overall presentation. These are community preferences, not a universal or scientifically established beauty standard. [Research on the community](https://doi.org/10.1177/1357034X251363787) describes these ideals and their social context.

Commonly discussed criteria include:

| Area | Features often favoured in these communities |
| --- | --- |
| Eyes and brows | Almond-shaped or hooded “hunter eyes”, a neutral-to-positive outer-corner tilt, low-set and defined brows |
| Jaw and chin | A defined jaw contour and a balanced, prominent chin |
| Cheekbones and proportions | Visible cheekbone definition and balanced relationships between the facial regions |
| Harmony and symmetry | A coherent combination of features and limited visible asymmetry |
| Presentation | Clear-looking skin, well-kept hair and effective grooming |

mogzilla borrows this vocabulary for its presentation and category names. Aesthetic labels are subjective; the app does not establish a person's worth, health, body-fat percentage or suitability for modelling.

## Features

- Photo upload, drag-and-drop and camera capture with front/rear camera switching.
- A visual score, a named tier and a mascot reaction.
- Eight interactive regions: eyes, brows, upper face, midface, nose/mouth, jaw/chin, symmetry and skin/lighting.
- Landmark overlays, measured segments, formulas and coordinate-level explanations.
- Technical checks for face size, pose, cropping, blur, lighting and closed eyes.
- Crop/reflection sensitivity checks and JSON export without the uploaded photograph.
- An example gallery accessible from **Examples** in the top navigation and the homepage poster.

## Data and annotations

Additional portrait annotations were supplied as **model-scout ratings on a 1–10 scale**. They are separate from the original crowd-average labels in the source datasets. The available exports do not establish agreement across a representative panel of scouts.

| Source | Role in the project |
| --- | --- |
| [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) | Human-rated portraits underlying the initial visual scoring model |
| [MEBeauty](https://huggingface.co/datasets/dr-irina-lebedeva/MEBeauty-Facial-Beauty-Prediction) | Additional photographs with project-specific scout annotations |
| [Pinterest portrait collection](data/README.md) | Supplied digitals and other portraits, with confirmed scout ratings; available as a separate archive |
| [WFLW](https://github.com/wywu/LAB) | Original landmark-model training and a technical audit of geometric measurements |
| [AFW / 300-W](https://ibug.doc.ic.ac.uk/resources/300-W/) | An earlier 68-landmark experiment using the AFW subset |
| [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) | Additional face-detection training data in an earlier version |

WFLW, AFW and WIDER FACE do not provide attractiveness targets for the current score. The Pinterest archive excludes photographs from the research datasets. Third-party photographs retain their original rights; no unrestricted image license is implied by this repository.

## How the result is produced

Python decodes the photograph, locates a face, checks technical suitability and calculates the visual score from face and wider-head crops. Landmark-based geometry is calculated separately. The geometric measurements describe the photograph; they are not a causal explanation of the image model's score.

The current trained scoring profile is revision `33c59a04cba4`. The repository includes its scoring head and updated visual block; setup downloads pinned upstream model assets and verifies their SHA-256 hashes. Model provenance, preprocessing, artifact hashes and limitations are documented in [the scoring report](reports/SCORING.md).

Tier names are project-defined intervals applied to the displayed score. They are not percentiles or a separate classifier. The displayed crop range describes sensitivity on one image, not a confidence interval.

## Validation

The latest usable annotated collection contains **717 fitting photographs and 200 held-out photographs**. On the held-out set, MAE is **1.20 points out of 10**, with **51%** of predictions within one point of the annotation.

The evaluation is limited: some held-out results were inspected in earlier experiments, full separation by person is not established, and overlap with upstream datasets cannot be ruled out. A small selected collection does not establish universal ranking accuracy. See [the scoring report](reports/SCORING.md) and [the technical geometry audit](reports/HOLISTIC.md).

## Run locally

Requirements: **Python 3.12**, **Node.js 22.12+** and **pnpm**. The scripts can also discover the bundled Codex runtimes on Windows.

```powershell
git clone https://github.com/selecteg12/facemetrics-ai.git
cd facemetrics-ai
git switch codex/holistic-rating
.\setup.ps1
.\start-mogzilla.ps1
```

Open **http://127.0.0.1:8008**. Stop the server with `Ctrl+C`. The trained profile is loaded from `training/checkpoints/holistic`; setup downloads approximately 98 MB of upstream assets to the same directory. Dependency versions are pinned in `requirements-lock.txt` and `frontend/pnpm-lock.yaml`.

For frontend development, run `start.ps1 -Port 8001`, then `pnpm dev` in `frontend/`. The development proxy targets port 8001. The repository's earlier FaceMetrics implementation remains in the `master` branch.

## Checks

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q --basetemp=.test-tmp
cd frontend
pnpm test:camera
pnpm test:rating
pnpm test:regions
pnpm build
```

Tests requiring undistributed research samples skip when those samples are absent. Scripts in `training/` cover the earlier landmark/detection experiments; scripts in `scripts/` cover scoring-profile preparation, fitting and evaluation.

## Project layout

```text
backend/       FastAPI, neural inference, geometry and tests
frontend/      React + TypeScript UI, mascot and example gallery
training/      Training code and selected scoring checkpoints
data/          Pinterest portrait archive and provenance notes
scripts/       Setup, evaluation, annotation and serving utilities
reports/       Dataset notes, experiments and measured limitations
tools/         Local portrait annotation tool
```

## Privacy and research use

Uploaded photographs are processed in memory. The app does not automatically save uploads, call external AI APIs, maintain a face database or perform identity recognition. Camera access requires the browser's permission. Photo uploads are limited to JPEG, PNG or WebP, up to 10 MB and 20 megapixels.

SCUT-FBP5500, MEBeauty and upstream model assets retain their research-use conditions. This is a non-commercial research project; public repository visibility does not replace the licenses of datasets, weights, photographs or dependencies. The Pinterest collection is credited to its supplied source, with original URLs preserved where available.

Docker and Vercel configuration files are retained from earlier iterations. The current scoring release is verified locally; those files alone do not establish a verified hosted deployment of this revision.
