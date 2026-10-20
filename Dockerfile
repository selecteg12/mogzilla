FROM node:24-bookworm-slim AS frontend
WORKDIR /build/frontend
RUN npm install --global pnpm@11.25.0
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm run test:camera && pnpm run build

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PORT=8000
WORKDIR /app
COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-runtime.txt
COPY backend/ ./backend/
COPY scripts/serve.py ./scripts/serve.py
COPY training/checkpoints/selected.pt training/checkpoints/regions-selected.pt ./training/checkpoints/
COPY reports/run/evaluation.json ./reports/run/evaluation.json
COPY reports/reference-values/examples.json reports/reference-values/example-1.jpg reports/reference-values/example-2.jpg reports/reference-values/example-3.jpg ./reports/reference-values/
COPY --from=frontend /build/frontend/dist/ ./frontend/dist/
RUN useradd --create-home --uid 10001 appuser
USER appuser
RUN python -m scripts.serve --check
EXPOSE 8000
CMD ["python", "-m", "scripts.serve"]
