import os

# Limit glibc heap arenas and TensorFlow CPU threads before workers fork
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# IMPORTANT: Do NOT use Render's default WEB_CONCURRENCY (which defaults to >=2 and
# causes >512 MB OOM SIGKILL when loading TensorFlow twice on Render Free Tier).
workers = int(os.environ.get("GUNICORN_WORKERS", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "1"))
worker_class = "sync"
preload_app = False

# Allow 120s for cold-start CPU throttling on Render Free Tier (0.1 vCPU)
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5
max_requests = 0

accesslog = "-"
errorlog = "-"
capture_output = True
loglevel = "info"


def post_worker_init(worker):
    print(
        f"[HemoScan][GUNICORN][pid={worker.pid}] Worker initialized "
        f"(workers={workers}, threads={threads}, timeout={timeout}s)",
        flush=True,
    )


def worker_int(worker):
    print(f"[HemoScan][GUNICORN][pid={worker.pid}] Worker received SIGINT/SIGQUIT", flush=True)


def worker_abort(worker):
    print(
        f"[HemoScan][GUNICORN][pid={worker.pid}] Worker received SIGABRT (WORKER TIMEOUT)",
        flush=True,
    )
