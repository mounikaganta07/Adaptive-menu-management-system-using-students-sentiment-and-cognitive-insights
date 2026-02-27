from fastapi import FastAPI
from pydantic import BaseModel
from src.config import Config
from src.pipeline import run_pipeline

app = FastAPI(title="Menu Management API", version="1.0")

class RunRequest(BaseModel):
    input_path: str | None = None
    output_dir: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run")
def run(req: RunRequest):
    cfg = Config(
        raw_data_path=req.input_path or Config().raw_data_path,
        output_dir=req.output_dir or Config().output_dir,
    )
    paths = run_pipeline(cfg)
    return {"paths": paths}
