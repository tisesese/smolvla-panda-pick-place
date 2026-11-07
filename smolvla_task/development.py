"""Small Step6 diagnostic on a training position, never a final test."""
from pathlib import Path

from .config import OUTPUT, SEED
from .runtime import read, setup, sha, write


def compare_windows():
    setup()
    from .model import load, processors
    from .rollout import run
    out=OUTPUT/"step6/execution_window_pilot_v1"
    assert not out.exists(),"Retain earlier diagnostic results"
    out.mkdir(parents=True)
    model=OUTPUT/"training/formal_20000_v1/checkpoints/020000/pretrained_model"
    write(out/"protocol.json",dict(model=model,model_sha256=sha(model/"model.safetensors"),
          stage=6,scope="One training position; execution-window diagnosis, not independent evaluation",
          episode=0,horizons=[25,50],noise_seed=SEED,
          changed_variable="Number of predicted native50Hz actions executed before observing again",
          action_executor_sha256=sha(Path(__file__).parent/"environment.py"),
          model_and_processor_fixed=True))
    p=load(model)
    pre,post,_=processors(p,model)
    results={}
    for horizon in [25,50]:
        result=run(model,episode=0,horizon=horizon,label=f"step6_20000_window{horizon}_pilot_v1",
                   noise_seed=SEED,bundle=(p,pre,post),scope="Step6 training-position diagnostic")
        results[str(horizon)]=result
        write(out/"results.json",results)
    write(out/"complete.json",dict(completed=True,stage=6,results=results,
          not_step6_acceptance=True,not_step8=True))

