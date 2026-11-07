"""A single immediately running 20,000-step experiment and subsequent evaluation."""
import gc
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .config import BASE, EVIDENCE, OLD, OUTPUT, SEED
from .runtime import append, guard, read, setup, sha, write


def launch(name):
    """Snapshot code so edits in the Windows workspace cannot change this run."""
    setup()
    guard()
    out = OUTPUT/"runs"/name
    assert not out.exists(), "Use an existing launch record to inspect/resume; never silently replace a run"
    code = out/"code"
    code.mkdir(parents=True)
    project = Path(__file__).resolve().parent.parent
    shutil.copytree(project/"smolvla_task", code/"smolvla_task", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(project/"run_task.py", code/"run_task.py")
    hashes = {str(p.relative_to(code)): sha(p) for p in code.rglob("*.py")}
    write(out/"code-sha256.json", hashes)
    log = (out/"pipeline.log").open("a")
    child = subprocess.Popen([sys.executable, "-u", str(code/"run_task.py"), "formal-pipeline", "--name", name],
                             cwd=code, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    log.close()
    write(out/"launch.json", dict(pid=child.pid, name=name, started_unix=time.time(), target_steps=20000,
                                 code=code, log=out/"pipeline.log", initialization=BASE,
                                 new_formal_updates=20000, diagnostic_updates_included=0))
    print(read(out/"launch.json"), flush=True)


def child(entry, *args):
    code = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, "-u", str(code/"run_task.py"), entry, *map(str, args)], cwd=code)
    assert result.returncode == 0, f"{entry} exited {result.returncode}; preserve the log and diagnose"


def status(name):
    import json
    result = {}
    for key, path in [("pipeline", OUTPUT/"runs"/name/"status.json"),
                      ("training", OUTPUT/"training"/name/"progress.json"),
                      ("evaluation", OUTPUT/"runs"/name/"evaluation-progress.json"),
                      ("complete", OUTPUT/"runs"/name/"complete.json"),
                      ("error", OUTPUT/"runs"/name/"error.json")]:
        if path.exists():
            result[key] = read(path)
    journal = OUTPUT/"training"/name/"updates.jsonl"
    if journal.exists():
        lines = journal.read_text().splitlines()[-50:]
        rows = [json.loads(line) for line in lines]
        seconds = float(np.mean([r["update_seconds"] for r in rows]))
        result["estimated_training_hours_remaining"] = max(0,20000-rows[-1]["step"])*seconds/3600
        result["recent_seconds_per_update"] = seconds
    if "error" in result and result.get("pipeline",{}).get("stage") != "stopped_with_error":
        result["previous_error"] = result.pop("error")
    repair_error=OUTPUT/"runs"/name/"evaluation-repair-v1/error.json"
    if repair_error.exists():
        key="previous_evaluation_stop" if (repair_error.parent/"user-stop-complete.json").exists() else "evaluation_repair_error"
        result[key]=read(repair_error)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def aggregate(rows):
    assert rows
    successes = sum(r["success"] for r in rows)
    return dict(attempts=len(rows), successes=successes, success_rate=successes/len(rows),
                successful_sim_time_mean_s=float(np.mean([r["sim_seconds"] for r in rows if r["success"]])) if successes else None,
                inference_mean_s=float(np.mean([r["inference_mean_seconds"] for r in rows])),
                dropped=sum(r["dropped"] for r in rows), collisions=sum(r["collision"] for r in rows),
                reasons={reason:sum(str(r["reason"])==reason for r in rows) for reason in sorted({str(r["reason"]) for r in rows})})


def fresh_positions():
    """Exclude registered actual XY positions, not just seed numbers."""
    paths = [EVIDENCE/"plan.json", OLD/"outputs/panda_formal_vla50_v1/protocol.json",
             OLD/"outputs/panda_history_finetune_v1/registry.json",
             OLD/"outputs/panda_deployment_scale_v1/data/plan.json",
             OLD/"outputs/panda_grasp_sprint_v1/data/plan.json"]
    excluded = []
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ["xyz_m", "cube_xyz_m", "position_m"] and isinstance(item, list) and len(item) == 3 and all(isinstance(x,(float,int)) for x in item):
                    excluded.append(np.asarray(item[:2], float))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    for path in paths:
        if path.exists():
            visit(read(path))
    rng = np.random.default_rng(20261023921)
    result = []
    for index in range(50):
        while True:
            x, y = rng.uniform(.105,.155), rng.uniform(-.215,-.165)
            if index >= 40:
                edge = (index-40)%4
                if edge == 0:x=.105
                elif edge == 1:x=.155
                elif edge == 2:y=-.215
                else:y=-.165
            xy = np.asarray([x,y])
            if all(np.linalg.norm(xy-p)>1e-6 for p in excluded):
                break
        excluded.append(xy)
        result.append(dict(seed=2026102300+index, xyz_m=[float(x),float(y),.782],
                           category="boundary" if index>=40 else "interior"))
    return result


def evaluate_after_training(name, checkpoint):
    from .model import load, processors
    from .rollout import run
    out = OUTPUT/"runs"/name
    model = checkpoint/"pretrained_model"
    # Validation observes the declared20k model. It cannot shorten training or
    # change the frozen executor used by the independent paired test below.
    child("offline-evaluate", "--model", model, "--label", name+"-validation20")
    division = read(EVIDENCE/"split.json")
    validation = []
    p = load(model)
    pre, post, _ = processors(p, model)
    for ep in division["validation_episodes"][:5]:
        validation.append(run(model, episode=ep, horizon=25, label=name+"-validation",
                              bundle=(p,pre,post), noise_seed=SEED+ep))
    del p,pre,post
    gc.collect()
    torch.cuda.empty_cache()
    metrics = aggregate(validation)
    write(out/"closed-loop-validation.json", metrics)
    from .acceptance import stage6_admission
    admission = stage6_admission(metrics)
    write(out/"step6-admission.json", {**admission,"model":model,"model_sha256":sha(model/"model.safetensors")})
    if not admission["passed"]:
        write(out/"status.json",dict(stage="step6_diagnosis",training_steps=20000,step6_complete=False,
              reason="Training completed, but closed-loop model selection has not passed",step8_started=False))
        print("STEP6_NOT_PASSED",admission,flush=True)
        return False
    freeze = dict(model=model, model_sha256=sha(model/"model.safetensors"),
                  base=BASE, base_sha256=sha(BASE/"model.safetensors"), training_steps=20000,
                  no_intermediate_checkpoint_selection=True,
                  action_executor_sha256=sha(Path(__file__).parent/"environment.py"),
                  horizon=25, initial_neutral_seconds=1, timeout_sim_seconds=30, stable_seconds=2,
                  noise_seed_base=SEED, positions=fresh_positions(),
                  brightness_stress=dict(indices=list(range(0,50,5)), multiplier=.9,
                                          description="RGB intensity transformation only; no physical illumination or geometry change"))
    write(out/"frozen-evaluation-protocol.json", freeze)
    all_rows = {}
    for label, directory in [("base",BASE), ("finetuned20000",model)]:
        p=load(directory)
        pre,post,_=processors(p,directory)
        for lighting, indices, factor in [("normal",range(50),1.), ("dim",range(0,50,5),.9)]:
            rows=[]
            for index in indices:
                position=freeze["positions"][index]
                row=run(directory, horizon=25, label=name+"-test-"+label+"-"+lighting,
                        noise_seed=SEED+index, position=position, bundle=(p,pre,post),
                        scope="independent held-out paired test", brightness=factor)
                rows.append(row)
                append(out/"test-results.jsonl", {**row, "model":label, "lighting":lighting, "index":index})
                write(out/"evaluation-progress.json", dict(model=label, lighting=lighting,
                      completed=len(rows), expected=len(list(indices)), metrics=aggregate(rows)))
            all_rows[label+"-"+lighting]=aggregate(rows)
        del p,pre,post
        gc.collect()
        torch.cuda.empty_cache()
    result=dict(completed=True, training_steps=20000, metrics=all_rows,
                task_has_autonomous_success=all_rows["finetuned20000-normal"]["successes"]>0,
                task_meets_suggested80percent=all_rows["finetuned20000-normal"]["success_rate"]>=.8,
                frozen_protocol_sha256=sha(out/"frozen-evaluation-protocol.json"))
    write(out/"complete.json",result)
    print("FORMAL_COMPLETE",result,flush=True)
    return True


def formal_pipeline(name):
    setup()
    out=OUTPUT/"runs"/name
    out.mkdir(parents=True,exist_ok=True)
    try:
        write(out/"status.json",dict(stage="training",target=20000,started_unix=time.time()))
        child("train", "--name",name,"--scope","formal","--steps",20000,
              "--batch-size",16,"--frames-per-episode",256)
        checkpoint=OUTPUT/"training"/name/"checkpoints/020000"
        assert checkpoint.exists(),"Stopped training is not a20k completion"
        write(out/"status.json",dict(stage="strict_reload",training_steps=20000))
        child("verify-checkpoint",checkpoint)
        write(out/"status.json",dict(stage="evaluation",training_steps=20000))
        if evaluate_after_training(name,checkpoint):
            write(out/"status.json",dict(stage="complete",training_steps=20000))
    except BaseException:
        write(out/"error.json",dict(traceback=traceback.format_exc(),unix_s=time.time(),automatic_restart=False))
        write(out/"status.json",dict(stage="stopped_with_error",target=20000))
        raise
