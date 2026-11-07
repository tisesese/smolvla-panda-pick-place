"""Finish the frozen paired test after a documented numerical endpoint fix."""
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .config import OUTPUT, SEED
from .runtime import append, guard, read, setup, sha, write


def launch(name):
    setup()
    guard()
    parent=OUTPUT/"runs"/name
    repair=parent/"evaluation-repair-v1"
    admission=parent/"step6-admission.json"
    assert admission.exists() and read(admission)["passed"],"Finish Step6 closed-loop model selection before restarting Step8"
    assert read(OUTPUT/"evaluation-boundary-repair-verification.json")["passed"]
    assert not repair.exists(),"Preserve prior repair attempts"
    code=repair/"code"
    code.mkdir(parents=True)
    source=Path(__file__).resolve().parent.parent
    shutil.copytree(source/"smolvla_task",code/"smolvla_task",ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(source/"run_task.py",code/"run_task.py")
    write(repair/"code-sha256.json",{str(p.relative_to(code)):sha(p) for p in code.rglob("*.py")})
    with (repair/"evaluation.log").open("a") as log:
        p=subprocess.Popen([sys.executable,"-u",str(code/"run_task.py"),"resume-evaluation","--name",name],
             cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(repair/"launch.json",dict(pid=p.pid,started_unix=time.time(),log=repair/"evaluation.log"))
    write(parent/"status.json",dict(stage="evaluation",training_steps=20000,evaluation_revision="float32-boundary-repair-v1"))
    print(read(repair/"launch.json"),flush=True)


def run(name):
    setup()
    from .model import load,processors
    from .pipeline import aggregate
    from .rollout import run as rollout
    parent=OUTPUT/"runs"/name
    out=parent/"evaluation-repair-v1"
    try:
        assert read(parent/"step6-admission.json")["passed"],"Step6 has not passed"
        frozen=read(parent/"frozen-evaluation-protocol.json")
        model=Path(frozen["model"])
        base=Path(frozen["base"])
        assert sha(model/"model.safetensors")==frozen["model_sha256"]
        assert sha(base/"model.safetensors")==frozen["base_sha256"]
        protocol={**frozen,"prior_protocol_sha256":sha(parent/"frozen-evaluation-protocol.json"),
                  "prior_action_executor_sha256":frozen["action_executor_sha256"],
                  "action_executor_sha256":sha(Path(__file__).parent/"environment.py"),
                  "numerical_repair":read(OUTPUT/"evaluation-boundary-repair-verification.json"),
                  "test_restarted_under_repaired_executor_for_both_models":True,
                  "old_completed_and_interrupted_trials_retained_separately":True,
                  "no_model_or_hyperparameter_selection_from_test":True}
        write(out/"protocol.json",protocol)
        bundles={}
        for label,directory in [("finetuned20000",model),("base",base)]:
            p=load(directory)
            pre,post,_=processors(p,directory)
            bundles[label]=(directory,(p,pre,post))
        results={}
        for lighting,indices,brightness in [("normal",list(range(50)),1.),("dim",list(range(0,50,5)),.9)]:
            for label in bundles:
                results[label+"-"+lighting]=[]
            for index in indices:
                for label,(directory,bundle) in bundles.items():
                    row=rollout(directory,horizon=frozen["horizon"],
                         label=name+"-repaired-"+label+"-"+lighting,
                         noise_seed=frozen["noise_seed_base"]+index,
                         position=frozen["positions"][index],bundle=bundle,
                         scope="frozen paired test, numerical endpoint repair",brightness=brightness)
                    results[label+"-"+lighting].append(row)
                    append(out/"results.jsonl",{**row,"model_condition":label,"lighting":lighting,"index":index})
                    metrics={key:aggregate(rows) for key,rows in results.items() if rows}
                    write(parent/"evaluation-progress.json",dict(revision="float32-boundary-repair-v1",
                          completed_trials=sum(map(len,results.values())),expected_trials=120,metrics=metrics))
        metrics={key:aggregate(rows) for key,rows in results.items()}
        result=dict(completed=True,training_steps=20000,metrics=metrics,
             task_has_autonomous_success=metrics["finetuned20000-normal"]["successes"]>0,
             task_meets_suggested80percent=metrics["finetuned20000-normal"]["success_rate"]>=.8,
             evaluation_revision="float32-boundary-repair-v1",frozen_protocol_sha256=sha(out/"protocol.json"))
        write(out/"complete.json",result)
        write(parent/"complete.json",result)
        write(parent/"status.json",dict(stage="complete",training_steps=20000,evaluation_revision="float32-boundary-repair-v1"))
        print("EVALUATION_COMPLETE",result,flush=True)
    except BaseException:
        write(out/"error.json",dict(traceback=traceback.format_exc(),unix_s=time.time(),automatic_restart=False))
        write(parent/"status.json",dict(stage="stopped_with_error",training_steps=20000,evaluation_revision="float32-boundary-repair-v1"))
        raise

