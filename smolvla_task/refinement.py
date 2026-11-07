"""Step 6 only: auditable continuation and closed-loop checkpoint selection."""
import gc
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import EVIDENCE, OUTPUT, SEED
from .data import AnchorPool, Demonstrations, split
from .model import batch, load, processors, train_mode
from .runtime import append, guard, read, resources, setup, sha, write
from .train import save_checkpoint

PARENT_STEPS = int(os.environ.get("TASK4_REFINE_PARENT_STEPS","20000"))
TOTAL = int(os.environ.get("TASK4_REFINE_ADDITIONAL_STEPS","3000"))
SOURCE = Path(os.environ.get("TASK4_REFINE_SOURCE",str(OUTPUT/"training/formal_20000_v1/checkpoints/020000/pretrained_model")))
DATA_SEED = SEED+71+(PARENT_STEPS-20000)
TAIL = [f"model.vlm_with_expert.vlm.model.text_model.layers.{i}" for i in [12,13]]
VARIANT = dict(fp32_modules=TAIL, parent_optimizer_steps=PARENT_STEPS,
               description="Expert and state projection plus VLM text layers 12 and 13; vision frozen")
VARIANTS = dict(tail=VARIANT,crop=dict(**VARIANT,image_crop_xyxy=[240,140,480,380]))
VARIANTS["crop_vision"]=dict(**VARIANTS["crop"])
VARIANTS["crop_vision"]["fp32_modules"]=TAIL+[
    "model.vlm_with_expert.vlm.model.connector.modality_projection.proj",
    "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11"]
VARIANTS["crop_vision"]["description"]="Expert/state, text layers 12/13, visual connector and final vision block"
VARIANTS["stride5"]=dict(**VARIANTS["crop_vision"],action_stride=5,execution_window=10,sampling_plan="lookahead")
VARIANTS["stride5_finish"]=dict(**VARIANTS["stride5"])
VARIANTS["stride5_finish"].update(sampling_plan="completion",
    description="Continue the 27k policy with separate support/release/retract sampling and a lower learning rate")


def benchmark_filename(variant_name):
    if PARENT_STEPS!=20000:return f"refinement-benchmark-{variant_name}-parent{PARENT_STEPS}.json"
    return "refinement-benchmark.json" if variant_name=="tail" else f"refinement-benchmark-{variant_name}.json"


def configure(policy,variant=None):
    policy._training_variant = variant or getattr(policy,"_training_variant",VARIANT)
    modules=policy._training_variant["fp32_modules"]
    for name in modules:
        policy.get_submodule(name).float()
    train_mode(policy)
    tail_ids = {id(p) for name in modules for p in policy.get_submodule(name).parameters()}
    tail = [p for p in policy.parameters() if p.requires_grad and id(p) in tail_ids]
    head = [p for p in policy.parameters() if p.requires_grad and id(p) not in tail_ids]
    assert sum(p.numel() for p in head) == 99880992
    expected=19664640 if len(modules)==2 else 38548992
    assert sum(p.numel() for p in tail) == expected
    assert all(p.dtype == torch.float32 for p in head+tail)
    preset = policy.config.get_optimizer_preset()
    optimizer = torch.optim.AdamW([dict(params=head, lr=2e-5),dict(params=tail,lr=5e-6)],
                                 betas=preset.betas,eps=preset.eps,weight_decay=preset.weight_decay,foreach=False)
    return head+tail, optimizer


def benchmark(variant_name="tail"):
    setup(DATA_SEED-1)
    p=load(SOURCE)
    variant=VARIANTS[variant_name]
    params,opt=configure(p,variant)
    pre,_,_=processors(p,SOURCE)
    ds=Demonstrations([0],action_stride=variant.get("action_stride",1))
    size=4 if variant_name in ["crop_vision","stride5","stride5_finish"] else 8
    prepared=batch([ds.sample(0,250+i*8) for i in range(size)],pre)
    modules=variant["fp32_modules"]
    selected={name:next(v for v in p.get_submodule(name).parameters() if v.ndim>=2) for name in modules}
    before={name:v.detach().clone() for name,v in selected.items()}
    evidence=[]
    for i in range(3):
        t=time.perf_counter()
        train_mode(p)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):loss,_=p(prepared)
        loss.backward()
        gradients={name:float(v.grad.norm()) for name,v in selected.items()}
        assert all(np.isfinite(g) and g>0 for g in gradients.values())
        trainable_ids={id(v) for name in modules for v in p.get_submodule(name).parameters()}
        assert all(v.grad is None for v in p.model.vlm_with_expert.vlm.parameters() if id(v) not in trainable_ids)
        norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        row=dict(update=i+1,loss=float(loss.detach()),gradient_norm=float(norm),tail_gradients=gradients,
                 seconds=time.perf_counter()-t,resources=resources())
        evidence.append(row)
        print(row,flush=True)
    changes={name:int(torch.count_nonzero(selected[name].detach()-v)) for name,v in before.items()}
    assert all(v>0 for v in changes.values())
    filename=benchmark_filename(variant_name)
    write(OUTPUT/"step6"/filename,dict(passed=True,discarded_updates=3,variant=variant,
          trainable_parameters=sum(v.numel() for v in params),updated_tail_values=changes,updates=evidence))


def train(name,until,variant_name="tail"):
    setup(DATA_SEED)
    out=OUTPUT/"training"/name
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/"run.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    total=TOTAL
    variant=VARIANTS[variant_name]
    peak_head_lr=(2e-5 if variant_name=="tail" else 5e-5) if PARENT_STEPS==20000 else 1e-5
    bs=4 if variant_name in ["crop_vision","stride5","stride5_finish"] else 8
    accumulation=4 if variant_name in ["crop_vision","stride5","stride5_finish"] else 2
    peak_tail_lr=(1e-5 if variant_name=="crop_vision" else 5e-6) if PARENT_STEPS==20000 else 2.5e-6
    if variant_name=="stride5":peak_head_lr,peak_tail_lr=1e-4,1e-5
    if variant_name=="stride5_finish":peak_head_lr,peak_tail_lr=3e-5,3e-6
    protocol=dict(name=name,stage=6,source=SOURCE,source_sha256=sha(SOURCE/"model.safetensors"),
                  parent_steps=PARENT_STEPS,additional_steps=total,batch_size=bs,gradient_accumulation=accumulation,
                  effective_batch_size=bs*accumulation,train_episodes=split()["train_episodes"],
                  validation_episodes=split()["validation_episodes"],variant=variant,
                  peak_head_lr=peak_head_lr,peak_tail_lr=peak_tail_lr,seed=DATA_SEED,frames_per_episode=256,
                  sampling="128 uniform,64 descend/grasp,32 lift,32 support/release/retract per train episode",
                  policy_inputs=["RGB","measured 8D robot state","task instruction"],
                  loss="Unmodified official masked flow matching",stats_sha256=sha(EVIDENCE/"train-only-stats.json"))
    if PARENT_STEPS!=20000:protocol["optimizer_initialized_fresh"]=True
    if variant_name in ["stride5","stride5_finish"]:
        protocol.update(action_stride=5,predicted_target_hz=10,native_control_hz=50,
                        learning_rate_floor_factor=.05,
                        sampling="128 uniform plus32 each approach,descend/grasp,lift,support/release/retract",
                        deployment="Linear interpolation of predicted 100ms targets at original50Hz; replan each1s")
    if variant_name=="stride5_finish":
        protocol["sampling"]="64 uniform plus32 each approach,descend/grasp,lift,place_support,release,retract per Train80 episode"
    if (out/"protocol.json").exists():assert read(out/"protocol.json")==read_dict(protocol)
    else:write(out/"protocol.json",protocol)
    ds=Demonstrations(split()["train_episodes"],action_stride=variant.get("action_stride",1))
    pool=AnchorPool(ds,256,DATA_SEED,focus=variant.get("sampling_plan",True))
    write(out/"anchors.json",dict(anchors=pool.anchors,compressed_bytes=pool.compressed_bytes))
    rng=np.random.default_rng(DATA_SEED+1)
    order=[]
    while len(order)<total*bs*accumulation:order.extend(rng.permutation(len(pool)).tolist())
    order=np.asarray(order[:total*bs*accumulation],dtype=np.int32)
    if (out/"sampler.npy").exists():np.testing.assert_array_equal(order,np.load(out/"sampler.npy"))
    else:np.save(out/"sampler.npy",order)
    latest=read(out/"latest.json") if (out/"latest.json").exists() else None
    start=latest["step"] if latest else 0
    assert start<until<=total
    source=Path(latest["checkpoint"])/"pretrained_model" if latest else SOURCE
    policy=load(source)
    params,optimizer=configure(policy,variant)
    pre,post,_=processors(policy,source)
    if latest:
        state=torch.load(Path(latest["checkpoint"])/"training_state.pt",map_location="cpu",weights_only=False)
        assert state["step"]==start and state["sampler_cursor"]==start*bs*accumulation
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"]);torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
        gc.collect()
    import json
    journal=[json.loads(l)["step"] for l in (out/"updates.jsonl").read_text().splitlines()] if (out/"updates.jsonl").exists() else []
    assert journal==list(range(1,start+1)),"Uncheckpointed journal requires investigation"
    write(out/"launch.json",dict(pid=os.getpid(),starting_step=start,target=until,resources=resources()))
    stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    begin=time.perf_counter()
    recent=[]
    for index in range(start,until):
        if index%25==0:guard()
        tick=time.perf_counter()
        train_mode(policy)
        optimizer.zero_grad(set_to_none=True)
        floor=.05 if variant_name in ["stride5","stride5_finish"] else .2
        factor=min((index+1)/50.,1.)*(floor+(1-floor)*.5*(1+math.cos(math.pi*max(index-50,0)/(total-50))))
        for group,peak in zip(optimizer.param_groups,[peak_head_lr,peak_tail_lr]):group["lr"]=peak*factor
        losses=[]
        for micro in range(accumulation):
            cursor=(index*accumulation+micro)*bs
            ids=order[cursor:cursor+bs]
            prepared=batch([pool[int(i)] for i in ids],pre)
            with torch.autocast("cuda",dtype=torch.bfloat16):loss,_=policy(prepared)
            assert torch.isfinite(loss)
            (loss/accumulation).backward()
            losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        completed=index+1
        row=dict(step=completed,total_lineage_steps=PARENT_STEPS+completed,loss=float(np.mean(losses)),
                 gradient_norm=float(norm),head_lr=optimizer.param_groups[0]["lr"],
                 seconds=time.perf_counter()-tick,elapsed_seconds=time.perf_counter()-begin)
        append(out/"updates.jsonl",row)
        recent.append(row["loss"])
        write(out/"progress.json",dict(**row,target=total,stage=6,recent_loss=float(np.mean(recent[-25:])),resources=resources()))
        if completed%25==0 or completed==1:print(row,flush=True)
        if completed%250==0 or completed==until or stop[0]:
            save_checkpoint(out,policy,optimizer,pre,post,completed,sha(out/"protocol.json"),completed*bs*accumulation)
        if stop[0]:write(out/"stopped.json",dict(step=completed,clean_checkpoint=True));return
    write(out/f"through-{until:06d}.json",dict(completed=True,additional_steps=until,total_lineage_steps=PARENT_STEPS+until))


def read_dict(value):
    import json
    from .runtime import convert
    return json.loads(json.dumps(value,default=convert))


def validate(model,label):
    """Reuse weights while resetting policy, RNG and simulator for each trial."""
    from .rollout import run
    from .pipeline import aggregate
    setup()
    model=Path(model)
    policy=load(model)
    pre,post,_=processors(policy,model)
    horizon=getattr(policy,"_training_variant",{}).get("execution_window",25)
    rows=[]
    for ep in [32,33,34,35,36]:
        result=run(model,episode=ep,horizon=horizon,label=label,bundle=(policy,pre,post))
        rows.append(result)
        rejected=sum(not r["success"] for r in rows)>=2
        write(OUTPUT/"rollouts"/label/"validation-summary.json",dict(completed=len(rows),metrics=aggregate(rows),results=rows,
              early_screen_rejected=rejected,screening_rule="Two failures make >=4/5 impossible; passing candidates complete all five"))
        if rejected:break


def status():
    """Report live PIDs separately from historical launch and progress files."""
    import json
    def process(pid,name):
        file=Path(f"/proc/{pid}/status")
        if not file.exists():return dict(running=False,state="exited")
        state=next(line.split(":",1)[1].strip() for line in file.read_text().splitlines() if line.startswith("State:"))
        command=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
        matches="run_task.py" in command and name in command
        return dict(running=matches and not state.startswith(("T","Z")),state=state,identity_matches=matches)
    heldout=[]
    for directory in sorted((OUTPUT/"heldout").glob("*")):
        if not (directory/"protocol.json").exists():continue
        item=dict(name=directory.name)
        if (directory/"status.json").exists():item["status"]=read(directory/"status.json")
        if (directory/"progress.json").exists():
            progress=read(directory/"progress.json")
            item["completed_trials"]=progress["completed_trials"]
            item["expected_trials"]=progress["expected_trials"]
            if "status" in item:
                item["status"]["completed_trials"]=progress["completed_trials"]
                item["status"]["expected_trials"]=progress["expected_trials"]
        active=directory/"active-launch.json"
        if not active.exists():active=directory/"launch.json"
        if active.exists():item["process"]=process(read(active)["pid"],directory.name)
        heldout.append(item)
    improvements=[]
    for directory in sorted((OUTPUT/"target80").glob("*")):
        if not (directory/"protocol.json").exists():continue
        item=dict(name=directory.name)
        if (directory/"status.json").exists():item["status"]=read(directory/"status.json")
        active=directory/"active-launch.json"
        if not active.exists():active=directory/"launch.json"
        if active.exists():item["process"]=process(read(active)["pid"],directory.name)
        training=OUTPUT/"training"/(directory.name+"_training")/"progress.json"
        if training.exists():item["training"]=read(training)
        if (directory/"independent50/complete.json").exists():item["new_test"]=read(directory/"independent50/complete.json")
        improvements.append(item)
    result=dict(stage=6,step8_enabled=bool(heldout),heldout_tests=heldout,branches={},target80_improvements=improvements)
    for name in ["step6_tail_refinement_v1","step6_crop_tail_refinement_v1","step6_crop_vision_refinement_v1","step6_crop_vision_long_v2","step6_stride5_vision_v1"]:
        out=OUTPUT/"step6"/name
        training=OUTPUT/"training"/name
        if not out.exists():continue
        branch={}
        for key,path in [("pipeline",out/"status.json"),("training",training/"progress.json"),
                         ("selected",out/"selected.json"),("error",out/"error.json")]:
            if path.exists():branch[key]=read(path)
        for key,root in [("scheduler",out),("trainer",training)]:
            if (root/"launch.json").exists():
                pid=read(root/"launch.json")["pid"]
                branch[f"{key}_pid"]=pid
                info=process(pid,name)
                branch[f"{key}_running"]=info["running"]
                branch[f"{key}_state"]=info["state"]
        manual=out/"execution-comparison-progress.json"
        if manual.exists():
            item=read(manual)
            branch["execution_comparison"]=dict(**item,process=process(item["pid"],name))
        action=branch.get("pipeline",{}).get("action","")
        if action=="closed-loop validation":
            steps=branch["pipeline"]["candidate_steps"]
            label=f"{name}_step{steps}"
            root=OUTPUT/"rollouts"/label
            branch["closed_loop_results"]=[dict(episode=r["episode"],success=r["success"],reason=r["reason"])
               for r in (read(p) for p in sorted(root.glob("*/result.json")))]
        result["branches"][name]=branch
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def launch(name,variant_name="tail"):
    setup()
    out=OUTPUT/"step6"/name
    assert not out.exists()
    code=out/"code"
    code.mkdir(parents=True)
    root=Path(__file__).resolve().parent.parent
    shutil.copytree(root/"smolvla_task",code/"smolvla_task",ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(root/"run_task.py",code/"run_task.py")
    write(out/"code-sha256.json",{str(p.relative_to(code)):sha(p) for p in code.rglob("*.py")})
    with (out/"pipeline.log").open("a") as log:
        proc=subprocess.Popen([sys.executable,"-u",str(code/"run_task.py"),"refinement-pipeline","--name",name,"--variant",variant_name],
                              cwd=code,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write(out/"launch.json",dict(pid=proc.pid,stage=6,name=name,log=out/"pipeline.log",source=SOURCE,variant=variant_name,
          parent_steps=PARENT_STEPS,additional_steps=TOTAL))
    print(read(out/"launch.json"),flush=True)


def pipeline(name,variant_name="tail"):
    """Bounded training/checkpoint comparisons. Does not contain any step 8 call."""
    from .pipeline import child,aggregate
    out=OUTPUT/"step6"/name
    try:
        write(out/"status.json",dict(stage=6,action="gradient and memory benchmark",step8_enabled=False))
        evidence_path=OUTPUT/"step6"/benchmark_filename(variant_name)
        if not evidence_path.exists():child("refinement-benchmark","--variant",variant_name)
        evidence=read(evidence_path)
        assert evidence["passed"] and evidence["variant"]==read_dict(VARIANTS[variant_name])
        comparisons=[]
        milestones=[500,1000,2000,3000] if TOTAL==3000 else sorted({s for s in [1000,3000,5000,TOTAL] if s<=TOTAL})
        if variant_name=="stride5":milestones=sorted({s for s in [500,1000,2000,3500,TOTAL] if s<=TOTAL})
        for milestone in milestones:
            write(out/"status.json",dict(stage=6,action="continuation training",additional_target=milestone,step8_enabled=False))
            child("refine","--name",name,"--until",milestone,"--variant",variant_name)
            checkpoint=OUTPUT/"training"/name/"checkpoints"/f"{milestone:06d}"
            child("verify-checkpoint",checkpoint)
            model=checkpoint/"pretrained_model"
            # Keep every compared model with hard links before recovery rotation.
            candidate=out/"candidates"/f"{PARENT_STEPS+milestone:06d}"
            candidate.mkdir(parents=True)
            shutil.copytree(model,candidate/"pretrained_model",copy_function=os.link)
            write(candidate/"lineage.json",dict(parent_steps=PARENT_STEPS,additional_steps=milestone,
                  source=SOURCE,weights_sha256=sha(model/"model.safetensors")))
            write(out/"status.json",dict(stage=6,action="closed-loop validation",candidate_steps=PARENT_STEPS+milestone,step8_enabled=False))
            label=f"{name}_step{PARENT_STEPS+milestone}"
            child("validate-candidate","--model",candidate/"pretrained_model","--label",label)
            rows=read(OUTPUT/"rollouts"/label/"validation-summary.json")["results"]
            write(out/"validation-progress.json",dict(candidate_steps=PARENT_STEPS+milestone,completed=len(rows),successes=sum(r["success"] for r in rows)))
            summary=aggregate(rows)
            record=dict(total_lineage_steps=PARENT_STEPS+milestone,model=candidate/"pretrained_model",validation=summary,results=rows)
            comparisons.append(record)
            write(out/"checkpoint-comparison.json",comparisons)
            if summary["successes"]>=4:
                write(out/"selected.json",dict(**record,stage6_complete=True,gate="Development target >=4/5 complete tasks; not a PPT requirement or population estimate",step8_started=False))
                write(out/"status.json",dict(stage=6,action="selected candidate passes development gate",selected_steps=PARENT_STEPS+milestone,step8_enabled=False))
                return
        write(out/"status.json",dict(stage=6,action="all continuation candidates evaluated; further diagnosis required",stage6_complete=False,step8_enabled=False))
    except Exception as exc:
        import traceback
        write(out/"error.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        write(out/"status.json",dict(stage=6,action="stopped with error",stage6_complete=False,step8_enabled=False))
        raise
