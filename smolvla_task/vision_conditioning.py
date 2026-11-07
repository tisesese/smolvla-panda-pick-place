"""Measure visual goal conditioning without changing robot state or noise."""
import os
import time
from pathlib import Path

import numpy as np
import torch

from .config import EVIDENCE,OUTPUT
from .data import Demonstrations,split
from .model import batch,load,processors
from .runtime import read,setup,sha,write


def run(model,label):
    setup()
    out=OUTPUT/"step6"/label
    assert not out.exists()
    out.mkdir(parents=True)
    write(out/"launch.json",dict(pid=os.getpid(),started_unix=time.time(),diagnostic_only=True))
    positions=read(EVIDENCE/"plan.json")["positions"]
    train=split()["train_episodes"]
    episodes=sorted({min(train,key=lambda e:positions[e]["xyz_m"][j]) for j in [0,1]}|
                    {max(train,key=lambda e:positions[e]["xyz_m"][j]) for j in [0,1]}|{0,40})
    ds=Demonstrations(episodes)
    states=np.stack([ds.numeric[e]["state"][0] for e in episodes])
    assert float(np.max(abs(states-states[0])))<1e-7
    model=Path(model)
    policy=load(model,frozen_bf16=False)
    policy.config.num_steps=20
    pre,post,_=processors(policy,model)
    noise=torch.randn(1,policy.config.chunk_size,policy.config.max_action_dim,device="cuda")
    predictions=[];targets=[]
    for ep in episodes:
        policy.reset()
        sample=ds.sample(ep,0)
        with torch.inference_mode():
            prediction=post(policy.predict_action_chunk(batch([sample],pre),noise=noise)).cpu().numpy()[0]
        predictions.append(prediction)
        targets.append(ds.numeric[ep]["action"][:50])
        print('VISUAL_CONDITIONING_EP',ep,flush=True)
    pred=np.stack(predictions)[:,:,:7]
    target=np.stack(targets)[:,:,:7]
    a=pred-pred.mean(axis=0,keepdims=True)
    b=target-target.mean(axis=0,keepdims=True)
    gain=float(np.linalg.norm(a)/np.linalg.norm(b))
    cosine=float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)))
    result=dict(diagnostic_only=True,not_task_success_evidence=True,episodes=episodes,
                positions_xy=[positions[e]["xyz_m"][:2] for e in episodes],
                initial_robot_state_max_variation=float(np.max(abs(states-states[0]))),
                same_fixed_Gaussian_noise=True,precision="float32",flow_integration_steps=20,
                model=model,visual_action_variation_gain=gain,centered_action_cosine=cosine,
                per_joint_predicted_std_rad=pred[:,-1].std(axis=0).tolist(),
                per_joint_teacher_std_rad=target[:,-1].std(axis=0).tolist(),
                model_sha256=sha(model/"model.safetensors"),code_sha256=sha(Path(__file__)))
    np.savez_compressed(out/"actions.npz",prediction=pred,target=target,episodes=episodes)
    write(out/"result.json",result)
    print(result,flush=True)
