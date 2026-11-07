"""Short interface/scene check; this is not a task-success evaluation."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');p.add_argument('--model',action='store_true');args=p.parse_args()
    from smolvla_task.runtime import setup,write
    from smolvla_task.config import IMAGE,SELECTED,OUTPUT
    from smolvla_task.environment import PandaTask
    setup()
    with PandaTask() as env:
        obs,_=env.reset(seed=20261023)
        assert obs['observation.state'].shape==(8,) and obs[IMAGE].shape==(480,640,3)
        assert abs(env.native.model.opt.timestep-.002)<1e-12
        before=float(env.native.data.time)
        action=env.native.initial_ctrl.copy()
        for _ in range(2):env.step(action,include_rgb=False)
        assert abs(float(env.native.data.time)-before-.04)<1e-9
        if args.preview:
            from PIL import Image
            Image.fromarray(obs[IMAGE]).save(ROOT/'assets/scene-preview.png')
    result={'scene':True,'state_dimensions':8,'RGB_shape':[480,640,3],'physics_Hz':500,'control_Hz':50,'model_check':False,'task_success_evaluation':False}
    if args.model:
        if not (SELECTED/'model.safetensors').exists():raise FileNotFoundError('Provide the final checkpoint with TASK4_MODEL; the weight is intentionally not in Git.')
        from smolvla_task.model import load,processors,batch
        from smolvla_task.rollout import policy_sample
        import torch
        policy=load(SELECTED);pre,post,_=processors(policy,SELECTED)
        prepared=batch([policy_sample(obs)],pre)
        assert prepared['observation.state'].shape[-1]==8
        assert policy._training_variant['action_stride']==5
        result.update(model_check=True,strict_weights_loaded=True,processor_batch_prepared=True)
        del policy,pre,post,prepared
        if torch.cuda.is_available():torch.cuda.empty_cache()
    write(OUTPUT/'checks/environment.json',result);print(json.dumps(result))

if __name__=='__main__':main()
