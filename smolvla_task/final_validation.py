"""Record the selected model's final held-out demonstration loss, without selection."""
import argparse
from pathlib import Path

from .runtime import setup,read,sha,write,guard
from .config import OUTPUT


def run(name):
    import numpy as np
    import torch
    from .data import Demonstrations,split
    from .model import load,processors,batch
    parent=OUTPUT/'target80'/name;selected=read(parent/'selected.json')
    complete=read(parent/'comparison120/complete.json')
    assert complete['completed'] and complete['target80_achieved']
    model=Path(selected['model']);assert sha(model/'model.safetensors')==selected['weights_sha256']
    workspace=Path(read(parent/'comparison120/protocol.json')['workspace_root'])
    dest=workspace/'artifacts'/f"第六步选模_{selected['steps']}步_v2"/'final-validation-loss.json'
    assert not dest.exists(),'Preserve the final fixed validation measurement'
    setup(74610240);guard()
    protocol=dict(seed=74610240,episodes=split()['validation_episodes'],fractions=[0,.2,.4,.6,.8,1.],
        action_stride=5,batch_size=4,model=model,weights_sha256=selected['weights_sha256'],
        use='Final record only; no checkpoint or parameter selection after formal tests')
    write(dest.with_name('final-validation-loss-protocol.json'),protocol)
    ds=Demonstrations(protocol['episodes'],action_stride=5)
    policy=load(model);pre,_,_=processors(policy,model);policy.eval()
    anchors=[(ep,round(fraction*(ds.numeric[ep]['n']-1))) for ep in protocol['episodes'] for fraction in protocol['fractions']]
    losses=[]
    for start in range(0,len(anchors),4):
        prepared=batch([ds.sample(ep,frame) for ep,frame in anchors[start:start+4]],pre)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):loss,_=policy(prepared)
        assert torch.isfinite(loss)
        losses.append(float(loss.detach()))
    assert len(anchors)==120 and len(losses)==30 and sha(model/'model.safetensors')==selected['weights_sha256']
    write(dest,dict(passed=True,mean_loss=float(np.mean(losses)),samples=120,episodes=20,batch_losses=losses,
        anchors=anchors,protocol_sha256=sha(dest.with_name('final-validation-loss-protocol.json')),
        weights_sha256=selected['weights_sha256'],no_optimizer_updates=True))
    print(dict(final_validation_samples=120,mean_loss=float(np.mean(losses))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='target80_finish_v1');run(parser.parse_args().name)
