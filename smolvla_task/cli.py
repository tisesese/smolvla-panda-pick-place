"""Portable entry points for the published task-four project."""
import argparse

def main():
    parser=argparse.ArgumentParser(description='SmolVLA Panda pick-and-place research project')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train');p.add_argument('--name',required=True);p.add_argument('--scope',choices=['formal','single-episode'],required=True)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--until',type=int);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--episode',type=int,default=0);p.add_argument('--frames-per-episode',type=int,default=256)
    p=sub.add_parser('refine');p.add_argument('--name',required=True);p.add_argument('--until',type=int,required=True)
    p.add_argument('--variant',choices=['tail','crop','crop_vision','stride5','stride5_finish'],default='stride5_finish')
    p=sub.add_parser('rollout');p.add_argument('--model',default='selected');p.add_argument('--episode',type=int,default=32)
    p.add_argument('--label',required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--horizon',type=int,choices=range(1,51));p.add_argument('--light-scale',type=float,default=1.0)
    p=sub.add_parser('offline-evaluate');p.add_argument('--model',required=True);p.add_argument('--label',required=True)
    p.add_argument('--episode',type=int);p.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    if args.command=='train':
        from .train import train_run
        train_run(args.name,args.scope,args.steps,args.batch_size,args.until,args.episode,args.frames_per_episode)
    elif args.command=='refine':
        from .refinement import train
        train(args.name,args.until,args.variant)
    elif args.command=='rollout':
        from .config import SELECTED
        from .rollout import run
        run(SELECTED if args.model=='selected' else args.model,episode=args.episode,horizon=args.horizon,label=args.label,device=args.device,light_scale=args.light_scale)
    else:
        from .evaluate import offline
        offline(args.model,args.label,args.episode,args.batch_size)
