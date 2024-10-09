"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time
import copy
import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util import (
    train_defaults,
    model_and_diffusion_defaults,
    cm_train_defaults,
    ctm_train_defaults,
    ctm_eval_defaults,
    ctm_loss_defaults,
    ctm_data_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.sample_util import karras_sample
import blobfile as bf
from torchvision.utils import make_grid, save_image
#import classifier_lib


def main():
    args = create_argparser().parse_args()

    if args.use_MPI:
        dist_util.setup_dist(args.device_id)
    else:
        dist_util.setup_dist_without_MPI(args.device_id)

    logger.configure(args, dir=args.out_dir)

    logger.log("creating model and diffusion...")

    if args.training_mode == 'edm':
        model, diffusion = create_model_and_diffusion(args, teacher=True)
    else:
        model, diffusion = create_model_and_diffusion(args)

    try:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location=dist_util.dev())
        )
    except:
        try:
            model.load_state_dict(
                dist_util.load_state_dict(args.model_path, map_location='cpu')
            )
        except:
            print("model path not loaded")
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    elif args.sampler in ["exact", "gamma", "cm_multistep", "gamma_multistep"]:
        try:
            ts = tuple(int(x) for x in args.ts.split(","))
        except:
            ts = []
    else:
        ts = None

    #for ind_1 in range(1,18):
    #    for ind_2 in range(ind_1+1):
    #        print("ind_1, ind_2: ", ind_1, ind_2)
    if args.stochastic_seed:
        args.eval_seed = np.random.randint(1000000)
    #generator = get_generator(args.generator, args.num_samples, args.seed)
    generator = get_generator(args.generator, args.eval_num_samples, args.eval_seed)

    step = args.model_path.split('.')[-2][-6:]
    try:
        ema = float(args.model_path.split('_')[-2])
        assert ema in [0.999, 0.9999, 0.9999432189950708]
    except:
        ema = 'model'
    if args.sampler in ['multistep', 'exact', 'cm_multistep']:
        out_dir = os.path.join(args.out_dir, f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{"".join([str(i) for i in ts])}')
    elif args.sampler in ["gamma"]:
        out_dir = os.path.join(args.out_dir, f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{"".join([str(i) for i in ts])}_ind1_{args.ind_1}_ind2_{args.ind_2}')
    elif args.sampler in ["gamma_multistep"]:
        out_dir = os.path.join(args.out_dir,
                               f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{"".join([str(i) for i in ts])}_gamma_{args.gamma}')

    else:
        out_dir = os.path.join(args.out_dir,
                                f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema')
    os.makedirs(out_dir, exist_ok=True)
    itr = 0
    eval_num_samples = 0
    while itr * args.batch_size < args.eval_num_samples:
        x_T = generator.randn(
            *(args.batch_size, args.in_channels, args.image_size, args.image_size),
            device=dist_util.dev()) * args.sigma_max
        #classes = generator.randint(0, 1000, (args.batch_size,))
        if args.large_log:
            print("x_T: ", x_T[0][0][0][0])
        current = time.time()
        model_kwargs = {}
        if args.class_cond:
            if args.train_classes >= 0:
                classes = th.ones(size=(args.batch_size,), device=dist_util.dev(), dtype=int) * int(args.train_classes)
            elif args.train_classes == -2:
                classes = [0, 1, 9, 11, 29, 31, 33, 55, 76, 89, 90, 130, 207, 250, 279, 281, 291, 323, 386, 387,
                           388, 417, 562, 614, 759, 789, 800, 812, 848, 933, 973, 980]
                assert args.batch_size % len(classes) == 0
                #print("!!!!!!!!!!!!!!: ", [x for x in classes for _ in range(args.batch_size // len(classes))])
                #model_kwargs["y"] = th.from_numpy(np.array([[[x] * (args.batch_size // len(classes)) for x in classes]]).reshape(-1)).to(dist_util.dev())
                classes = th.tensor([x for x in classes for _ in range(args.batch_size // len(classes))], device=dist_util.dev())
            else:
                classes = th.randint(
                    low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
                )
            model_kwargs["y"] = classes
            if args.large_log:
                print("classes: ", model_kwargs)
        with th.no_grad():
            x = karras_sample(
                diffusion=diffusion,
                model=model,
                shape=(args.batch_size, args.in_channels, args.image_size, args.image_size),
                steps=args.sampling_steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=False if args.data_name in ['church'] else True if args.training_mode=='edm' else args.clip_denoised,
                sampler=args.sampler,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                s_churn=args.s_churn,
                s_tmin=args.s_tmin,
                s_tmax=args.s_tmax,
                s_noise=args.s_noise,
                generator=None,
                ts=ts,
                teacher = True if args.training_mode == 'edm' else False,
                clip_output=args.clip_output,
                ctm=True if args.training_mode.lower() == 'ctm' else False,
                x_T=x_T if args.stochastic_seed == False else None,
                ind_1=args.ind_1,
                ind_2=args.ind_2,
                gamma=args.gamma,
            )
            #print(x[0])

        sample = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        if dist.get_rank() == 0:
            sample = sample.cpu().detach()
            if args.large_log:
                print(f"{(itr-1) * args.batch_size} sampling complete...")
            r = np.random.randint(1000000)
            if args.save_format == 'npz':
                if args.class_cond:
                    np.savez(os.path.join(out_dir, f"sample_{r}.npz"), sample.numpy(), classes.cpu().detach().numpy())
                else:
                    np.savez(os.path.join(out_dir, f"sample_{r}.npz"), sample.numpy())
            if args.save_format == 'png' or itr == 1:
                print("x range: ", x.min(), x.max())
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid((x + 1.) / 2., nrow, padding=2)
                if args.class_cond:
                    with bf.BlobFile(os.path.join(out_dir, f"class_{args.train_classes}_sample_{r}.png"), "wb") as fout:
                        save_image(image_grid, fout)
                else:
                    with bf.BlobFile(os.path.join(out_dir, f"sample_{r}.png"), "wb") as fout:
                        save_image(image_grid, fout)

        eval_num_samples += sample.shape[0]
        if args.large_log:
            print(f"sample {eval_num_samples} time {time.time() - current} sec")
        itr += 1

    dist.barrier()
    logger.log("sampling complete")

def create_argparser():

    defaults = dict(
        generator="determ",
        eval_batch=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        sampling_steps=40,
        model_path="",
        eval_seed=42,
        save_format='png',
        stochastic_seed=False,
        data_name='cifar10',
        #data_name='imagenet64',
        #schedule_sampler="lognormal",
        ind_1=0,
        ind_2=0,
        gamma=0.5,
    )
    defaults.update(train_defaults(defaults['data_name']))
    defaults.update(model_and_diffusion_defaults(defaults['data_name']))
    defaults.update(cm_train_defaults(defaults['data_name']))
    defaults.update(ctm_train_defaults(defaults['data_name']))
    defaults.update(ctm_eval_defaults(defaults['data_name']))
    defaults.update(ctm_loss_defaults(defaults['data_name']))
    defaults.update(ctm_data_defaults(defaults['data_name']))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
