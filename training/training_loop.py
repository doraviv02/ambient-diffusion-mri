# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from collections import OrderedDict
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import wandb

#----------------------------------------------------------------------------

def _run_validation(ema, val_data, device, loss_fn, run_dir, cur_kimg,
                    initial_val, max_items=48):
    """Subject-level held-out (cross-view) validation. No EMA/optimizer/state update.

    Returns (subject_mean, frac_subjects_improved, per_subject_dict).
    """
    from training.dataset import MultiViewKspaceDataset
    try:
        ds = MultiViewKspaceDataset(path=val_data)
    except Exception as e:
        dist.print0(f'  [validation] could not load {val_data}: {e}')
        return None
    n = min(max_items, len(ds))
    # spread sampled slices across the whole dataset so many subjects are covered
    indices = sorted(set(np.linspace(0, len(ds) - 1, n).round().astype(int).tolist()))
    ema_mod = ema._orig_mod if hasattr(ema, '_orig_mod') else ema
    was_training = ema_mod.training
    ema_mod.eval()
    per_subject = {}
    with torch.no_grad():
        for idx in indices:
            torch.manual_seed(100000 + idx)          # deterministic masks per item
            item = ds[idx]
            batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else [v])
                     for k, v in item.items()}
            out = loss_fn(net=ema_mod, batch=batch, labels=None)
            subj = item["subject_id"]
            per_subject.setdefault(subj, []).append(float(out["cross_view_loss"].mean().item()))
    if was_training:
        ema_mod.train()
    subj_means = {s: float(np.mean(v)) for s, v in per_subject.items()}
    subject_mean = float(np.mean(list(subj_means.values()))) if subj_means else float('nan')
    frac_improved = None
    if initial_val:
        common = [s for s in subj_means if s in initial_val]
        if common:
            frac_improved = float(np.mean([subj_means[s] < initial_val[s] for s in common]))
    # append a small csv row (rank 0 only)
    if dist.get_rank() == 0 and run_dir is not None:
        import csv
        vpath = os.path.join(run_dir, 'validation.csv')
        newf = not os.path.exists(vpath)
        with open(vpath, 'a', newline='') as f:
            w = csv.writer(f)
            if newf:
                w.writerow(['kimg', 'n_subjects', 'subject_mean_cross', 'frac_subjects_improved'])
            w.writerow([cur_kimg, len(subj_means), subject_mean,
                        '' if frac_improved is None else frac_improved])
    dist.print0(f'  [validation] kimg={cur_kimg} subjects={len(subj_means)} '
                f'cross_view_mean={subject_mean:.4f} frac_improved='
                f'{"NA" if frac_improved is None else f"{frac_improved:.2f}"}')
    return subject_mean, frac_improved, subj_means


def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    max_grad_norm       = None,     # gradient clipping.
    compile_network     = True,     # Use torch.compile (this project disables it).
    dataset_mode        = None,     # 'image'|'numpy_ambient'|'multiview_kspace'.
    tracking            = 'wandb',  # 'wandb' or 'disabled' (offline cluster jobs).
    transfer_strict     = False,    # Require all trainable tensors on --transfer.
    val_data            = None,     # Validation data dir (subject-level held-out check).
    validation_every_kimg = None,   # Cadence of the validation pass.
    img_channels_override = None,   # Override net input channels (ambient_mv = 4).
    img_resolution_override = None, # Override net img_resolution (from checkpoint).
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    dist.print0(f"batch_gpu: {batch_gpu}, Acc rounds: {num_accumulation_rounds}, World size: {dist.get_world_size()}")
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    precond_name = getattr(dataset_obj, 'precond', None)
    img_res = img_resolution_override if img_resolution_override is not None else dataset_obj.resolution
    if img_channels_override is not None:
        img_ch = img_channels_override
    elif precond_name in ("ambient", "ambient_mv"):
        img_ch = 2 * dataset_obj.num_channels
    else:
        img_ch = dataset_obj.num_channels
    interface_kwargs = dict(img_resolution=img_res, label_dim=dataset_obj.label_dim, img_channels=img_ch)
    dist.print0(f'  network interface: {interface_kwargs}')
    dist.print0(f'  network_kwargs: {dict(network_kwargs)}')
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module


    net.train().requires_grad_(True).to(device)
    with torch.no_grad():
        images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        sigma = torch.ones([batch_gpu], device=device)
        labels = torch.zeros([batch_gpu, net.label_dim], device=device)
        misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    if compile_network:
        net = torch.compile(net)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)


    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        src = data['ema']
        if isinstance(src, (dict, OrderedDict)):
            # Published checkpoints store EMA as a state_dict.
            sd = OrderedDict({k.replace('_orig_mod.', ''): v for k, v in src.items()})
            base = net._orig_mod if hasattr(net, '_orig_mod') else net
            ema_base = ema._orig_mod if hasattr(ema, '_orig_mod') else ema
            missing, unexpected = base.load_state_dict(sd, strict=False)
            ema_base.load_state_dict(sd, strict=False)
            state = base.state_dict()
            matched = sum(1 for k, v in state.items() if k in sd and sd[k].shape == v.shape)
            trainable = {n for n, _ in base.named_parameters()}
            miss_trainable = [m for m in missing if m in trainable]
            shape_bad = [k for k in sd if k in state and sd[k].shape != state[k].shape]
            dist.print0(f'Checkpoint transfer: matched {matched}/{len(state)} tensors '
                        f'({100.0*matched/max(len(state),1):.1f}%); missing_trainable={len(miss_trainable)}; '
                        f'shape_incompatible={len(shape_bad)}')
            if transfer_strict and (miss_trainable or shape_bad):
                raise RuntimeError(
                    f'Strict transfer failed: {len(miss_trainable)} missing trainable tensors, '
                    f'{len(shape_bad)} shape-incompatible tensors.')
        else:
            misc.copy_params_and_buffers(src_module=src, dst_module=net, require_all=transfer_strict)
            misc.copy_params_and_buffers(src_module=src, dst_module=ema, require_all=transfer_strict)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        with dnnlib.util.open_url(resume_state_dump, verbose=(dist.get_rank() == 0)) as f:
            data = torch.load(f, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    cur_nimg = resume_kimg * 1000
    # cur_tick = 0
    cur_tick = resume_kimg // kimg_per_tick
    dist.print0("Starting from tick: ")
    dist.print0(f'Starting wandb step: {cur_tick * snapshot_ticks}')
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    initial_loss = None
    last_val_nimg = -1
    initial_val_subject = None
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                dataset_iter = next(dataset_iterator)
                if isinstance(dataset_iter, dict):
                    # Multi-view cross-view fine-tuning (dict batch).
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                             for k, v in dataset_iter.items()}
                    loss_dict = loss_fn(net=ddp, batch=batch, labels=None, augment_pipe=augment_pipe)
                    loss = loss_dict['loss']
                    training_stats.report('Loss/loss', loss)
                    training_stats.report('Loss/cross_view', loss_dict['cross_view_loss'])
                    training_stats.report('Loss/input_heldout', loss_dict['input_heldout_loss'])
                    training_stats.report('Loss/per_view_residual', loss_dict['per_view_residual'])
                    training_stats.report('Loss/prediction_norm', loss_dict['prediction_norm'])

                elif len(dataset_iter) == 2:
                    images, labels = dataset_iter
                    images = images.to(device).to(torch.float32)
                    labels = labels.to(device)

                    loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                    training_stats.report('Loss/loss', loss)

                elif len(dataset_iter) == 5:
                    images, labels, corruption_matrix, hat_corruption_matrix, maps = dataset_iter
                    corruption_matrix = corruption_matrix.to(device)
                    hat_corruption_matrix = hat_corruption_matrix.to(device)
                    maps = maps.to(device)
                    images = images.to(device).to(torch.float32)
                    labels = labels.to(device)

                    train_loss, val_loss, test_loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe, 
                    corruption_matrix=corruption_matrix, hat_corruption_matrix=hat_corruption_matrix, maps=maps)
                    loss = val_loss
                    training_stats.report('Loss/loss', loss)
                    training_stats.report('Loss/loss_test', test_loss)
                    training_stats.report('Loss/loss_train', train_loss)
                else:
                    raise ValueError(f"Invalid dataset iterator length: {len(dataset_iter)}")
                
                scalar_loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                if initial_loss is None:
                    initial_loss = scalar_loss.item()
                
                scalar_loss.backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        
        # apply gradient clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema.state_dict(), loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with dnnlib.util.open_url(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), verbose=(dist.get_rank() == 0), read_mode='wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Subject-level validation pass (no state update).
        if (val_data is not None) and (validation_every_kimg is not None) and (
                done or (cur_nimg - last_val_nimg) >= validation_every_kimg * 1000):
            last_val_nimg = cur_nimg
            vres = _run_validation(ema, val_data, device, loss_fn, run_dir, cur_nimg // 1000,
                                   initial_val_subject)
            if vres is not None:
                subject_mean, frac_improved, subj_means = vres
                if initial_val_subject is None:
                    initial_val_subject = subj_means
                training_stats.report0('Val/cross_view_mean', subject_mean)
                if frac_improved is not None:
                    training_stats.report0('Val/frac_subjects_improved', frac_improved)

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net.state_dict(), optimizer_state=optimizer.state_dict()), 
                dnnlib.util.open_url(os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'), verbose=(dist.get_rank() == 0), read_mode='wb'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = dnnlib.util.open_url(os.path.join(run_dir, 'stats.jsonl'), read_mode='at', verbose=(dist.get_rank() == 0))
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            # report to wandb (skip in disabled/offline tracking mode)
            if tracking != 'disabled':
                for key, value in training_stats.default_collector.as_dict().items():
                    wandb.log({key: value}, step=cur_tick * snapshot_ticks)
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------