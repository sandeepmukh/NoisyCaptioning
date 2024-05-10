import json
import logging
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.mixture import GaussianMixture

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast
from .sequence_mixing import seqmix


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

@torch.no_grad()
def fit_gmm(model, data, device, n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4):
    losses = []
    logging.info("Fitting GMM")
    for i, (img, text) in enumerate(data):
        img = img.to(device=device, non_blocking=True)
        text = text.to(device=device, non_blocking=True)
        preds = model(img, text)
        loss = F.cross_entropy(preds['logits'].permute(0, 2, 1), preds['labels'], reduction='none').flatten().detach().cpu().numpy()
        losses.extend(loss)
    losses = np.array(losses).reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, tol=tol, reg_covar=reg_covar)
    gmm.fit(losses)
    logging.info("GMM fitted")
    return gmm

@torch.no_grad()
def update_swa_model(model, dist_model, batch, args, epoch, swa_scheduler=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    
    logging.info("Updating SWA model")
    dist_model.update_parameters(model)
    if swa_scheduler is not None:
        swa_scheduler.step()
        
    # logging.info("Updating SWA bn statistics")
    # update BN statistics
    # dist_model.train()
    img, text = batch
    img = img.to(device=device, non_blocking=True, dtype=input_dtype)
    text = text.to(device=device, non_blocking=True)
    with autocast():
        dist_model(img, text)     
    # dist_model.eval()
    logging.info("SWA model updated")
    
def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()
        
    if args.elr_distill:
        dist_model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill or (args.elr_distill and epoch > args.elr_teacher_warmup):
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})

                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
    
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
        
        if args.elr_distill and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            update_swa_model(model, dist_model, batch, args, epoch)
    # end for

@torch.no_grad()
def coguessing(imgs, text, model, swa_model, unlabel_token_mask, gmm_w, args):
    # label co-guessing of unlabeled samples
    outputs_x = model(imgs, text)['logits']
    outputs_x2 = swa_model(imgs, text)['logits']   
    
    targets = torch.zeros_like(outputs_x)        
    
    px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
    gmm_w = gmm_w.reshape(-1,1,1)
    # mix with labels
    targets[unlabel_token_mask] = gmm_w*F.one_hot(text[unlabel_token_mask]) + (1-gmm_w)*px[unlabel_token_mask]             
    
    targets[~unlabel_token_mask] = px[~unlabel_token_mask] 
    targets = targets**(1/args.T)
    
    targets = targets / targets.sum(dim=-1, keepdim=True) # normalize
    targets = targets.detach()
    
    return targets

@torch.no_grad()
def get_unlabel_token_mask(gmm, model, img, text, tau):
    clean_idx = np.argmin(gmm.means_.flatten())
    preds = model(img, text)
    loss = F.cross_entropy(preds['logits'].permute(0, 2, 1), preds['labels'], reduction='none').flatten().detach().cpu().numpy()
    clean_probs = gmm.predict_proba(loss.reshape(-1, 1))[:, clean_idx]
    noisy_mask = clean_probs < tau
    return noisy_mask, clean_probs

def train_one_dividemix_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, data_eval_train, tb_writer=None,):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()
        
    if args.elr_distill or args.dividemix:
        dist_model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # Fit GMM and split data
    gmm = fit_gmm(dist_model, data_eval_train['train'].dataloader, device)

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        
        noisy_mask, clean_probs = get_unlabel_token_mask(gmm, dist_model, images, texts, args.clean_threshold)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            lam = np.random.beta(args.dividemix_alpha, args.dividemix_alpha)
            lam = max(lam, 1-lam)
            perm = torch.randperm(images.size(0))
            model_out = model(images, texts, dmix_permutation=perm, dmix_lam=lam)
            logit_scale = model_out["logit_scale"]
            with torch.no_grad():
                if args.dividemix:
                    pseudotargets = coguessing(images, texts, model, dist_model, noisy_mask, clean_probs, args)
                    # mixup
                    pseudotargets = pseudotargets * lam + pseudotargets[perm] * (1 - lam)
                    model_out.update('labels', pseudotargets)
                else:
                    dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
            
            losses = loss(**model_out, output_dict=True)

            total_loss = sum(losses.values())
            losses["loss"] = total_loss
        
        backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
    
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
        
        if (args.elr_distill or args.dividemix) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            update_swa_model(model, dist_model, batch, args, epoch)
    # end for


def train_and_split_clean_noisy(model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    clean_set = []
    noisy_set = []
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                losses = loss(**model_out, output_dict=True)
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            probs = fit_gmm(losses)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                probs = fit_gmm(losses)

        # Use clean-label probabilities to split up data
        mask = probs >= args.clean_threshold
        clean_set.extend(list(zip(images[mask], texts[mask])))
        noisy_set.extend(list(zip(images[~mask], texts[~mask])))
    
    return clean_set, noisy_set


def train_one_epoch_dividemix(model, model2, clean_data, noisy_data, loss, epoch, optimizer, scaler, scheduler, augmentations, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    clean_data.set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    clean_dataloader = clean_data.dataloader
    noisy_data.set_epoch(epoch)
    noisy_dataloader = noisy_data.dataloader

    num_batches_per_epoch = clean_dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(clean_dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    noisy_iter = iter(noisy_dataloader)
    for i, batch in enumerate(clean_dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        try:
            noisy_images, noisy_texts = noisy_iter.next()
        except:
            noisy_iter = iter(noisy_dataloader)
            noisy_images, noisy_texts = noisy_iter.next()

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        augmented_images = torch.cat([aug(images) for aug in augmentations], dim=0)
        augmented_noisy_images = torch.cat([aug(noisy_images) for aug in augmentations], dim=0)
        texts = texts.to(device=device, non_blocking=True)
        repeated_texts = texts.repeat(len(augmentations))
        repeated_noisy_texts = noisy_texts.repeat(len(augmentations))

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # "Co-divide"
                model_out = model(augmented_images, repeated_texts)
                outputs_by_image = model_out.view(-1, len(augmentations), *model_out.shape[1:])
                average_labels = torch.stack([seqmix(outputs_by_image[j]) for j in range(outputs_by_image.shape[0])])
                new_texts = torch.stack([seqmix(texts[j], average_labels[j]) for j in range(average_labels.shape[0])])
                
                logit_scale = model_out["logit_scale"]
                losses = loss(**model_out, output_dict=True)
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

                # Co-guessing
                noisy_out1 = model(augmented_noisy_images, repeated_noisy_texts)
                noisy_out2 = model2(augmented_noisy_images, repeated_noisy_texts)
                combined_noisy_outputs = torch.cat([noisy_out1, noisy_out2], dim=0)
                outputs_by_image = combined_noisy_outputs.view(-1, 2 * len(augmentations), *combined_noisy_outputs.shape[1:])
                new_noisy_texts = torch.stack([seqmix(outputs_by_image[j]) for j in range(outputs_by_image.shape[0])])

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        all_images = torch.cat([augmented_images, augmented_noisy_images], dim=0)
        all_captions = torch.cat([new_texts, new_texts, new_noisy_texts, new_noisy_texts], dim=0)

        idx = torch.randperm(all_images.size(0))

        image_a, image_b = all_images, all_images[idx]
        caption_a, caption_b = all_captions, all_captions[idx]
        
        mixed_image = l * image_a + (1 - l) * image_b        
        mixed_caption = seqmix(caption_a, caption_b, weights = [l, 1 - l])
                
        mixmatch_output = model(mixed_image, mixed_caption)
           
        # TODO: use mixmatch losses + losses from before to update network

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = clean_dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
    
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def evaluate_subset(model, checkpoints, data, og_data, args, tokenizer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    
    for checkpoint in checkpoints:
        checkpoint_dict = torch.load(checkpoint)
        saved_epoch = checkpoint_dict["epoch"]
        try:
            model.load_state_dict(checkpoint_dict["state_dict"])
        except RuntimeError:  # smth about parallel modules prepending "module" to every key
            old_state_dict = checkpoint_dict["state_dict"]
            state_dict = {key.replace("module.", ""): old_state_dict[key] for key in old_state_dict}
            model.load_state_dict(state_dict)
        model.eval()
        metrics = {}
        sub_metrics = {}
        if not is_master(args):
            return metrics
        metrics["saved_epoch"] = saved_epoch
        sub_metrics["saved_epoch"] = saved_epoch

        # NOTE: no zero shot analysis yet for presentation
        # zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
        # metrics.update(zero_shot_metrics)

        all_image_features, all_text_features = [], []
        all_loss, all_gen_loss = [], []
        all_labels, all_logits, all_logit_scales = [], [], []
        all_logits_per_image, all_logits_per_text = [], []
        all_predictions, all_per_position_losses = [], []
        all_per_token_losses = {}
        token_label_counts = {}
        pred_attn_regions = []

        with torch.no_grad():
            for i, (img, text) in enumerate(data):
                img = img.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                text = text.to(device=device, non_blocking=True).unsqueeze(0)
                print("input img", img.shape)
                print("input caption", text.shape)
                with autocast():
                    model_out = model(img, text)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    attn_scores = model_out.get("attention_scores", None)
                    
                    logit_scale = model_out["logit_scale"]
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    all_image_features.append(image_features.cpu().tolist()[0])
                    all_text_features.append(text_features.cpu().tolist()[0])
                    all_logits.append(model_out["logits"].cpu().tolist()[0])
                    all_logit_scales.append(logit_scale.item())
                    decoded_label_tokens = [tokenizer.decode([token.item()]) for token in model_out["labels"][0]]
                    all_labels.append(" ".join(decoded_label_tokens))
                    all_logits_per_image.append(logits_per_image.cpu().tolist()[0])
                    all_logits_per_text.append(logits_per_text.cpu().tolist()[0])

                    batch_size = img.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
                    gen_loss = maybe_compute_generative_loss(model_out)

                    logits = model_out["logits"]
                    label_pred = F.softmax(logits, dim = -1).argmax(dim = -1)
                    decoded_pred_tokens = [tokenizer.decode([token.item()]) for token in label_pred[0]]
                    label_pred = " ".join(decoded_pred_tokens)
                    all_predictions.append(label_pred)

                    if attn_scores is not None:
                        process_attention_scores(torch.stack(attn_scores, dim = 0), decoded_pred_tokens, og_data[i][0], i, args, average_over_layers = False, average_over_tokens = False)

                    position_losses = F.cross_entropy(logits.transpose(1, 2), model_out["labels"], reduction = "none")[0]
                    all_per_position_losses.append(position_losses.cpu().tolist())

                    for j, label_token in enumerate(model_out["labels"][0]):
                        lt = tokenizer.decode([label_token.item()])
                        if lt not in all_per_token_losses:
                            all_per_token_losses[lt] = 0
                            token_label_counts[lt] = 0
                        token_label_counts[lt] += 1
                        all_per_token_losses[lt] = (all_per_token_losses[lt] + position_losses[j].item()) / token_label_counts[lt]
                
                all_loss.append((total_loss * batch_size).item())
                if gen_loss is not None:
                    all_gen_loss.append((gen_loss * batch_size).item())
        
        metrics["all_loss"] = all_loss
        metrics["all_gen_loss"] = all_gen_loss
        metrics["all_labels"] = all_labels
        metrics["all_predictions"] = all_predictions
        metrics["all_per_position_losses"] = all_per_position_losses
        metrics["all_per_token_losses"] = all_per_token_losses
        metrics["all_pred_attentions"] = pred_attn_regions
        
        sub_metrics["all_image_features"] = all_image_features
        sub_metrics["all_text_features"] = all_text_features
        sub_metrics["all_logits"] = all_logits
        sub_metrics["all_logit_scales"] = all_logit_scales
        sub_metrics["all_logits_per_image"] = all_logits_per_image
        sub_metrics["all_logits_per_text"] = all_logits_per_text
        sub_metrics["token_label_counts"] = token_label_counts

        #with open(os.path.join(args.eval_log_dir, f"checkpoint_{saved_epoch}_metrics.json"), "w") as f:
        #    f.write(json.dumps(metrics))
        #    f.write("\n")
        #with open(os.path.join(args.eval_log_dir, f"checkpoint_{saved_epoch}_submetrics.json"), "w") as f:
        #    f.write(json.dumps(sub_metrics))
        #    f.write("\n")
        print("Done with checkpoint", saved_epoch)
    
    print("DONE WITH EVALUATION YIPPEEEEE")


def process_attention_scores(scores, tokens, og_img, og_idx, args, average_over_layers = False, average_over_tokens = False):
    scores = scores.squeeze()  # (12, 1, 75, 255) => (12, 75, 255)
    print("scores", scores.shape)

    num_patches_side = int(np.sqrt(scores.size(-1)))  # 15
    patch_size = 224 / num_patches_side
    scale_factor = og_img.size[0] / 224  # 256 / 224
    print("original image", og_img.size)

    if average_over_layers:
        layer_attn = scores.mean(dim = 0)
        if average_over_tokens:
            attn = layer_attn.mean(dim = 0)
            process_token_attention_scores(attn, None, og_img, og_idx, num_patches_side, patch_size, scale_factor, None, args)
        else:
            process_token_attention_scores(layer_attn, tokens, og_img, og_idx, num_patches_side, patch_size, scale_factor, None, args)
    else:
        for layer in range(scores.size(0)):  # a slice of (75, 255)
            layer_attn = scores[layer]
            if average_over_tokens:
                attn = layer_attn.mean(dim = 0)
                process_token_attention_scores(layer_attn, None, og_img, og_idx, num_patches_side, patch_size, scale_factor, layer, args)
            else:
                process_token_attention_scores(layer_attn, tokens, og_img, og_idx, num_patches_side, patch_size, scale_factor, layer, args)
    with open(os.path.join(args.eval_attention_dir, f"sample_{og_idx}", f"caption.txt"), "w") as f:
        f.write("\n".join([f"{i}. {token}" for i, token in enumerate(tokens)]))


def process_token_attention_scores(attn, tokens, og_img, og_idx, num_patches_side, patch_size, scale_factor, layer, args):
    if tokens is not None:
        for i, token in enumerate(tokens):  # for each of 75 tokens
            if token != "<end_of_text>":
                og_img_array = np.array(og_img)
                token_attn = attn[i] / attn[i].max()
                for j in range(attn.size(-1)):  # for each of 255 patches
                    row = j // num_patches_side
                    col = j % num_patches_side
                    intensity = token_attn[j].item()
                    x = int(col * patch_size * scale_factor)
                    y = int(row * patch_size * scale_factor)
                    x_end = int((col + 1) * patch_size * scale_factor)
                    y_end = int((row + 1) * patch_size * scale_factor)
                    og_img_array[y:y_end, x:x_end] = (og_img_array[y:y_end, x:x_end] * intensity).astype(np.uint8)
                blur_patch_borders(og_img_array, num_patches_side, patch_size, scale_factor)
                if layer:
                    img_name = f"layer_{layer if layer != -1 else '11'}_token_{i}.png"
                else:
                    img_name = f"token_{i}.png"
                Image.fromarray(og_img_array).save(os.path.join(args.eval_attention_dir, f"sample_{og_idx}", img_name))
    else:
        og_img_array = np.array(og_img)
        attn = attn / attn.max()
        for j in range(attn.size(-1)):  # for each of 255 patches
            row = j // num_patches_side
            col = j % num_patches_side
            intensity = token_attn[j].item()
            x = int(col * patch_size * scale_factor)
            y = int(row * patch_size * scale_factor)
            x_end = int((col + 1) * patch_size * scale_factor)
            y_end = int((row + 1) * patch_size * scale_factor)
            og_img_array[y:y_end, x:x_end] = (og_img_array[y:y_end, x:x_end] * intensity).astype(np.uint8)
            blur_patch_borders(og_img_array, num_patches_side, patch_size, scale_factor)
        if layer:
            img_name = f"layer_{layer if layer != -1 else '11'}.png"
        else:
            img_name = f"attention.png"
        Image.fromarray(og_img_array).save(os.path.join(args.eval_attention_dir, f"sample_{og_idx}", img_name))


def blur_patch_borders(img_array, num_patches_side, patch_size, scale_factor):
    patch_height = patch_width = int(patch_size * scale_factor)
    for row in range(num_patches_side):
        for col in range(num_patches_side):
            x_start = col * patch_width
            y_start = row * patch_height
            x_end = (col + 1) * patch_width
            y_end = (row + 1) * patch_height

            if row > 0:
                img_array[y_start:(y_start + 1), x_start:x_end] = (img_array[y_start:(y_start + 1), x_start:x_end] + img_array[(y_start - 1):y_start, x_start:x_end]) / 2
            if row < num_patches_side - 1:
                img_array[(y_end - 1):y_end, x_start:x_end] = (img_array[(y_end - 1):y_end, x_start:x_end] + img_array[y_end:(y_end + 1), x_start:x_end]) / 2

            if col > 0:
                img_array[y_start:y_end, x_start:(x_start + 1)] = (img_array[y_start:y_end, x_start:(x_start + 1)] + img_array[y_start:y_end, (x_start - 1):x_start]) / 2
            if col < num_patches_side - 1:
                img_array[y_start:y_end, (x_end - 1):x_end] = (img_array[y_start:y_end, (x_end - 1):x_end] + img_array[y_start:y_end, x_end:(x_end + 1)]) / 2


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
