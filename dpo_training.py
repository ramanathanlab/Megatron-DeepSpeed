# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import os
from rich import print
import torch
import math
import numpy as np

# The earliest we can measure the start time.
import time
from datetime import datetime
import threading

from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group, update_rotary_pos_emb
from megatron.arguments import core_transformer_config_from_args
from megatron.utils import (
    report_memory,
    throughput_calculator,
    checkpoint_throughput_calculator
)
from pathlib import Path

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
import subprocess
import wandb

import time
from torch import nn
import torch.nn.functional as F

# from ezpz import get_logger
from ezpz.dist import get_world_size, setup_wandb, get_rank

# More imports
from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options
from megatron.training import print_datetime, _create_ds_config_dict
from megatron.training import setup_model_and_optimizer
from megatron.training import load_model_weights_only, get_model
from megatron.training import load_model_weights_only_modified
from megatron.training import get_optimizer_param_scheduler, cyclic_iter
from megatron.training import train, train_step
from megatron.training import train_step_dpo, training_log_dpo
from megatron.optimizer import get_megatron_optimizer
from megatron.checkpointing import load_checkpoint
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.arguments import core_transformer_config_from_args
from megatron import update_num_microbatches
from megatron import get_num_microbatches
from megatron.utils import throughput_calculator, get_parameters_in_billions
from megatron.text_generation import generate_and_post_process, beam_search_and_post_process
from megatron.text_generation.forward_step import ForwardStep, InferenceParams
from megatron.text_generation.sampling import sample
from megatron.text_generation.tokenization import detokenize_generations
from megatron.text_generation.communication import (
    copy_from_last_to_first_pipeline_stage,
    broadcast_from_last_pipeline_stage,
    broadcast_from_last_to_first_pipeline_stage)
from megatron.checkpointing import save_checkpoint
from megatron.utils import get_ltor_masks_and_position_ids
from generate_utils import generate_post_training

# RANK = setup_torch(
#     backend='deepspeed',
#     port='5432',
# )
RANK = get_rank()
WORLD_SIZE = get_world_size()
LEVEL = "DEBUG" if RANK == 0 else "CRITICAL"

WANDB_MODE = os.environ.get('WANDB_MODE', None)
DISABLE_WANDB = (
    WANDB_MODE is not None and str(WANDB_MODE).lower() == 'disabled'
)

if RANK == 0 and not DISABLE_WANDB:
    project_name = (
        os.environ.get(
            'WB_PROJECT',
            os.environ.get(
                'WANDB_PROJECT',
                'AuroraGPT'
            ),
        )
    )
    print('--------------------------------------------------')
    print(f"Setting up W&B from: {RANK} with {project_name}")
    print('--------------------------------------------------')
    #setup_wandb(project_name=project_name)

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    print_rank_0('building GPT model ...')
    see_memory_usage("Before Building Model", force=True)
    args = get_args()
    config = core_transformer_config_from_args(args)
    if wandb.run is not None:
        print(f"Updating WandB run: [{wandb.run.name}]({wandb.run.url})")
        wandb.run.config.update({"args": vars(args)}, allow_val_change=True)
    if RANK == 0:
        git_ds_info()
    if hasattr(mpu, 'get_sequence_parallel_group'):
        dpg = mpu.get_sequence_parallel_group()
    elif hasattr(mpu, 'get_data_parallel_group'):
        dpg = mpu.get_data_parallel_group()
    else:
        dpg = None
    if wandb is not None and wandb.run is not None:
        assert wandb is not None and wandb.run is not None
        print(f'Updating {wandb.run.name=} at {wandb.run.url=}')
        wandb.run.config.update({'args': vars(args)}, allow_val_change=True)
    with deepspeed.zero.Init(
            data_parallel_group=dpg,
            remote_device=(
                None if args.remote_device == 'none' else args.remote_device
            ),
            config_dict_or_path=args.deepspeed_config_dict,
            enabled=args.zero_stage == 3,
            mpu=mpu
    ):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(
                config=config,
                num_tokentypes=0,
                parallel_output=True
            )
            # This is a hack to give us a reference to
            # get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args.
            # This avoids having to pipeline it
            # as an activation during training.
            # The mask is constant, and thus we can reuse it.
            attention_mask = torch.tril(
                torch.ones(
                    (1, args.seq_length, args.seq_length),
                    device=get_accelerator().current_device_name()
                )
            ).view(1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)

            # For prertaining, since sequence length is fixed,
            # cache rotary embedding in args, to avoid communicating around
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.seq_length)

        else:
            print(f'Building model check..')
            model = GPTModel(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print_rank_0('\n ------------------------ ')
    # print_rank_0(f'num of parameters {num_params}')
    # print_rank_0('------------------------\n ')
    print_rank_0(80 * '-')
    print_rank_0(f"Number of parameters in model: {num_params}")
    print_rank_0(80 * '-')
    see_memory_usage("After Building Model", force=True)
    if wandb.run is not None:
        wandb.run.config.update({'num_params': num_params}, allow_val_change=True)
    #     wandb.run.watch(
    #         model,
    #         log='all',
    #         log_graph=True,
    #     )
    #     wandb.run.config.update({'num_params': num_params})
    return model

def throughput_flops(model, args, iteration_time, total_iterations):
    batch_size = args.micro_batch_size * get_num_microbatches() * args.data_parallel_size
    approx_parameters_in_billions = None if (model is None) else get_parameters_in_billions(model)
    elapsed_time_per_iter = iteration_time/total_iterations
    samples_per_second = batch_size / elapsed_time_per_iter

    #flops calculator
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    vocab_size = args.padded_vocab_size

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3.
    checkpoint_activations_factor = 3
    if hasattr(args, 'checkpoint_activations') and args.checkpoint_activations:
        checkpoint_activations_factor = 4
    if hasattr(args, 'recompute_granularity') and (args.recompute_granularity == 'selective' or args.recompute_granularity == 'full'):
        checkpoint_activations_factor = 4
    seq_len = args.seq_length
    if hasattr(args, 'actual_seq_length'):
        seq_len = args.actual_seq_length
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
    tflops = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))

    return tflops

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()
    # print(f'len(tokenizer.vocab): {len(tokenizer.vocab)}')

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    data = next(data_iterator) if data_iterator is not None else None
    # # Broadcast data.
    # if data_iterator is not None:
    #     data = next(data_iterator)
    # else:
    #     data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    skip_mask = args.use_flash_attn or args.use_flash_attn_triton
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        skip_mask)

    # For DS's sequence parallel
    seq_parallel_world_size = mpu.get_sequence_parallel_world_size()
    seq_parallel_world_rank = mpu.get_sequence_parallel_rank()

    # For Megatron's sequence parallel
    if args.sequence_parallel:
        seq_parallel_world_size = mpu.get_tensor_model_parallel_world_size()
        seq_parallel_world_rank = mpu.get_tensor_model_parallel_rank()
    seq_length = tokens.size(1)

    assert seq_length % seq_parallel_world_size == 0
    sub_seq_length = seq_length // seq_parallel_world_size
    sub_seq_start = seq_parallel_world_rank * sub_seq_length
    sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

    tokens = tokens[:, sub_seq_start:sub_seq_end]
    position_ids = position_ids[:, sub_seq_start:sub_seq_end]
    # For DS's sequence parallel
    if mpu.get_sequence_parallel_world_size() > 1:
        labels = labels[:, sub_seq_start:sub_seq_end]

    return tokens, labels, loss_mask, attention_mask, position_ids


def data_post_process(data, data_sampler_state_dict):
    args = get_args()
    if args.data_efficiency_curriculum_learning:
        if 'seqlen_truncate' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_truncate'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_truncate']
            if current_seqlen < args.seq_length:
                data['text'] = data['text'][:, :(current_seqlen+1)].contiguous()
        elif 'seqlen_reshape' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_reshape'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_reshape']
            if current_seqlen < args.seq_length:
                orig_num_token = torch.numel(data['text'])
                reshape_len = (data['text'].size()[1] // (current_seqlen+1)) * (current_seqlen+1)
                data['text'] = torch.cat((data['text'][:, :reshape_len].contiguous().view(-1, current_seqlen+1),
                    data['text'][:, -(current_seqlen+1):]), 0).contiguous()
                num_row = math.ceil(orig_num_token / (current_seqlen+1))
                num_row = min(num_row, data['text'].size()[0])
                if num_row > 1 and num_row % 2 != 0:
                    num_row -= 1
                data['text'] = data['text'][:num_row, :].contiguous()
        else:
            args.data_efficiency_curriculum_learning_seqlen_type = None
    return data


def get_batch_pipe(data):
    """
    Modification of `get_batch` to work on `next(data_iterator)`
    instead of `data_iterator`
    """
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    if (
                args.curriculum_learning_legacy
                and args.curriculum_seqlen < tokens.size()[1]
    ):
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask
        # have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        if labels is not None:
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, moe_loss, mos_loss, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    if args.mos or args.kd:
        # assert max(args.num_experts) >= 1
        loss = loss + moe_loss + mos_loss
        if args.mos:
            return loss, {
                'total loss': loss,
                'lm loss': averaged_loss[0],
                'moe loss': moe_loss,
                'mos loss': mos_loss
            }
        elif args.kd:
            return loss, {
                'total loss': loss,
                'lm loss': averaged_loss[0],
                'moe loss': moe_loss,
                'kd loss': mos_loss
            }
        print_rank_0(
            f'>>> total loss: {loss}, '
            f'lm loss {averaged_loss[0]}, '
            f'kd loss {mos_loss}'
        )
    else:
        if max(args.num_experts) <= 1:
            return loss, {'lm loss': averaged_loss[0]}
        loss = loss + moe_loss
        return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}

def dpo_loss_func(loss_mask, dpo_loss, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    if args.mos or args.kd:
        # assert max(args.num_experts) >= 1
        loss = loss + moe_loss + mos_loss
        if args.mos:
            return loss, {
                'total loss': loss,
                'lm loss': averaged_loss[0],
                'moe loss': moe_loss,
                'mos loss': mos_loss
            }
        elif args.kd:
            return loss, {
                'total loss': loss,
                'lm loss': averaged_loss[0],
                'moe loss': moe_loss,
                'kd loss': mos_loss
            }
        print_rank_0(
            f'>>> total loss: {loss}, '
            f'lm loss {averaged_loss[0]}, '
            f'kd loss {mos_loss}'
        )
    # else:
    #     if max(args.num_experts) <= 1:
    #         return loss, {'lm loss': averaged_loss[0]}
    #     loss = loss + moe_loss
    #     return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}
    else:
        # if max(args.num_experts) <= 1:
            # return loss, {'lm loss': averaged_loss[0]}
        loss = dpo_loss
        return loss, {'lm loss': averaged_loss[0], 'dpo loss': dpo_loss}

def batch_seq_logprobs(logits, labels):
    """ Function to compute a batch of sequence log probabilities """

    logits = logits[:-1, :, :] # skip last logit
    logits_logsoftmax = logits.log_softmax(-1) # compute log softmax of logits

    labels = labels[1:, :].clone() # clone labels

    # # Loss mask to avoid padded tokens while computing loss
    # loss_mask = labels != tokenizer.pad_token_id

    # print(f'Labels shape: {labels.shape}')
    # print(f'loss_mask shape: {loss_mask.shape}')
    # print(f'loss_mask dtype: {loss_mask.dtype}')

    # Gather logps and squeeze last dimension
    logprobs = torch.gather(logits_logsoftmax, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    # print(f'seq_logprobs shape: {logprobs.shape}')

    # Weighted sum over logprobs using loss mask
    # seq_logprobs = (logprobs * loss_mask).sum(-1)
    seq_logprobs = logprobs.sum(-1)

    return seq_logprobs


def calculate_mos_loss(
        args,
        stu_output,
        teacher_model,
        tokens,
        position_ids,
        attention_mask
):
    mos_loss = 0
    alpha = args.kd_alpha_ce
    beta = args.kd_beta_ce
    kd_temp = args.kd_temp

    if teacher_model:
        with torch.no_grad():
            if (
                        args.curriculum_learning_legacy and
                        args.curriculum_seqlen < args.seq_length
            ):
                assert args.curriculum_seqlen is not None
                curriculum_seqlen = args.curriculum_seqlen
                tokens = tokens[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                csl = curriculum_seqlen
                attention_mask = (
                        attention_mask[:, :, :csl, :csl].contiguous()
                )
                # No need to truncate labels
                # as we do not need it for the teacher logits
            tea_output, tea_other_losses = teacher_model(
                tokens,
                position_ids,
                attention_mask
            )
            assert stu_output.size() == tea_output.size(), (
                    'teacher and student output should match in size. '
                    f'Student: {stu_output.size()}, '
                    f'Teacher: {tea_output.size()}, '
                    f'CL seq length {args.curriculum_seqlen}'
            )

        student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
        # The target logits is expected to be probabilities.
        # If we use log_softmax,
        # then we need to set target_log to true
        # when initializing the KLDivLoss.
        tea_logits = F.softmax(tea_output / kd_temp, dim=2)

        mos_loss = kd_temp * kd_temp * nn.KLDivLoss(reduction='batchmean')(
            student_logits,
            tea_logits
        )

        mos_loss = mos_loss.div(args.seq_length) * beta
    return mos_loss

def calculate_dpo_loss(
        args,
        stu_output,
        teacher_model,
        logprobs_p,
        logprobs_u,
        ref_logprobs_p,
        ref_logprobs_u,
        tokens,
        position_ids,
        attention_mask
):
    mos_loss = 0
    alpha = args.kd_alpha_ce
    beta = args.kd_beta_ce
    kd_temp = args.kd_temp
    kd_temp = 1.0
    beta = 0.1 # add to cmdline args

    if teacher_model:
        with torch.no_grad():
            if (
                        args.curriculum_learning_legacy and
                        args.curriculum_seqlen < args.seq_length
            ):
                assert args.curriculum_seqlen is not None
                curriculum_seqlen = args.curriculum_seqlen
                tokens = tokens[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                csl = curriculum_seqlen
                attention_mask = (
                        attention_mask[:, :, :csl, :csl].contiguous()
                )
                # No need to truncate labels
                # as we do not need it for the teacher logits
            ref_output, ref_other_losses = teacher_model(
                tokens,
                position_ids,
                attention_mask
            )
            assert stu_output.size() == ref_output.size(), (
                    'ref and student output should match in size. '
                    f'Student: {stu_output.size()}, '
                    f'Reference: {ref_output.size()}, '
                    f'CL seq length {args.curriculum_seqlen}'
            )

        student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
        # Labels ?
        logprobs = torch.gather(student_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # The target logits is expected to be probabilities.
        # If we use log_softmax,
        # then we need to set target_log to true
        # when initializing the KLDivLoss.

    # Get ratios of preferred log probabilities from model and ref model
    logprob_ratio_p = logprobs_p - ref_logprobs_p

    # Get ratios of unpreferred log probabilities from model and ref model
    logprob_ratio_u = logprobs_u - ref_logprobs_u

    # Difference of logprobs ratios scaled by beta
    scaled_diff_logprob_ratios = beta * (logprob_ratio_p - logprob_ratio_u)

    # Losses computed as negative logsigmoid of scaled difference
    losses = -F.logsigmoid(scaled_diff_logprob_ratios)

    # preferred dpo rewards
    pref_dpo_rewards = (beta * logprob_ratio_p).detach()

    # unpreferred dpo rewards
    unpref_dpo_rewards = (beta * logprob_ratio_u).detach()

    # Implicit DPO rewards
    implicit_dpo_rewards = (pref_dpo_rewards > unpref_dpo_rewards).float()
    rewards = implicit_dpo_rewards.cpu().mean()

    # Compute mean loss
    dpo_loss = losses.mean()
    # print(f'Loss dtype: {loss.dtype}')

    return dpo_loss, rewards

def compute_dp_loss(logprobs_p, ref_logprobs_p,
                    logprobs_u, ref_logprobs_u,
                    beta=0.1):

    # Get ratios of preferred log probabilities from model and ref model
    logprob_ratio_p = logprobs_p - ref_logprobs_p

    # Get ratios of unpreferred log probabilities from model and ref model
    logprob_ratio_u = logprobs_u - ref_logprobs_u

    # Difference of logprobs ratios scaled by beta
    scaled_diff_logprob_ratios = beta * (logprob_ratio_p - logprob_ratio_u)

    # Losses computed as negative logsigmoid of scaled difference
    losses = -F.logsigmoid(scaled_diff_logprob_ratios)

    # Compute mean loss
    dp_loss = losses.mean()

    return dp_loss



def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if args.data_efficiency_curriculum_learning:
        args.curriculum_seqlen = tokens.size()[1]
        if (
                hasattr(
                    args,
                    'data_efficiency_curriculum_learning_seqlen_type')
                and (
                    args.data_efficiency_curriculum_learning_seqlen_type
                    == 'seqlen_reshape'
                )
        ):
            args.data_efficiency_curriculum_learning_numel = (
                    torch.numel(tokens)
            )

    if args.mos or args.kd:
        # The forward func can return either the loss or the logits,
        # depending on whether passing in the labels or not.
        stu_output, other_losses = model(tokens, position_ids, attention_mask)
        if (
                    args.curriculum_learning_legacy
                    and args.curriculum_seqlen < args.seq_length
        ):
            assert args.curriculum_seqlen is not None
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        output_tensor = tensor_parallel.vocab_parallel_cross_entropy(
            stu_output.contiguous().float(),
            labels
        )
    else:
        output_tensor, other_losses = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels
        )
    if (
                args.curriculum_learning_legacy and
                args.curriculum_seqlen < args.seq_length
    ):
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    moe_losses = []
    for moe_loss in other_losses:
        if moe_loss is not None:
            moe_losses.append(moe_loss)
    moe_loss = sum(moe_losses) * args.moe_loss_coeff

    mos_loss = 0
    if args.mos or args.kd:
        assert model.training
        if args.teacher_forward and args.teacher_model is not None:
            mos_loss = calculate_mos_loss(
                args,
                stu_output,
                args.teacher_model[0],
                tokens,
                position_ids,
                attention_mask
            )

    # Output_tensor stores the standard loss,
    # loss_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask, moe_loss, mos_loss)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    files = []
    if args.data_file_list is not None:
        with open(args.data_file_list, 'r') as flist:
            for f in flist.readlines():
                w, fname = f.split()
                files.append(float(w))
                files.append(fname)
    elif len(args.data_path) == 1 and os.path.isdir(args.data_path[0]):
        path = args.data_path[0] + "/"
        for f in os.listdir(path):
            if (os.path.isfile(path + f) and f.find(".bin") != -1):
                files.append(1)
                files.append(path + f.split(".bin")[0])
    else:
        files = args.data_path
    print_rank_0(f"file list {files}")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=files,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=True,
        # skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def command_exists(cmd):
    result = subprocess.Popen(
        f'type {cmd}',
        stdout=subprocess.PIPE,
        shell=True
    )
    return result.wait() == 0


def git_ds_info():
    if RANK != 0:
        return
    from deepspeed.env_report import main as ds_report
    ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(
        f'**** Git info for Megatron: '
        f'git_hash={git_hash} git_branch={git_branch} ****'
    )


def main():
    # if RANK == 0:
    #     setup_wandb()

    if os.getenv('TORCH_PROFILER_ENABLED') == '1':
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            # Initalize and get arguments, timers, and Tensorboard writer.
            initialize_megatron(
                                # extra_args_provider=extra_args_provider,
                                args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}, 
                                # external_args=external_args
                                )
            # Set pytorch JIT layer fusion options and warmup JIT functions.
            if get_accelerator().device_name() == 'cuda':
                set_jit_fusion_options()

            args = get_args()
            timers = get_timers()

            # model = model_provider()
            model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

        prof.export_chrome_trace(f"{args.tensorboard_dir}/torch-trace-{RANK}-of-{WORLD_SIZE}.json")
    else:
        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(
                            # extra_args_provider=extra_args_provider,
                            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}, 
                            # external_args=external_args
                            )
        # Set pytorch JIT layer fusion options and warmup JIT functions.
        if get_accelerator().device_name() == 'cuda':
            set_jit_fusion_options()

        args = get_args()
        timers = get_timers()

        if args.deepspeed:
            args.deepspeed_config_dict = _create_ds_config_dict()
            if "curriculum_learning" in args.deepspeed_config_dict and \
                "enabled" in args.deepspeed_config_dict["curriculum_learning"]:
                args.curriculum_learning_legacy = args.deepspeed_config_dict[ \
                    "curriculum_learning"]["enabled"]
            if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
                from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                    import CurriculumScheduler
                args.curriculum_scheduler = CurriculumScheduler( \
                    args.deepspeed_config_dict["curriculum_learning"])
            if "compression_training" in args.deepspeed_config_dict:
                args.compression_training = True

        # model = model_provider()
        # model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
        model = get_model(model_provider, ModelType.encoder_or_decoder) # works but does it load from a checkpoint or randomly initializes?
        # TRY deepspeed init and load_checkpoint directly here from model_ref = get_model(model_provider)
        optimizer = get_megatron_optimizer(model, None, None, 1.0)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
        model, optimizer, _, opt_param_scheduler = deepspeed.initialize(
                                                                model=model[0],
                                                                optimizer=optimizer,
                                                                args=args,
                                                                lr_scheduler=opt_param_scheduler,
                                                                mpu=mpu if args.no_pipeline_parallel else None,
                                                                config=args.deepspeed_config_dict,
                                                            )
        model = [model]
        print_rank_0(get_parameters_in_billions(model))
        #exit()

        # ---------- Reference model -------------
        # model_ref, _, _ = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder) # throwing assertion error
        model_ref = get_model(model_provider, ModelType.encoder_or_decoder) # works but does it load from a checkpoint or randomly initializes?
        # TRY deepspeed init and load_checkpoint directly here from model_ref = get_model(model_provider)
        optimizer_2 = get_megatron_optimizer(model_ref, None, None, 1.0)
        opt_param_scheduler_2 = get_optimizer_param_scheduler(optimizer_2)
        model_ref, optimizer_2, _, opt_param_scheduler_2 = deepspeed.initialize(
                                                                model=model_ref[0],
                                                                optimizer=optimizer_2,
                                                                args=args,
                                                                lr_scheduler=opt_param_scheduler_2,
                                                                mpu=mpu if args.no_pipeline_parallel else None,
                                                                config=args.deepspeed_config_dict,
                                                            )
        # model_ref, _, _, _ = deepspeed.initialize(
        #                         model=model_ref[0],
        #                         optimizer=None,
        #                         args=args,
        #                         lr_scheduler=None,
        #                         mpu=mpu if args.no_pipeline_parallel else None,
        #                         config=args.deepspeed_config_dict,
        #                     )
        # engine = deepspeed.init_inference(model=model_ref[0],
        #                             mp_size=args.tensor_model_parallel_size,
        #                             tensor_parallel={"mpu": mpu},
        #                             dtype=torch.half,
        #                             replace_with_kernel_inject=True,
        #                             # moe_experts=args.num_experts,
        #                             # moe_type=args.mlp_type
        #                             )
        # model_ref = engine.module


        if isinstance(model_ref, deepspeed.PipelineEngine):
            print(f'Doing assertion checks on model_ref..')
            # hack to get batch_fn from pretrain_gpt.py
            model_ref.set_batch_fn(model_ref.module._megatron_batch_fn)

            assert model_ref.grid.get_pipe_parallel_rank() == mpu.get_pipeline_model_parallel_rank()
            assert model_ref.grid.get_slice_parallel_rank() == mpu.get_tensor_model_parallel_rank()
            assert model_ref.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()

        model_ref = [model_ref]
        iteration2 = load_checkpoint(model_ref, optimizer_2, opt_param_scheduler_2) # THIS WORKED!! After commenting out assert args.consumed_train_samples == 0 in load_checkpoint()

        # THINGS THAT DID NOT WORK FOR LOADING FROM CHECKPOINT
        # model_ref, optimizer_ref, lr_scheduler_ref = load_model_weights_only(model_provider) # DID NOT WORK - train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size 32 != 8 * 1 * 8
        # model_ref, optimizer_ref, lr_scheduler_ref = load_model_weights_only_modified(model_provider) # DID NOT WORK -     optimizer = FusedAdam(TypeError: FusedAdam.__init__() got an unexpected keyword argument 'beta1'
        # ----------------------------------------

        if args.data_file_list_u is not None:
            print(f'data files list unpreferred: {args.data_file_list_u}')

            # Number of train/valid/test samples.
            if args.train_samples:
                print(f'args.train_samples: {args.train_samples}')
                train_samples = args.train_samples
            else:
                print(f'args.train_iters: {args.train_iters}')
                print(f'args.global_batch_size: {args.global_batch_size}')
                train_samples = args.train_iters * args.global_batch_size

            print(f'args.eval_interval: {args.eval_interval}')
            print(f'args.eval_iters: {args.eval_iters}')
            eval_iters = (args.train_iters // args.eval_interval + 1) * \
                        args.eval_iters
            test_iters = args.eval_iters
            train_val_test_num_samples = [train_samples,
                                        eval_iters * args.global_batch_size,
                                        test_iters * args.global_batch_size]
            print_rank_0(f'train_val_test_num_samples: {train_val_test_num_samples}')
            # print(f'args.data_impl: {args.data_impl}')
            # print(f'args.split: {args.split}')
            # print(f'args.seq_length: {args.seq_length}')
            # print(f'args.seed: {args.seed}')
            # print(f'args.train_data_path: {args.train_data_path}')
            # print(f'args.valid_data_path: {args.valid_data_path}')
            # print(f'args.test_data_path: {args.test_data_path}')
            # print(f'args.data_cache_path: {args.data_cache_path}')

            files_u = []
            with open(args.data_file_list_u, 'r') as flist:
                for f in flist.readlines():
                    w, fname = f.split()
                    files_u.append(float(w))
                    files_u.append(fname)
            train_ds_u, valid_ds_u, test_ds_u = build_train_valid_test_datasets(
            data_prefix=files_u,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=True,
            # skip_warmup=(not args.mmap_warmup),
            train_data_prefix=args.train_data_path,
            valid_data_prefix=args.valid_data_path,
            test_data_prefix=args.test_data_path,
            data_cache_path=args.data_cache_path)
            print_rank_0("> finished creating unpreferred GPT datasets ...")

        if args.data_file_list_p is not None:
            print_rank_0(f'data files list preferred: {args.data_file_list_p}')

            files_p = []
            with open(args.data_file_list_p, 'r') as flist:
                for f in flist.readlines():
                    w, fname = f.split()
                    files_p.append(float(w))
                    files_p.append(fname)
            train_ds_p, valid_ds_p, test_ds_p = build_train_valid_test_datasets(
            data_prefix=files_p,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=True,
            # skip_warmup=(not args.mmap_warmup),
            train_data_prefix=args.train_data_path,
            valid_data_prefix=args.valid_data_path,
            test_data_prefix=args.test_data_path,
            data_cache_path=args.data_cache_path)
            print_rank_0("> finished creating preferred GPT datasets ...")

        # Data loaders
        print_rank_0(f'args.consumed_train_samples: {args.consumed_train_samples}')
        print_rank_0(f'args.dataloader_type: {args.dataloader_type}')
        train_dataloader_u = build_pretraining_data_loader(
                                    train_ds_u, args.consumed_train_samples)
        train_dataloader_p = build_pretraining_data_loader(
                                    train_ds_p, args.consumed_train_samples)

        # Build train iterators
        dl_type = args.dataloader_type
        assert dl_type in ['single', 'cyclic']

        if train_dataloader_u is not None:
            print_rank_0(f'unpreferred train_dataloader is not None..')
            train_data_iterator_u = iter(train_dataloader_u) if dl_type == 'single' \
                                else iter(cyclic_iter(train_dataloader_u))
        print_rank_0("> finished creating unpreferred train_data_iterator...")
        if train_dataloader_p is not None:
            print_rank_0(f'preferred train_dataloader is not None..')
            train_data_iterator_p = iter(train_dataloader_p) if dl_type == 'single' \
                                else iter(cyclic_iter(train_dataloader_p))
        print_rank_0("> finished creating preferred train_data_iterator...")


        print_rank_0(f'args.train_iters: {args.train_iters}')
        print_rank_0(f'args.save_interval: {args.save_interval}')
        report_memory_flag = True

        # Train model
        model[0].train()

        if torch.distributed.get_rank() == 0:
            averaged_loss_iter = []
            averaged_rewards_iter = []
            avg_loss_epoch = []
            avg_rewards_epoch = []

        for epoch in range(1):
            iteration = 0
            for i in range(args.train_iters):
                # Get batch
                timers = get_timers()
                timers('batch-generator-unpreferred', log_level=2).start()
                tokens_u, labels_u, loss_mask_u, attention_mask_u, position_ids_u = get_batch(
                                                                                        train_data_iterator_u)
                timers('batch-generator-unpreferred').stop()
                # print_rank_0(f'tokens_u[0].size(): {tokens_u[0].size()}')
                # print_rank_0(f'tokens_u[0,400:1024]: {tokens_u[0,400:1024]}')
                print_rank_0("> finished extracting batch of tokens, labels, attn mask etc. for unpref train_data_iterator ...")

                timers('batch-generator-preferred', log_level=2).start()
                tokens_p, labels_p, loss_mask_p, attention_mask_p, position_ids_p = get_batch(
                                                                                        train_data_iterator_p)
                timers('batch-generator-preferred').stop()
                # print(f'tokens shape: {tokens_u.shape}')
                print_rank_0("> finished extracting batch of tokens, labels, attn mask etc. for pref train_data_iterator ...")

                # Model forward
                # output_tensor, other_losses = model[0](
                #                                 tokens_u,
                #                                 position_ids_u,
                #                                 attention_mask_u,
                #                                 labels=labels_u
                #                             ) # OUT OF MEMORY ERROR even with 4 nodes

                # Model forward with concatenated inputs
                tokens_c = torch.cat((tokens_p,tokens_u), 0)
                position_ids_c = torch.cat((position_ids_p,position_ids_u), 0)
                labels_c = torch.cat((labels_p,labels_u), 0)
                loss_mask_c = torch.cat((loss_mask_p,loss_mask_u), 0)

                # Logits and loss
                output_c, other_losses_c = model[0](
                                    tokens_c,
                                    position_ids_c,
                                    None,
                                    # labels=labels_u
                                )

                loss_c = tensor_parallel.vocab_parallel_cross_entropy(
                                output_c.contiguous().float(),
                                labels_c
                            )

                # Reference model forward with concatenated inputs
                with torch.no_grad():
                    # Logits and loss
                    routput_c, rother_losses_c = model_ref[0](
                            tokens_c,
                            position_ids_c,
                            None,
                            # labels=labels_u
                        )
                    rloss_c = tensor_parallel.vocab_parallel_cross_entropy(
                        routput_c.contiguous().float(),
                        labels_c
                    )

                # # Print statements for debugging
                # print(f'tokens_p: {tokens_p}')
                # print(f'tokens_u: {tokens_u}')
                # # print(f'output_p[0]: {output_p[0]}')
                # # print(f'output_u[0]: {output_u[0]}')
                # print(f'output_c[0]: {output_c[0]}')
                # print(f'tokens_p shape: {tokens_p.size()}, tokens_u shape: {tokens_u.size()}')
                # print(f'tokens_c shape: {tokens_c.size()}')
                # print(f'position_ids_p shape: {position_ids_p.size()}, position_ids_u shape: {position_ids_u.size()}')
                # print(f'position_ids_c shape: {position_ids_c.size()}')
                # print(f'output_c shape: {output_c.size()}')
                # print(f'loss_c shape: {loss_c.size()}')
                # print(f'routput_c shape: {routput_c.size()}')
                # print(f'rloss_c shape: {rloss_c.size()}')
                # print(f'loss_mask_p shape: {loss_mask_p.size()}')
                # print(f'loss_mask_u shape: {loss_mask_u.size()}')
                # print(f'loss_mask_c shape: {loss_mask_c.size()}')
                # print(f'attention_mask_u: {attention_mask_u}')
                # print(f'loss_mask_p sum: {torch.sum(loss_mask_p), 8*4096}')# print(f'loss_mask_p shape: {loss_mask_p.size()}')

                # Seq logprobs
                print_rank_0(f'args.micro_batch_size: {args.micro_batch_size}')
                seq_logps_p = torch.sum(loss_c[:args.micro_batch_size,:] * loss_mask_p, dim=-1) / torch.sum(loss_mask_p, dim=-1)
                seq_logps_u = torch.sum(loss_c[args.micro_batch_size:,:] * loss_mask_u, dim=-1) / torch.sum(loss_mask_u, dim=-1)
                rseq_logps_p = torch.sum(rloss_c[:args.micro_batch_size,:] * loss_mask_p, dim=-1) / torch.sum(loss_mask_p, dim=-1)
                rseq_logps_u = torch.sum(rloss_c[args.micro_batch_size:,:] * loss_mask_u, dim=-1) / torch.sum(loss_mask_u, dim=-1)

                # # Print statements for debugging
                # print(f'seq_logps_p shape: {seq_logps_p.size()}')
                # print(f'seq_logps_u shape: {seq_logps_u.size()}')
                # print(f'rseq_logps_p shape: {rseq_logps_p.size()}')
                # print(f'rseq_logps_u shape: {rseq_logps_u.size()}')

                # Loss
                pu_ratio = seq_logps_p - seq_logps_u
                rpu_ratio = rseq_logps_p - rseq_logps_u
                sdiff_ratio = 0.1*(pu_ratio - rpu_ratio)
                # print(f'sdiff_ratio: {sdiff_ratio}')
                final = -F.logsigmoid(sdiff_ratio)
                # print(f'final: {final}')
                # dloss = torch.sum(final)
                dloss = torch.mean(final)
                
                # Model backward and update
                model[0].backward(dloss)

                increment = get_num_microbatches() * \
                                            args.micro_batch_size * \
                                            args.data_parallel_size
                # print(f'increment: {increment}')
                # model[0].step(lr_kwargs={'increment': increment})
                model[0].step()
                update_successful = model[0].was_step_applied()
                print_rank_0(f'update_successful: {update_successful}')

                # Iteration updates
                iteration += 1
                args.iteration = iteration
                # print(f'args.consumed_train_samples: {args.consumed_train_samples}')
                new_samples = mpu.get_data_parallel_world_size() * \
                                            args.micro_batch_size * \
                                            get_num_microbatches()

                
                args.consumed_train_samples += new_samples
                # print(f'args.consumed_train_samples: {args.consumed_train_samples}')

                # Reduce loss for logging.
                averaged_loss = average_losses_across_data_parallel_group([dloss])
                loss_dict = {'loss': averaged_loss}
                print_rank_0(f'iteration: {iteration}, dloss: {averaged_loss.detach().cpu().tolist()}')
                psrewards_p = (0.1 * (seq_logps_p - rseq_logps_p)).detach()
                psrewards_u = (0.1 * (seq_logps_u - rseq_logps_u)).detach()
                psrewards = (psrewards_p > psrewards_u).float()
                rewards = psrewards.cpu().mean()
                print_rank_0(f'iteration: {iteration}, rewards: {rewards}')

                # wandb logging
                # report_memory_flag = training_log_dpo(loss_dict, iteration, report_memory_flag)

                if torch.distributed.get_rank() == 0:
                    averaged_loss_iter.append(averaged_loss.detach().cpu().tolist()[0])
                    averaged_rewards_iter.append(rewards.tolist())

                if (i % args.save_interval == 0) and (i > 0) and (torch.distributed.get_rank() == 0):
                    TPL = args.tensor_model_parallel_size
                    GRAD_ACC = os.environ.get('GRAD_ACC_STEPS')
                    print(f'Checkpointing loss and rewards at iteration {i} ..')
                    np.savez(f'./runs/loss-rewards_indels_textseq_nranks-{WORLD_SIZE}_model-nlayers-{args.num_layers}_TP-{TPL}_zero-{args.zero_stage}_gradacc-{GRAD_ACC}_lr-{args.lr}_seq-{args.seq_length}_bs-{args.micro_batch_size}_iters-{args.train_iters}-chkpt-{i}.npz', loss=np.array(averaged_loss_iter), rewards=np.array(averaged_rewards_iter))

            # if torch.distributed.get_rank() == 0:
            #     avg_loss_epoch.append(np.array(averaged_loss_iter).mean())
            #     avg_rewards_epoch.append(np.array(averaged_rewards_iter).mean())

        # Aggregated loss and rewards
        # torch.distributed.barrier()
        # if torch.distributed.get_rank() == 0:
            # print(averaged_loss_iter)
            # print(averaged_rewards_iter)
            # print(avg_loss_epoch)
            # print(avg_rewards_epoch)
            # np.savez(f'./runs/proteingym_indels/loss-rewards_iters-{args.train_iters}.npz', loss=np.array(averaged_loss_iter), rewards=np.array(averaged_rewards_iter))

        # Generate - NOT WORKING
        if False:
            model[0].eval()
            print_rank_0(f'Generation mode..')
            print_rank_0(f'args.seq_length: {args.seq_length}')
            tokenizer = get_tokenizer()
            print_rank_0(f'len(tokenizer.vocab): {len(tokenizer.vocab)}')
            prompts=["A sequence", "A sequence","A sequence", "A sequence", "A sequence"]
            tokens_to_generate = 64
            add_BOS = False
            if add_BOS:
                prompts_tokens = [[tokenizer.eod] + tokenizer.tokenize(prompt)
                                for prompt in prompts]
            else:
                prompts_tokens = [tokenizer.tokenize(prompt) for prompt in prompts]

            print_rank_0(f'prompts_tokens: {prompts_tokens}')

            # Now we have a list of list of tokens which each list has a different
            # size. We want to extend this list to:
            #   - incorporate the tokens that need to be generated
            #   - make all the sequences equal length.
            # Get the prompts length.
            prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
            # Get the max prompts length.
            max_prompt_len = max(prompts_length)
            # Number of tokens in the each sample of the batch.
            samples_length = max_prompt_len + tokens_to_generate
            # Now update the list of list to be of the same size: samples_length.
            for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
                padding_size = samples_length - prompt_length
                prompt_tokens.extend([tokenizer.eod] * padding_size)

            # Now we are in a structured format, we can convert to tensors.
            prompts_tokens_tensor = torch.cuda.LongTensor(prompts_tokens)
            prompts_length_tensor = torch.cuda.LongTensor(prompts_length)
            print_rank_0(f'prompts_tokens_tensor: {prompts_tokens_tensor}')
            print_rank_0(f'prompts_length_tensor: {prompts_length_tensor}')

            batch_size = prompts_tokens_tensor.size(0)
            min_prompt_length = prompts_length_tensor.min().item()
            max_sequence_length = prompts_tokens_tensor.size(1)

            print_rank_0(f'batch_size: {batch_size}')
            print_rank_0(f'min_prompt_length: {min_prompt_length}')
            print_rank_0(f'max_sequence_length: {max_sequence_length}')
            print_rank_0(f'max_position_embeddings: {args.max_position_embeddings}')
            print_rank_0(f'args.max_tokens_to_oom: {args.max_tokens_to_oom}')
            if max_sequence_length > args.max_position_embeddings:
                raise ValueError("Length of prompt + tokens_to_generate longer than allowed")

            if max_sequence_length * batch_size > args.max_tokens_to_oom:
                raise ValueError("Too many tokens.  " + str(max_sequence_length*batch_size)+ " is greater than "+str(args.max_tokens_to_oom))

            # INSTANTIATING FORWARD_STEP ?
            model_fwd = ForwardStep(model[0], batch_size, max_sequence_length)
            inference_params = InferenceParams(batch_size,
                                                max_sequence_length)

            if hasattr(args, 'eos_id'):
                termination_id = args.eos_id
                print(f'args.eos_id: {args.eos_id}')
            else:
                termination_id = tokenizer.eod
                print(f'tokenizer.eod: {tokenizer.eod}')

            # Log probability of the sequence (prompt + generated tokens).
            output_log_probs = None
            output_log_probs_size = (batch_size, max_sequence_length - 1)
            # Lengths of generated seuquence including including prompts.
            generated_sequence_lengths = None
            top_k = 0
            top_p = 1.0
            temperature = 1.0
            top_p_decay=0.0
            top_p_bound=0.0
            add_BOS=False
            use_eod_token_for_early_termination=True
            stop_on_double_eol=False
            stop_on_eol=False
            prevent_newline_after_colon=False
            random_seed=42
            return_output_log_probs = False
            if mpu.is_pipeline_last_stage():
                if return_output_log_probs:
                    output_log_probs = torch.empty(output_log_probs_size,
                                                dtype=torch.float32,
                                                device=torch.cuda.current_device())
                    print_rank_0(f'On mpu.is_pipeline_last_stage branch and output_log_probs is set: {output_log_probs}')
                generated_sequence_lengths = torch.ones(
                        batch_size, dtype=torch.int64,
                        device=torch.cuda.current_device()) * max_sequence_length
                print_rank_0(f'On mpu.is_pipeline_last_stage branch and generated_sequence_lengths: {generated_sequence_lengths}')

            # Whether we have reached a termination id.
            is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                            device=torch.cuda.current_device())


            with torch.no_grad():
                prompts_attention_mask, _, prompts_position_ids = get_ltor_masks_and_position_ids(
                                                            data=prompts_tokens_tensor,
                                                            eod_token=None,
                                                            reset_position_ids=False,
                                                            reset_attention_mask=False,
                                                            eod_mask_loss=False
                                                            )
                prev_context_length = 0
                for context_length in range(min_prompt_length, max_sequence_length):
                    # Pick the slice that we need to pass through the network.
                    tokens2use = prompts_tokens_tensor[:, prev_context_length:context_length]
                    positions2use = prompts_position_ids[:, prev_context_length:context_length]
                    attention_mask2use = prompts_attention_mask[
                        ..., prev_context_length:context_length, :context_length]

                    # logits will be meanigful only in the last pipeline stage.
                    # print_rank_0(f'tokens2use shape: {tokens2use.size()}')
                    # print_rank_0(f'positions2use shape: {positions2use.size()}')
                    # print_rank_0(f'attention_mask2use shape: {attention_mask2use.size()}')
                    # print_rank_0(f'prompts_tokens_tensor shape: {prompts_tokens_tensor.size()}')
                    # print_rank_0(f'prompts_position_ids shape: {prompts_position_ids.size()}')
                    # print_rank_0(f'prompts_attention_mask shape: {prompts_attention_mask.size()}')

                    # ------
                    # plogits = forward_step(tokens2use, positions2use, attention_mask2use)
                    # plogits = plogits[0]
                    # print_rank_0(f'context_length: {context_length}, plogits: {plogits}')

                    # plogits = model[0](prompts_tokens_tensor, 
                    #                     prompts_position_ids, 
                    #                     prompts_attention_mask, 
                    #                     inference_params=inference_params
                    #                 )
                    # print_rank_0(f'logits: {plogits}')
                    #-------
                    inference_params = InferenceParams(batch_size,
                                                tokens2use.size(1))
                    plogits = model[0](tokens2use, 
                                        positions2use, 
                                        attention_mask2use, 
                                        inference_params=inference_params
                                    )
                    plogits = plogits[0]
                    # plogits = torch.cuda.FloatTensor(plogits)
                    # print_rank_0(f'plogits: {plogits.size()}')
                    # print_rank_0(f'plogits type: {plogits.dtype}')

                    if mpu.is_pipeline_last_stage():
                        if prevent_newline_after_colon:
                            plogits[tokens2use[:, -1] == tokenizer.tokenize(':')[0], -1, tokenizer.tokenize('\n')[0]] = -1e10 # disable "\n" after ":"
                        # Always the last stage should have an output.
                        assert plogits is not None

                        # Sample.
                        last_token_logits = plogits[:, -1, :]
                        new_sample = sample(last_token_logits,
                                            top_k=top_k,
                                            top_p=top_p,
                                            temperature=temperature,
                                            vocab_size=tokenizer.vocab_size)
                        if top_p > 0.0 and top_p_decay > 0.0:
                            top_p = top_p * top_p_decay
                            if top_p_bound > 0.0:
                                top_p = max(top_p, top_p_bound)
                        print_rank_0(f'new_sample: {new_sample}')
                        for nidx, ns in enumerate(new_sample.cpu().numpy().tolist()):
                            print_rank_0(f'nidx: {nidx}, new_sample[{nidx}]: {tokenizer.detokenize(ns)}')
                        # If a prompt length is smaller or equal th current context
                        # length, it means we have started generating tokens
                        started = prompts_length_tensor <= context_length
                        # Update the tokens.
                        print_rank_0(f'started: {started}')
                        # print_rank_0(f'prompts_tokens_tensor before copying new_sample: {prompts_tokens_tensor}')
                        for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
                            print_rank_0(f'nidx: {nidx}, prompts_tokens_tensor before[{nidx}]: {tokenizer.detokenize(ns)}')
                        prompts_tokens_tensor[started, context_length] = new_sample[started]
                        # print_rank_0(f'prompts_tokens_tensor after copying new_sample: {prompts_tokens_tensor}')
                        for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
                            print_rank_0(f'nidx: {nidx}, prompts_tokens_tensor after[{nidx}]: {tokenizer.detokenize(ns)}')

                    # Update the tokens on the first stage so the next input to
                    # the network is correct.
                    copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                        prompts_tokens_tensor[:, context_length])
                    for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
                        print_rank_0(f'nidx: {nidx}, prompts_tokens_tensor after copy_from_last_to_first_pipeline_stage [{nidx}]: {tokenizer.detokenize(ns)}')

                    # Update the context length for the next token generation.
                    prev_context_length = context_length
                    print_rank_0(f'prev_context_length: {prev_context_length}')

                    # Check if all the sequences have hit the termination_id.
                    done = None
                    if mpu.is_pipeline_last_stage():
                        # These stopping methods are tokenizer dependent
                        # instead tokenization should be in the inference loop so stop sequences can be used
                        if stop_on_double_eol:
                            hit_double_eol = (new_sample == 628).byte() & started.byte()
                            hit_two_eols = (new_sample == 198).byte() & (tokens[:, context_length-1] == 198).byte() & started.byte()
                            done_token = hit_double_eol | hit_two_eols
                        elif stop_on_eol:
                            hit_double_eol = (new_sample == 628).byte() & started.byte()
                            hit_eol = (new_sample == 198).byte() & started.byte()
                            done_token = hit_double_eol | hit_eol
                        else:
                            done_token = (new_sample == termination_id).byte() & \
                                started.byte()

                        just_finished = (done_token & ~is_generation_done).bool()
                        generated_sequence_lengths[just_finished.view(-1)] = \
                            context_length + 1
                        is_generation_done = is_generation_done | done_token
                        done = torch.all(is_generation_done)
                    done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                            tensor=done)
                    if use_eod_token_for_early_termination and done:
                        print_rank_0(f'done: {done}')
                        break

            # ===================================================
            # Update the length of based on max generated length.
            # ===================================================
            for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
                print_rank_0(f'nidx: {nidx}, detokenized prompts_tokens_tensor after the generate loop [{nidx}]: {tokenizer.detokenize(ns)}')
            prompts_tokens_tensor = prompts_tokens_tensor[:, :(context_length + 1)]
            for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
                print_rank_0(f'nidx: {nidx}, detokenized prompts_tokens_tensor after the generate loop and slicing with ctx length[{nidx}]: {tokenizer.detokenize(ns)}')
            if mpu.is_pipeline_last_stage():
                if return_output_log_probs:
                    output_log_probs = output_log_probs[:, :context_length]

            # ======================================
            # Broadcast to the first pipeline stage.
            # ======================================

            generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
                batch_size, torch.int64, generated_sequence_lengths)
            if return_output_log_probs:
                output_log_probs_size = (batch_size, context_length)
                output_log_probs = broadcast_from_last_to_first_pipeline_stage(
                    output_log_probs_size, torch.float32, output_log_probs)

            for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
                print_rank_0(f'nidx: {nidx}, detokenized prompts_tokens_tensor after the generate loop and befoer final post-process[{nidx}]: {tokenizer.detokenize(ns)}')
            # Only post-process on first stage.
            if mpu.is_pipeline_first_stage():
                prompts_plus_generations = []

                for nidx, ns in enumerate(prompts_tokens_tensor.cpu().numpy().tolist()):
                    print_rank_0(f'nidx: {nidx}, detokenized prompts_tokens_tensor after the generate loop and after final post-process[{nidx}]: {tokenizer.detokenize(ns)}')

                rtokens = prompts_tokens_tensor.cpu().numpy().tolist()
                rlengths = prompts_length_tensor.cpu().numpy().tolist()
                print_rank_0(f'rlengths: {rlengths}')
                for sequence_tokens, slength in zip(rtokens, rlengths):
                    sequence_tokens = sequence_tokens[:slength]
                    prompts_plus_generations.append(
                        tokenizer.detokenize(sequence_tokens))
                # _, prompts_plus_generations, prompts_plus_generations_segments = \
                #     detokenize_generations(prompts_tokens_tensor, prompts_length_tensor, True)

            print_rank_0(f'prompts_plus_generations: {prompts_plus_generations}')

        if True:
            prompts=["Pen is mightier than", "A sequence", "Pythagoras theorem", "A sequence", "Hello world"]
            tokens_to_generate = 64
            generated_responses = generate_post_training(model, prompts, tokens_to_generate, fprint=False)

        if False:
            print_rank_0(f'Generation mode..')
            print_rank_0(f'args.seq_length: {args.seq_length}')
            tokenizer = get_tokenizer()
            print_rank_0(f'len(tokenizer.vocab): {len(tokenizer.vocab)}')
            model[0].eval()
            choice = torch.cuda.LongTensor(1)
            torch.distributed.broadcast(choice, 0)
            # if choice[0].item() == 0:
            try:
                tokens_to_generate_len = 1021
                response, _, _, _ = generate_and_post_process(model[0], prompts=["A sequence", "A sequence","A sequence", "A sequence", "A sequence", "A sequence","A sequence", "A sequence"], tokens_to_generate=tokens_to_generate_len)
                print_rank_0(f'generation completed..\n response:{response}')
            except ValueError as ve:
                print_rank_0(f'ValueError: {ve}')
                pass
        # elif choice[0].item() == 1:
        #     try:
        #         response, _, _ = beam_search_and_post_process(model[0], prompts=["A sequence", "A sequence", "A sequence", "A sequence",], tokens_to_generate=32)
        #         print(f'generation completed..\n response:{response}')
        #     except ValueError as ve:
        #         print(f'ValueError: {ve}')
        #         pass

        # # Checkpointing
        # if args.save and iteration != 0:
        #         save_checkpoint(iteration, model, optimizer, opt_param_scheduler)

    return model

# def main():
#     # if RANK == 0:
#     #     setup_wandb()
#     if os.getenv('TORCH_PROFILER_ENABLED') == '1':
#         from torch.profiler import profile, record_function, ProfilerActivity
#         with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#             model = pretrain(
#                 train_valid_test_datasets_provider,
#                 model_provider,
#                 ModelType.encoder_or_decoder,
#                 forward_step,
#                 args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
#                 data_post_process=data_post_process
#             )

#         prof.export_chrome_trace(f"{args.tensorboard_dir}/torch-trace-{RANK}-of-{WORLD_SIZE}.json")
#     else:
#         model = pretrain(
#             train_valid_test_datasets_provider,
#             model_provider,
#             ModelType.encoder_or_decoder,
#             forward_step,
#             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
#             data_post_process=data_post_process
#         )
#     return model


if __name__ == "__main__":
    # git_ds_info()
    # pretrain(train_valid_test_datasets_provider,
    #          model_provider,
    #          ModelType.encoder_or_decoder,
    #          forward_step,
    #          args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    #          data_post_process=data_post_process)
    import sys
    import deepspeed.comm as dist

    # Return trained model
    model = main()

    dist.log_summary()
    if wandb.run is not None:
        print(f"wandb.run.name: {wandb.run.name}")
        print(f"wandb.run.url: {wandb.run.url}")
        wandb.finish()
    sys.exit()
