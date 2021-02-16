# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Authors: Yossi Adi (adiyoss)

import argparse
import logging
import os
import sys

import librosa
import torch
import tqdm

from .data.data import EvalDataLoader, EvalDataset
from . import distrib
from .utils import remove_pad, normalized_cross_correlation

from .utils import bold, deserialize_model, LogProgress
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser("Speech separation using MulCat blocks")
parser.add_argument("model_path", type=str, help="Model name")
parser.add_argument("out_dir", type=str, default="exp/result", help="Directory putting enhanced wav files")
parser.add_argument("--mix_dir", type=str, default=None, help="Directory including mix wav files")
parser.add_argument("--mix_json", type=str, default=None, help="Json file including mix wav files")
parser.add_argument("--device", default="cuda")
parser.add_argument("--sample_rate", default=8000, type=int, help="Sample rate")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("-v", "--verbose", action="store_const", const=logging.DEBUG, default=logging.INFO, help="More loggging")
parser.add_argument("--window-size", type=int, default=420, help="Sliding window size in seconds")
parser.add_argument("--stride", type=int, default=390, help="Sliding window stride in seconds")


def save_wavs(estimate_source, mix_sig, lengths, filenames, out_dir, sr=8000):
    # Remove padding and flat
    flat_estimate = remove_pad(estimate_source, lengths)
    mix_sig = remove_pad(mix_sig, lengths)
    # Write result
    for i, filename in enumerate(filenames):
        filename = os.path.join(
            out_dir, os.path.basename(filename).strip(".wav"))
        write(mix_sig[i], filename + ".wav", sr=sr)
        C = flat_estimate[i].shape[0]
        # future support for wave playing
        for c in range(C):
            write(flat_estimate[i][c], filename + f"_s{c + 1}.wav")


def write(inputs, filename, sr=8000):
    librosa.output.write_wav(filename, inputs, sr, norm=True)


def get_mix_paths(args):
    mix_dir = None
    mix_json = None
    # fix mix dir
    try:
        if args.dset.mix_dir:
           mix_dir = args.dset.mix_dir
    except:
        mix_dir = args.mix_dir

    # fix mix json
    try:
        if args.dset.mix_json:
            mix_json = args.dset.mix_json
    except:
        mix_json = args.mix_json
    return mix_dir, mix_json


def align_estimated_sources(separated_audio, stride, lengths):
    previous_piece = separated_audio[0]
    overlap = separated_audio[0].shape[-1] - stride
    for i, audio_piece in enumerate(separated_audio[1:]):
        length_mask = (i * stride + audio_piece.shape[-1]) < lengths

        previous_piece_overlap = previous_piece[length_mask][..., -overlap:].contiguous()
        new_piece_overlap = audio_piece[length_mask][..., :overlap].contiguous()
        direct_cross_correlation = normalized_cross_correlation(previous_piece_overlap, new_piece_overlap)
        inverse_cross_correlation = normalized_cross_correlation(previous_piece_overlap, torch.flip(new_piece_overlap, dims=[1]))
        inverse_mask = inverse_cross_correlation > direct_cross_correlation

        # combine inverse and length mask into one
        _mask = length_mask
        _mask[length_mask] = inverse_mask
        inverse_mask = _mask

        audio_piece[inverse_mask] = torch.flip(audio_piece[inverse_mask], dims=[1])
        previous_piece = audio_piece
    return torch.cat(separated_audio, dim=-1)


def separate(args, model=None, local_out_dir=None):
    mix_dir, mix_json = get_mix_paths(args)
    if not mix_json and not mix_dir:
        logger.error("Must provide mix_dir or mix_json! "
                     "When providing mix_dir, mix_json is ignored.")
    # Load model
    if not model:
        # model
        pkg = torch.load(args.model_path)
        if 'model' in pkg:
            model = pkg['model']
        else:
            model = pkg
        model = deserialize_model(model)
        logger.debug(model)
    model.eval()
    model.to(args.device)
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir

    # Load data
    eval_dataset = EvalDataset(
        mix_dir,
        mix_json,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
    )
    eval_loader = distrib.loader(
        eval_dataset, batch_size=1, klass=EvalDataLoader)

    if distrib.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    distrib.barrier()

    window_size = args.window_size * args.sample_rate
    stride = args.stride * args.sample_rate

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(eval_loader, ncols=120)):
            separated_audio = []
            # Get batch data
            mixture, lengths, filenames = data
            lengths = lengths.to(args.device)
            for j in range(mixture.shape[-1] // stride + 1):
                mixture_piece = mixture[..., stride*j:stride*j + window_size].to(args.device)
                # Forward
                separated_audio.append(model(mixture_piece)[-1])

            estimate_sources = align_estimated_sources(separated_audio, stride, lengths)
            # save wav files
            save_wavs(estimate_sources, mixture, lengths, filenames, out_dir, sr=args.sample_rate)


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    separate(args, local_out_dir=args.out_dir)
