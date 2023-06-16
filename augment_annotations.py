import codecs
import copy
import csv
import logging
import math
import sys
import os
import random
from typing import Tuple, List

os.environ['WANDB_DISABLED'] = 'true'

import nltk
import numpy as np
from transliterate import translit
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig


TARGET_SAMPLING_RATE = 16_000
dataset_augmentation_logger = logging.getLogger(__name__)


def load_metadata(fname: str) -> List[Tuple[str, str]]:
    true_header = ['file_name', 'transcription']
    loaded_header = []
    res = []
    with codecs.open(fname, mode='r', encoding='utf-8') as fp:
        data_reader = csv.reader(fp, quotechar='"', delimiter=',')
        line_idx = 1
        for row in data_reader:
            if len(row) > 0:
                err_msg = f'File "{fname}": line {line_idx} is wrong!'
                if len(loaded_header) == 0:
                    loaded_header = copy.copy(row)
                    if loaded_header != true_header:
                        err_msg += f' {loaded_header} != {true_header}'
                        raise ValueError(err_msg)
                else:
                    if len(row) != len(loaded_header):
                        err_msg += f' {len(row)} != {len(loaded_header)}'
                        raise ValueError(err_msg)
                    sound_path = os.path.join(os.path.dirname(fname), os.path.normpath(row[0]))
                    sound_annotation = ' '.join(row[1].strip().split())
                    if not os.path.isfile(sound_path):
                        err_msg += f' The sound "{sound_path}" does not exist!'
                        raise ValueError(err_msg)
                    res.append((row[0], sound_annotation))
            line_idx += 1
    return res


def initialize_paraphraser(paraphraser_model_path: str, dataset: List[Tuple[str, str]], max_paraphrases: int) -> \
        Tuple[T5Tokenizer, T5ForConditionalGeneration, GenerationConfig]:
    model_ = T5ForConditionalGeneration.from_pretrained(paraphraser_model_path, device_map='auto',
                                                        load_in_8bit=True)
    tokenizer_ = T5Tokenizer.from_pretrained(paraphraser_model_path)
    model_.eval()
    max_size = min(max(round(1.5 * max(map(lambda it: len(it[1]), dataset))), 10), 512)
    config_ = GenerationConfig(
        encoder_no_repeat_ngram_size=4, early_stopping=True, do_sample=True,
        num_beams=7, max_new_tokens=max_size, no_repeat_ngram_size=4,
        num_return_sequences=max_paraphrases, output_scores=True, return_dict_in_generate=True
    )
    return tokenizer_, model_, config_


def normalize_text(s: str) -> str:
    normalized = s.strip()
    if len(normalized) == 0:
        return ''
    normalized = ' '.join(s.split()).strip()
    if len(normalized) == 0:
        return ''
    normalized = ' '.join(list(
        filter(
            lambda it2: (len(it2) > 0) and it2.isalnum(),
            map(
                lambda it1: it1.lower().strip(),
                nltk.wordpunct_tokenize(normalized)
            )
        )
    )).strip().replace('ё', 'е')
    return translit(normalized, 'ru')


def paraphrase(src: List[Tuple[str, str]], tokenizer: T5Tokenizer, paraphraser: T5ForConditionalGeneration,
               config: GenerationConfig, batch_size: int) -> List[List[str]]:
    n_batches = math.ceil(len(src) / batch_size)
    prepared = []
    row_size = 0
    data_part_size = math.ceil(n_batches / 20)
    data_part_counter = 0
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(len(src), batch_start + batch_size)
        texts_in_batch = [cur[1] for cur in src[batch_start:batch_end]]
        x = tokenizer(texts_in_batch, return_tensors='pt', padding=True).to(paraphraser.device)
        out = paraphraser.generate(**x, generation_config=config)
        for sample_idx in range(len(texts_in_batch)):
            scores = []
            paraphrases = []
            for variant_idx in range(config.num_return_sequences):
                scores.append(float(out.sequences_scores[variant_idx + sample_idx * config.num_return_sequences]))
                paraphrases.append(tokenizer.decode(
                    token_ids=out.sequences[variant_idx + sample_idx * config.num_return_sequences],
                    skip_special_tokens=True
                ))
            normalized_paraphrases = {normalize_text(texts_in_batch[sample_idx])}
            filtered_paraphrases = []
            filtered_scores = []
            for v, score in zip(paraphrases, scores):
                k = normalize_text(v)
                if k not in normalized_paraphrases:
                    normalized_paraphrases.add(k)
                    filtered_paraphrases.append(v)
                    filtered_scores.append(float(np.exp(score)))
            if len(filtered_scores) > 0:
                score_sum = sum(filtered_scores)
                for variant_idx in range(len(filtered_scores)):
                    filtered_scores[variant_idx] /= score_sum
            new_line = [
                src[batch_start + sample_idx][0],
                src[batch_start + sample_idx][1],
                normalize_text(texts_in_batch[sample_idx])
            ]
            for variant_idx in range(len(filtered_scores)):
                new_line += [filtered_paraphrases[variant_idx], f'{filtered_scores[variant_idx]}']
            for variant_idx in range(len(filtered_scores), config.num_return_sequences):
                new_line += ['', '']
            assert len(new_line) > 0
            if row_size == 0:
                row_size = len(new_line)
            else:
                assert row_size == len(new_line), f'{row_size} != {len(new_line)}'
            prepared.append(new_line)
            del filtered_paraphrases, filtered_scores, paraphrases, scores, normalized_paraphrases, new_line
        if (batch_idx + 1) % data_part_size == 0:
            data_part_counter += 1
            if data_part_counter < 20:
                info_msg = f'{round(100.0 * (data_part_counter / 20))}% of texts are paraphrased.'
                dataset_augmentation_logger.info(info_msg)
    dataset_augmentation_logger.info('100% of texts are paraphrased.')
    return prepared


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    if len(sys.argv) < 2:
        err_msg = 'The dataset path is not specified!'
        dataset_augmentation_logger.error(err_msg)
        raise ValueError(err_msg)
    dataset_name = os.path.normpath(sys.argv[1])
    if not os.path.isdir(dataset_name):
        err_msg = f'Directory "{dataset_name}" does not exist!'
        dataset_augmentation_logger.error(err_msg)
        raise ValueError(err_msg)

    if len(sys.argv) < 3:
        err_msg = 'The paraphraser name is not specified!'
        dataset_augmentation_logger.error(err_msg)
        raise ValueError(err_msg)
    paraphraser_name = os.path.normpath(sys.argv[2])
    if not os.path.isdir(paraphraser_name):
        if os.path.isdir(os.path.normpath(paraphraser_name)):
            paraphraser_name = os.path.normpath(paraphraser_name)

    if len(sys.argv) < 4:
        err_msg = 'The mini-batch size is not specified!'
        dataset_augmentation_logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        minibatch_size = int(sys.argv[3])
    except:
        minibatch_size = -1
    if minibatch_size <= 0:
        err_msg = f'{sys.argv[3]} is a wrong value for the mini-batch size!'
        dataset_augmentation_logger.error(err_msg)
        raise ValueError(err_msg)

    if len(sys.argv) < 5:
        max_paraphrases = 7
    else:
        try:
            max_paraphrases = int(sys.argv[4])
        except:
            max_paraphrases = -1
        if max_paraphrases <= 0:
            err_msg = f'{sys.argv[3]} is a wrong value for the paraphrase number!'
            dataset_augmentation_logger.error(err_msg)
            raise ValueError(err_msg)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        dataset_augmentation_logger.error(err_msg)
        raise ValueError(err_msg)

    metadata_fname = os.path.join(dataset_name, 'metadata.csv')
    try:
        dataset = load_metadata(metadata_fname)
    except BaseException as ex:
        err_msg = str(ex)
        dataset_augmentation_logger.error(err_msg)
        raise
    dataset_augmentation_logger.info(f'The metadata for the "{dataset_name}" is loaded! '
                                     f'The number of samples is {len(dataset)}.')

    try:
        tokenizer, paraphraser, paraphrasing_config = initialize_paraphraser(
            paraphraser_model_path=paraphraser_name,
            dataset=dataset,
            max_paraphrases=max_paraphrases
        )
    except BaseException as ex:
        err_msg = str(ex)
        dataset_augmentation_logger.error(err_msg)
        raise
    dataset_augmentation_logger.info(f'The paraphraser is loaded from the "{paraphraser_name}".')

    try:
        enriched_dataset = paraphrase(
            src=dataset,
            tokenizer=tokenizer,
            paraphraser=paraphraser,
            config=paraphrasing_config,
            batch_size=minibatch_size
        )
    except BaseException as ex:
        err_msg = str(ex)
        dataset_augmentation_logger.error(err_msg)
        raise
    dataset_augmentation_logger.info('All text annotations are paraphrased!')

    header = ['file_name', 'transcription', 'normalized']
    for variant_idx in range(1, max_paraphrases + 1):
        header += [f'paraphrase{variant_idx}', f'score{variant_idx}']
    for idx, row in enumerate(enriched_dataset):
        if len(row) != len(header):
            err_msg = f'Row {idx} of the enriched dataset is wrong! Expected {len(header)}, got {len(row)}.'
            dataset_augmentation_logger.error(err_msg)
            raise ValueError(err_msg)

    with codecs.open(metadata_fname, mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, quotechar='"', delimiter=',')
        data_writer.writerow(header)
        for row in enriched_dataset:
            data_writer.writerow(row)


if __name__ == '__main__':
    dataset_augmentation_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    dataset_augmentation_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('dataset_augmentation.log')
    file_handler.setFormatter(formatter)
    dataset_augmentation_logger.addHandler(file_handler)
    main()