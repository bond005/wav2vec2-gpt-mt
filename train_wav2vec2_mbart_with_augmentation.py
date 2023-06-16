import gc
import logging
import re
import sys
import os
import random
from typing import Tuple, List, Optional

os.environ['WANDB_DISABLED'] = 'true'

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from evaluate import load as load_metric
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchaudio.sox_effects import apply_effects_tensor
from datasets import load_dataset
from datasets.features import Audio
import datasets.utils.logging as datasets_logging
import transformers.utils.logging as transformers_logging
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import MBart50Tokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, MBartForCausalLM
from transformers import SpeechEncoderDecoderModel, SpeechEncoderDecoderConfig, GenerationConfig, Wav2Vec2Config


TARGET_SAMPLING_RATE = 16_000
TRAINING_BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 4
wav2vec2_mbart_logger = get_logger(__name__)


class SpeechDataset(Dataset):
    def __init__(self, dataset_name: str, dataset_split: str,
                 wav2vec2_tokenizer: Wav2Vec2CTCTokenizer, mbart_tokenizer: MBart50Tokenizer,
                 with_augmentation: bool = True, max_size: int = None):
        super().__init__()
        self.with_augmentation = with_augmentation
        train_dataset = load_dataset(dataset_name, split=dataset_split)
        wav2vec2_mbart_logger.info(f'The {dataset_split} dataset is loaded from the "{dataset_name}". '
                                   f'The dataset size is {len(train_dataset)}.')
        train_dataset = train_dataset.filter(
            lambda it2: (it2["transcription"] is not None) and (len(it2["transcription"]) > 3) and
                        (len(it2["normalized"]) > 3)
        )
        wav2vec2_mbart_logger.info(f'The {dataset_split} dataset size after filtering is {len(train_dataset)}.')
        if train_dataset.features['audio'].sampling_rate != TARGET_SAMPLING_RATE:
            train_dataset = train_dataset.cast_column(
                'audio',
                Audio(sampling_rate=TARGET_SAMPLING_RATE)
            )
        if max_size is not None:
            if max_size <= 0:
                raise ValueError(f'max_size = {max_size} is wrong! Expected a positive integer!')
            if max_size < len(train_dataset):
                train_dataset = train_dataset.train_test_split(test_size=max_size / float(len(train_dataset)))['test']
                wav2vec2_mbart_logger.info(f'The final dataset size is {len(train_dataset)}.')
        self.samples_ = []
        if self.with_augmentation:
            number_of_paraphrases = 0
            for sample_idx, source_sample in tqdm(enumerate(train_dataset)):
                audio_data = source_sample['audio']
                speech_array = audio_data['array']
                if not isinstance(speech_array, np.ndarray):
                    speech_array = np.array(speech_array, dtype=np.float32)
                sampling_rate = audio_data['sampling_rate']
                assert sampling_rate == TARGET_SAMPLING_RATE, f'{sampling_rate} != {TARGET_SAMPLING_RATE}'
                true_annotation_for_mbart = ' '.join(source_sample['transcription'].strip().split())
                true_annotation_for_wav2vec2 = ' '.join(source_sample['normalized'].strip().split())
                target_labels_for_wav2vec2_ = wav2vec2_tokenizer(
                    true_annotation_for_wav2vec2,
                    padding='longest',
                    return_tensors='pt'
                )
                target_labels_for_wav2vec2 = target_labels_for_wav2vec2_.input_ids.masked_fill(
                    target_labels_for_wav2vec2_.attention_mask.ne(1),
                    -100
                )
                del target_labels_for_wav2vec2_
                if number_of_paraphrases == 0:
                    number_of_paraphrases = self.detect_number_of_paraphrases(list(source_sample.keys()))
                    if number_of_paraphrases == 0:
                        err_msg = f'The sample {sample_idx} is wrong! There are no paraphrases.'
                        raise ValueError(err_msg)
                    wav2vec2_mbart_logger.info(f'There are {number_of_paraphrases} paraphrases.')
                else:
                    current_number_of_paraphrases = self.detect_number_of_paraphrases(list(source_sample.keys()))
                    if current_number_of_paraphrases != number_of_paraphrases:
                        err_msg = f'The sample {sample_idx} is wrong! ' \
                                  f'Its paraphrase number = {current_number_of_paraphrases} does not equal to ' \
                                  f'the total paraphrase number = {number_of_paraphrases}.'
                        raise ValueError(err_msg)
                paraphrases_of_annotation = []
                paraphrase_probabilities = []
                proba_sum = 0.0
                for paraphrase_idx in range(1, number_of_paraphrases + 1):
                    paraphrase_field = f'paraphrase{paraphrase_idx}'
                    score_field = f'score{paraphrase_idx}'
                    new_paraphrase = source_sample[paraphrase_field]
                    new_score = source_sample[score_field]
                    if (new_paraphrase is None) or (new_score is None):
                        break
                    if len(new_paraphrase.strip()) == 0:
                        break
                    if not isinstance(new_score, float):
                        new_score = float(new_score)
                    if (new_score < 0.0) or (new_score > 1.0):
                        err_msg = f'The paraphrase {paraphrase_idx} of the sample {sample_idx} has ' \
                                  f'a wrong probability = {new_score}.'
                        raise ValueError(err_msg)
                    paraphrases_of_annotation.append(new_paraphrase)
                    paraphrase_probabilities.append(new_score)
                    proba_sum += new_score
                if len(paraphrases_of_annotation) == 0:
                    warn_msg = f'The sample {sample_idx} is wrong! There are no paraphrases of the sample annotation.'
                    wav2vec2_mbart_logger.warning(warn_msg)
                    self.samples_.append(
                        {
                            'speech': torch.from_numpy(speech_array).float(),
                            'annotation_for_wav2vec2': target_labels_for_wav2vec2.long()[0],
                            'annotation_for_mbart': self.tokenize(
                                s=true_annotation_for_mbart,
                                tokenizer_=mbart_tokenizer,
                                decoder_start_token_id=mbart_tokenizer.bos_token_id
                            ),
                            'paraphrases_for_mbart': [],
                            'paraphrase_probabilities': []
                        }
                    )
                else:
                    if abs(1.0 - proba_sum) > 1e-4:
                        err_msg = f'The sample {sample_idx} is wrong! ' \
                                  f'Its paraphrase probability sum is not 1.0. It is equal to {proba_sum}.'
                        raise ValueError(err_msg)
                    self.samples_.append(
                        {
                            'speech': torch.from_numpy(speech_array).float(),
                            'annotation_for_wav2vec2': target_labels_for_wav2vec2.long()[0],
                            'annotation_for_mbart': self.tokenize(
                                s=true_annotation_for_mbart,
                                tokenizer_=mbart_tokenizer,
                                decoder_start_token_id=mbart_tokenizer.bos_token_id
                            ),
                            'paraphrases_for_mbart': [
                                self.tokenize(
                                    s=cur,
                                    tokenizer_=mbart_tokenizer,
                                    decoder_start_token_id=mbart_tokenizer.bos_token_id
                                ) for cur in sorted(paraphrases_of_annotation)
                            ],
                            'paraphrase_probabilities': paraphrase_probabilities
                        }
                    )
                del audio_data, speech_array, true_annotation_for_wav2vec2, true_annotation_for_mbart
                del paraphrase_probabilities, paraphrases_of_annotation
                del target_labels_for_wav2vec2
            del train_dataset
        else:
            for source_sample in tqdm(train_dataset):
                audio_data = source_sample['audio']
                speech_array = audio_data['array']
                if not isinstance(speech_array, np.ndarray):
                    speech_array = np.array(speech_array, dtype=np.float32)
                sampling_rate = audio_data['sampling_rate']
                assert sampling_rate == TARGET_SAMPLING_RATE, f'{sampling_rate} != {TARGET_SAMPLING_RATE}'
                true_annotation_for_mbart = ' '.join(source_sample['transcription'].strip().split())
                true_annotation_for_wav2vec2 = ' '.join(source_sample['normalized'].strip().split())
                target_labels_for_wav2vec2_ = wav2vec2_tokenizer(
                    true_annotation_for_wav2vec2,
                    padding='longest',
                    return_tensors='pt'
                )
                target_labels_for_wav2vec2 = target_labels_for_wav2vec2_.input_ids.masked_fill(
                    target_labels_for_wav2vec2_.attention_mask.ne(1),
                    -100
                )
                del target_labels_for_wav2vec2_
                self.samples_.append(
                    {
                        'speech': torch.from_numpy(speech_array).float(),
                        'annotation_for_wav2vec2': target_labels_for_wav2vec2.long()[0],
                        'annotation_for_mbart': self.tokenize(
                            s=true_annotation_for_mbart,
                            tokenizer_=mbart_tokenizer,
                            decoder_start_token_id=mbart_tokenizer.bos_token_id
                        ),
                        'paraphrases_for_mbart': [],
                        'paraphrase_probabilities': []
                    }
                )
                del audio_data, speech_array
                del target_labels_for_wav2vec2
        self.samples_.sort(
            key=lambda it: (
                it['speech'].shape[0],
                it['annotation_for_wav2vec2'].shape[0],
                len(it['paraphrases_for_mbart'])
            )
        )
        gc.collect()
        wav2vec2_mbart_logger.info('The training dataset is prepared.')

    def __len__(self):
        return len(self.samples_)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cur_sample = self.samples_[idx]
        n_paraphrases = len(cur_sample['paraphrases_for_mbart'])
        n_probabilities = len(cur_sample['paraphrase_probabilities'])
        assert n_paraphrases == n_probabilities, f'{n_paraphrases} != {n_probabilities}'
        if self.with_augmentation:
            if random.random() > 0.6667:
                augmentations = self.construct_augmentation_effects()
            else:
                augmentations = []
        else:
            augmentations = []
        normalized_annotation = cur_sample['annotation_for_wav2vec2']
        if n_paraphrases == 0:
            target_annotation = cur_sample['annotation_for_mbart']
        else:
            if random.random() > 0.3:
                target_annotation = cur_sample['annotation_for_mbart']
            else:
                target_annotation = random.choices(
                    population=cur_sample['paraphrases_for_mbart'],
                    weights=cur_sample['paraphrase_probabilities'],
                    k=1
                )[0]
        augmented_speech = self.normalize_sound(self.augment(cur_sample['speech'], augmentations))
        attention_mask = torch.ones(size=augmented_speech.shape, dtype=torch.float32)
        del cur_sample, augmentations
        return augmented_speech, attention_mask, normalized_annotation, target_annotation

    @staticmethod
    def detect_number_of_paraphrases(columns: List[str]) -> int:
        re_for_paraphrase = re.compile(r'^paraphrase\d+$')
        n = len('paraphrase')
        columns_ids = list(map(
            lambda it2: int(it2[n:]),
            filter(
                lambda it1: re_for_paraphrase.search(it1) is not None,
                columns
            )
        ))
        if len(columns_ids) != max(columns_ids):
            err_msg = f'The column list {columns} is wrong! It contains an incorrect information of paraphrases!'
            raise ValueError(err_msg)
        return len(columns_ids)

    @staticmethod
    def normalize_sound(sound: torch.Tensor) -> torch.Tensor:
        assert isinstance(sound, torch.Tensor), f'type(sound): {type(sound)}'
        assert len(sound.shape) == 1, f'sound.shape: {len(sound.shape)} != 1'
        var, mean = torch.var_mean(sound, dim=0, keepdim=False)
        return (sound - mean) / torch.sqrt(var + 1e-7)

    @staticmethod
    def tokenize(s: str, tokenizer_: MBart50Tokenizer, decoder_start_token_id: int) -> torch.Tensor:
        labels_ = tokenizer_(s, padding='longest', return_tensors='pt')
        labels = labels_["input_ids"].masked_fill(
            labels_.attention_mask.ne(1),
            -100
        )
        if (labels[:, 0] == decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        return labels.long()[0]

    @staticmethod
    def augment(src: torch.Tensor, augmentations: List[List[str]]) -> torch.Tensor:
        assert len(src.shape) == 1, f'{len(src.shape)} != 1'
        if len(augmentations) == 0:
            return src
        res, _ = apply_effects_tensor(
            tensor=src[None, :],
            sample_rate=TARGET_SAMPLING_RATE,
            effects=augmentations,
            channels_first=True
        )
        return torch.squeeze(res)

    @staticmethod
    def construct_augmentation_effects() -> List[List[str]]:
        augmentation_effects = []
        if random.random() > 0.7:
            pitch_val = random.randint(1, 8)
            if random.random() > 0.5:
                pitch_val = -pitch_val
            augmentation_effects.append(["pitch", str(pitch_val)])
        if random.random() > 0.7:
            tempo_val = random.uniform(0.8, 1.2)
            if abs(tempo_val - 1.0) < 1e-2:
                augmentation_effects.append(["tempo", "-q", f'{tempo_val}'])
        if random.random() > 0.7:
            augmentation_effects.append(["lowpass", "-1", "300"])
        if random.random() > 0.7:
            augmentation_effects.append(["gain", "-n"])
        speed = random.uniform(0.7, 1.3)
        if abs(speed - 1.0) > 0.1:
            augmentation_effects.append(["speed", f'{speed:.5f}'])
        if random.random() > 0.8:
            augmentation_effects.append(["reverb", "-w"])
            augmentation_effects.append(["channels", "1"])
        return augmentation_effects


class MultitaskSpeechEncoderDecoder(torch.nn.Module):
    def __init__(
            self,
            seq2seq: SpeechEncoderDecoderModel,
            ctc_layer: torch.nn.Linear,
            dropout: torch.nn.Dropout,
    ):
        super(MultitaskSpeechEncoderDecoder, self).__init__()
        self.seq2seq = seq2seq
        self.dropout = dropout
        self.ctc_layer = ctc_layer

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor,
                acoustic_targets: Optional[torch.Tensor] = None,
                seq2seq_targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Seq2SeqLMOutput]:
        encoder_outputs = self.seq2seq.encoder(
            inputs, attention_mask,
            output_hidden_states=True, output_attentions=True,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs[0]
        encoder_hidden_states = self.dropout(encoder_hidden_states)
        encoder_logits = self.ctc_layer(encoder_hidden_states)

        input_lengths = self.seq2seq.encoder._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        acoustic_targets_mask = acoustic_targets >= 0
        acoustic_target_lengths = acoustic_targets_mask.sum(-1)
        acoustic_flattened_targets = acoustic_targets.masked_select(acoustic_targets_mask)
        acoustic_log_probs = torch.nn.functional.log_softmax(
            encoder_logits, dim=-1, dtype=torch.float32
        ).transpose(0, 1)
        with torch.backends.cudnn.flags(enabled=False):
            acoustic_loss = torch.nn.functional.ctc_loss(
                acoustic_log_probs,
                acoustic_flattened_targets,
                input_lengths,
                acoustic_target_lengths,
                blank=self.seq2seq.encoder.config.pad_token_id,
                reduction=self.seq2seq.encoder.config.ctc_loss_reduction,
                zero_infinity=self.seq2seq.encoder.config.ctc_zero_infinity,
            )

        seq2seq_outputs = self.seq2seq(
            encoder_outputs=(
                encoder_outputs.hidden_states[-1],
                encoder_outputs.hidden_states,
                encoder_outputs.attentions
            ),
            labels=seq2seq_targets,
            return_dict=True
        )
        return acoustic_loss, seq2seq_outputs


def pad_collate_fn(old_batch):
    speech, attention_mask, normalized_annotation, target_annotation = zip(*old_batch)
    assert isinstance(speech, list) or isinstance(speech, tuple), f'type(speech) = {type(speech)}'
    assert isinstance(attention_mask, list) or isinstance(attention_mask, tuple), f'type(attention_mask) = {type(attention_mask)}'
    assert isinstance(normalized_annotation, list) or isinstance(normalized_annotation, tuple), f'type(normalized_annotation) = {type(normalized_annotation)}'
    assert isinstance(target_annotation, list) or isinstance(target_annotation, tuple), f'type(target_annotation) = {type(target_annotation)}'
    speech_pad = torch.nn.utils.rnn.pad_sequence(
        speech,
        batch_first=True, padding_value=0
    ).to(torch.float32)
    assert len(speech_pad.shape) == 2, f'len(speech_pad.shape) = {len(speech_pad.shape)}'
    attention_mask_pad = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True, padding_value=0
    ).to(torch.float32)
    assert len(attention_mask_pad.shape) == 2, f'len(attention_mask_pad.shape) = {len(attention_mask_pad.shape)}'
    assert speech_pad.shape == attention_mask_pad.shape, f'{speech_pad.shape} != {attention_mask_pad.shape}'
    normalized_annotation_pad = torch.nn.utils.rnn.pad_sequence(
        normalized_annotation,
        batch_first=True, padding_value=-100
    ).to(torch.long)
    assert len(normalized_annotation_pad.shape) == 2, f'len(normalized_annotation_pad.shape) = {len(normalized_annotation_pad.shape)}'
    target_annotation_pad = torch.nn.utils.rnn.pad_sequence(
        target_annotation,
        batch_first=True, padding_value=-100
    ).to(torch.long)
    assert len(target_annotation_pad.shape) == 2, f'len(target_annotation_pad.shape) = {len(target_annotation_pad.shape)}'
    return speech_pad, attention_mask_pad, normalized_annotation_pad, target_annotation_pad


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) < 2:
        err_msg = 'The directory for fine-tuned model saving is not specified!'
        raise ValueError(err_msg)
    finetuned_dir_name = os.path.normpath(sys.argv[1])
    if (len(finetuned_dir_name) == 0) or (finetuned_dir_name == '.'):
        raise ValueError(f'Directory "{finetuned_dir_name}" is incorrect!')
    parent_dir_name = os.path.dirname(finetuned_dir_name)
    if len(parent_dir_name) > 0:
        if not os.path.isdir(parent_dir_name):
            raise ValueError(f'Directory "{parent_dir_name}" does not exist!')

    if len(sys.argv) < 3:
        err_msg = 'The dataset path is not specified!'
        raise ValueError(err_msg)
    dataset_name = os.path.normpath(sys.argv[2])
    if not os.path.isdir(dataset_name):
        raise ValueError(f'Directory "{dataset_name}" does not exist!')

    if len(sys.argv) < 4:
        err_msg = 'The acoustic model name is not specified!'
        raise ValueError(err_msg)
    acoustic_model_name = os.path.normpath(sys.argv[3])
    if not os.path.isdir(acoustic_model_name):
        if os.path.isdir(os.path.normpath(acoustic_model_name)):
            acoustic_model_name = os.path.normpath(acoustic_model_name)

    if len(sys.argv) < 5:
        err_msg = 'The language model name is not specified!'
        raise ValueError(err_msg)
    language_model_name = os.path.normpath(sys.argv[4])
    if not os.path.isdir(language_model_name):
        if os.path.isdir(os.path.normpath(language_model_name)):
            language_model_name = os.path.normpath(language_model_name)

    if not os.path.isdir(finetuned_dir_name):
        os.mkdir(finetuned_dir_name)
    if not os.path.isdir(finetuned_dir_name):
        os.mkdir(finetuned_dir_name)

    if not torch.cuda.is_available():
        raise ValueError('CUDA is not available!')

    accelerator = Accelerator(project_dir=finetuned_dir_name)
    wav2vec2_mbart_logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_main_process:
        fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
                  '[%(asctime)s]  %(message)s'
        logging.basicConfig(
            format=fmt_str,
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        wav2vec2_mbart_logger.info(accelerator.state, main_process_only=False)
        datasets_logging.set_verbosity_warning()
        transformers_logging.set_verbosity_info()

    acoustic_fe = Wav2Vec2FeatureExtractor.from_pretrained(acoustic_model_name)
    if accelerator.is_main_process:
        wav2vec2_mbart_logger.info(f'{type(acoustic_fe)}')

    acoustic_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(acoustic_model_name)
    wav2vec2_mbart_logger.info(f'{type(acoustic_tokenizer)}')

    seq2seq_tokenizer = MBart50Tokenizer.from_pretrained(language_model_name, use_fast=False,
                                                         tgt_lang="en_XX", src_lang="ru_RU")
    wav2vec2_mbart_logger.info(f'{type(seq2seq_tokenizer)}')

    acoustic_model_processor = Wav2Vec2Processor(acoustic_fe, seq2seq_tokenizer)
    acoustic_model_processor.save_pretrained(finetuned_dir_name)
    wav2vec2_mbart_logger.info(f'The preprocessor is saved into the "{finetuned_dir_name}".')

    train_dataset = SpeechDataset(
        dataset_name=dataset_name, dataset_split='train', with_augmentation=True,
        wav2vec2_tokenizer=acoustic_tokenizer, mbart_tokenizer=seq2seq_tokenizer
    )

    validation_dataset = SpeechDataset(
        dataset_name=dataset_name, dataset_split='validation', with_augmentation=False, max_size=500,
        wav2vec2_tokenizer=acoustic_tokenizer, mbart_tokenizer=seq2seq_tokenizer
    )

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE,
                                                    shuffle=True, collate_fn=pad_collate_fn)
    val_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=VALIDATION_BATCH_SIZE,
                                                  shuffle=False, collate_fn=pad_collate_fn)
    device = accelerator.device
    wav2vec2_mbart_logger.info(f'accelerator.device = {accelerator.device}')

    acoustic_model_config = Wav2Vec2Config.from_pretrained(acoustic_model_name)
    acoustic_model_config.add_adapter = True
    acoustic_model_config.adapter_kernel_size = 5
    acoustic_model_config.adapter_stride = 3
    acoustic_model_config.ctc_zero_infinity = True
    acoustic_model_config.hidden_dropout = 0.1
    acoustic_model_config.attention_dropout = 0.1
    acoustic_model_config.feat_proj_dropout = 0.0
    acoustic_model_config.layerdrop = 0.1
    acoustic_model = Wav2Vec2ForCTC.from_pretrained(
        acoustic_model_name,
        config=acoustic_model_config,
        pad_token_id=acoustic_tokenizer.pad_token_id,
        vocab_size=len(acoustic_tokenizer)
    )
    acoustic_model.config.add_adapter = True
    acoustic_model.config.adapter_kernel_size = 5
    acoustic_model.config.adapter_stride = 3
    wav2vec2_mbart_logger.info(f'The acoustic model is loaded from the "{acoustic_model_name}".')

    language_model = MBartForCausalLM.from_pretrained(language_model_name, attention_dropout=0.1, dropout=0.1)
    if (language_model.config.is_decoder is False) or (language_model.config.add_cross_attention is False):
        language_model.config.is_decoder = True
        language_model.config.add_cross_attention = True

    seq2seq_config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(
        acoustic_model.config, language_model.config,
        encoder_add_adapter=True,
        encoder_attention_dropout=0.1,
        encoder_hidden_dropout=0.1,
        encoder_feat_proj_dropout=0.0,
        encoder_mask_time_prob=0.05,
        encoder_layerdrop=0.1,
        decoder_attention_dropout=0.1,
        decoder_dropout=0.1,
        max_new_tokens=512,
        num_beams=5
    )

    seq2seq_config.tie_word_embeddings = False
    seq2seq_config.decoder_start_token_id = language_model.config.bos_token_id
    seq2seq_config.pad_token_id = language_model.config.pad_token_id
    seq2seq_config.eos_token_id = language_model.config.eos_token_id
    seq2seq_model = SpeechEncoderDecoderModel(
        encoder=acoustic_model.wav2vec2,
        decoder=language_model,
        config=seq2seq_config
    )
    seq2seq_generation_config = GenerationConfig(
        decoder_start_token_id=language_model.config.bos_token_id,
        pad_token_id=language_model.config.pad_token_id,
        eos_token_id=language_model.config.eos_token_id,
        max_new_tokens=seq2seq_config.max_new_tokens,
        num_beams=1, do_sample=False
    )
    seq2seq_generation_config.save_pretrained(finetuned_dir_name)
    seq2seq_config.save_pretrained(finetuned_dir_name)

    seq2seq_model.freeze_feature_encoder()
    wav2vec2_mbart_logger.info('The acoustic and language models are combined into SpeechEncoderDecoderModel.')

    multitask_model = MultitaskSpeechEncoderDecoder(
        seq2seq=seq2seq_model,
        ctc_layer=acoustic_model.lm_head,
        dropout=acoustic_model.dropout
    )
    del acoustic_model, seq2seq_model
    multitask_model.to(device)

    optimizer = torch.optim.RAdam(params=multitask_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=10)

    multitask_model, optimizer, scheduler, train_data_loader, val_data_loader = accelerator.prepare(
        multitask_model,
        optimizer,
        scheduler,
        train_data_loader,
        val_data_loader
    )

    metric = load_metric('wer')
    number_of_epochs = 1000
    steps_per_eval = 10_000 // TRAINING_BATCH_SIZE
    step_counter = 1
    best_wer = None
    last_checkpoint_dir = os.path.join(finetuned_dir_name, 'last_checkpoint')
    if os.path.isdir(last_checkpoint_dir):
        if os.path.isfile(os.path.join(last_checkpoint_dir, 'pytorch_model.bin')):
            accelerator.load_state(last_checkpoint_dir)
            wav2vec2_mbart_logger.info(f'The training process is restored from the checkpoint "{last_checkpoint_dir}".')
    else:
        os.mkdir(last_checkpoint_dir)
    multitask_model.train()
    wav2vec2_mbart_logger.info('Training is started.')
    total_acoustic_loss = 0
    total_seq2seq_loss = 0
    for epoch in range(1, number_of_epochs + 1):
        for batch in train_data_loader:
            optimizer.zero_grad()
            inputs, attention_mask, acoustic_targets, seq2seq_targets = batch
            acoustic_loss, seq2seq_outputs = multitask_model(inputs, attention_mask, acoustic_targets, seq2seq_targets)
            seq2seq_loss = seq2seq_outputs.loss
            total_acoustic_loss += acoustic_loss.detach().float()
            total_seq2seq_loss += seq2seq_loss.detach().float()
            accelerator.backward(acoustic_loss, retain_graph=True)
            accelerator.backward(seq2seq_loss)
            optimizer.step()
            del inputs, attention_mask, acoustic_targets, seq2seq_targets
            step_counter += 1
            if step_counter % steps_per_eval == 0:
                acoustic_loss_value = total_acoustic_loss.item() / steps_per_eval
                seq2seq_loss_value = total_seq2seq_loss.item() / steps_per_eval
                wav2vec2_mbart_logger.info(f'Epoch {epoch}, step {step_counter}: '
                                           f'training acoustic loss is {acoustic_loss_value}, '
                                           f'training seq2seq loss is {seq2seq_loss_value}.')
                total_seq2seq_loss = 0
                total_acoustic_loss = 0
                multitask_model.eval()
                samples_seen = 0
                for step, test_batch in enumerate(val_data_loader):
                    inputs, attention_mask, acoustic_targets, seq2seq_targets = test_batch
                    seq2seq_targets[seq2seq_targets == -100] = seq2seq_config.pad_token_id
                    with torch.no_grad():
                        predicted_ids = multitask_model.seq2seq.generate(
                            input_values=inputs.to(device), attention_mask=attention_mask.to(device),
                            generation_config=seq2seq_generation_config
                        )
                    predictions = seq2seq_tokenizer.batch_decode(predicted_ids.cpu(), skip_special_tokens=True)
                    references = seq2seq_tokenizer.batch_decode(seq2seq_targets.cpu(), skip_special_tokens=True)
                    predictions, references = accelerator.gather_for_metrics((predictions, references))
                    if accelerator.use_distributed:
                        if step == len(val_data_loader) - 1:
                            predictions = predictions[: len(val_data_loader.dataset) - samples_seen]
                            references = references[: len(val_data_loader.dataset) - samples_seen]
                        else:
                            samples_seen += references.shape[0]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )
                    del inputs, attention_mask, acoustic_targets, seq2seq_targets
                    del predictions, references, predicted_ids
                gc.collect()
                validation_wer = metric.compute()
                wav2vec2_mbart_logger.info(f'Epoch {epoch}, step {step_counter}: validation WER is {validation_wer}.')
                if best_wer is None:
                    best_wer = validation_wer
                    unwrapped_model = accelerator.unwrap_model(multitask_model)
                    unwrapped_model.seq2seq.save_pretrained(
                        finetuned_dir_name, is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        accelerator.save(unwrapped_model.ctc_layer.state_dict(),
                                         os.path.join(finetuned_dir_name, 'w2v2_lm_head.bin'))
                    del unwrapped_model
                elif validation_wer < best_wer:
                    best_wer = validation_wer
                    unwrapped_model = accelerator.unwrap_model(multitask_model)
                    unwrapped_model.seq2seq.save_pretrained(
                        finetuned_dir_name, is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        accelerator.save(unwrapped_model.ctc_layer.state_dict(),
                                         os.path.join(finetuned_dir_name, 'w2v2_lm_head.bin'))
                    del unwrapped_model
                accelerator.save_state(last_checkpoint_dir)
                gc.collect()
                multitask_model.train()
        scheduler.step()
    if step_counter % steps_per_eval != 0:
        acoustic_loss_value = total_acoustic_loss.item() / steps_per_eval
        seq2seq_loss_value = total_seq2seq_loss.item() / steps_per_eval
        wav2vec2_mbart_logger.info(f'Epoch {number_of_epochs}, step {step_counter}: '
                                   f'training acoustic loss is {acoustic_loss_value}, '
                                   f'training seq2seq loss is {seq2seq_loss_value}.')
        multitask_model.eval()
        samples_seen = 0
        for step, test_batch in enumerate(val_data_loader):
            inputs, attention_mask, acoustic_targets, seq2seq_targets = test_batch
            seq2seq_targets[seq2seq_targets == -100] = seq2seq_config.pad_token_id
            with torch.no_grad():
                predicted_ids = multitask_model.seq2seq.generate(
                    input_values=inputs.to(device), attention_mask=attention_mask.to(device),
                    generation_config=seq2seq_generation_config
                )
            predictions = seq2seq_tokenizer.batch_decode(predicted_ids.cpu(), skip_special_tokens=True)
            references = seq2seq_tokenizer.batch_decode(seq2seq_targets.cpu(), skip_special_tokens=True)
            predictions, references = accelerator.gather((predictions, references))
            if accelerator.use_distributed:
                if step == len(val_data_loader) - 1:
                    predictions = predictions[: len(val_data_loader.dataset) - samples_seen]
                    references = references[: len(val_data_loader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            del inputs, attention_mask, acoustic_targets, seq2seq_targets
        gc.collect()
        validation_wer = metric.compute()
        wav2vec2_mbart_logger.info(f'Epoch {number_of_epochs}, step {step_counter}: '
                                   f'validation WER is {validation_wer}.')
        if best_wer is None:
            unwrapped_model = accelerator.unwrap_model(multitask_model)
            unwrapped_model.seq2seq.save_pretrained(
                finetuned_dir_name, is_main_process=accelerator.is_main_process,
                save_function=accelerator.save
            )
            if accelerator.is_main_process:
                accelerator.save(unwrapped_model.ctc_layer.state_dict(),
                                 os.path.join(finetuned_dir_name, 'w2v2_lm_head.bin'))
            del unwrapped_model
            gc.collect()
        elif validation_wer < best_wer:
            unwrapped_model = accelerator.unwrap_model(multitask_model)
            unwrapped_model.seq2seq.save_pretrained(
                finetuned_dir_name, is_main_process=accelerator.is_main_process,
                save_function=accelerator.save
            )
            if accelerator.is_main_process:
                accelerator.save(unwrapped_model.ctc_layer.state_dict(),
                                 os.path.join(finetuned_dir_name, 'w2v2_lm_head.bin'))
            del unwrapped_model
            gc.collect()
        accelerator.save_state(last_checkpoint_dir)
    wav2vec2_mbart_logger.info('Training is finished.')


if __name__ == '__main__':
    main()
