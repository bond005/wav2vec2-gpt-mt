import codecs
import copy
import csv
import os
import random
import sys
from typing import List

from datasets import load_dataset
from datasets.features import Audio
import epitran
import librosa
import nltk
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm


TARGET_SAMPLING_RATE = 16_000


def get_sound_frames_number(sound: np.ndarray) -> int:
    n_sound = sound.shape[0]
    n_frames = n_sound // int(round(0.02 * TARGET_SAMPLING_RATE))
    return n_frames


def save_sound(speech_array: np.ndarray, fs: int, fname: str):
    max_speech_array = float(np.max(np.abs(speech_array)))
    if max_speech_array <= 1.00001:
        normalized_speech_array = ((speech_array / max_speech_array) * 32767.0).astype(dtype=np.int16)
    else:
        normalized_speech_array = speech_array.astype(dtype=np.int16)
    write(filename=fname, rate=fs, data=normalized_speech_array)


def apply_g2p(source_text: str, epi: epitran.Epitran) -> str:
    words = list(filter(
        lambda it2: len(it2) > 0,
        map(
            lambda it1: it1.strip().lower(),
            nltk.wordpunct_tokenize(source_text)
        )
    ))
    if len(words) == 0:
        return '<s> </s>'
    transcription = '<s>'
    for cur_word in words:
        new_trans = epi.trans_list(cur_word)
        if len(new_trans) > 0:
            for cur_phone in new_trans:
                if len(cur_phone.strip()) > 0:
                    transcription += (' ' + cur_phone.strip())
    transcription += ' </s>'
    return transcription


def load_nonspeech_corpus(dir_name: str) -> List[np.ndarray]:
    true_title = ['fname', 'labels']
    loaded_title = []
    sounds = []
    line_idx = 1
    with codecs.open(dir_name + '.csv', mode='r', encoding='utf-8') as fp:
        data_reader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in data_reader:
            if len(row) > 0:
                err_msg = f'The file "{dir_name + ".csv"}": line {line_idx} is wrong!'
                if len(row) != true_title:
                    raise ValueError(err_msg)
                if len(loaded_title) == 0:
                    loaded_title = copy.copy(row)
                    if loaded_title != true_title:
                        raise ValueError(err_msg + f' {loaded_title} != {true_title}')
                else:
                    base_name = row[0]
                    full_name = os.path.join(dir_name, base_name)
                    if not os.path.isfile(full_name):
                        err_msg2 = err_msg + f' The file "{full_name}" does not exist!'
                        raise ValueError(err_msg2)
                    y, sr = librosa.load(full_name, sr=TARGET_SAMPLING_RATE, mono=True)
                    if sr != TARGET_SAMPLING_RATE:
                        err_msg2 = err_msg + f' The sound "{full_name}" has inadmissible sampling rate! ' \
                                             f'{sr} != {TARGET_SAMPLING_RATE}'
                        raise ValueError(err_msg2)
                    if len(y.shape) != 1:
                        err_msg2 = err_msg + f' The sound "{full_name}" has inadmissible channels number! ' \
                                             f'{len(y.shape)} != 1'
                        raise ValueError(err_msg2)
                    if (y.shape[0] >= TARGET_SAMPLING_RATE // 3) and (y.shape[0] <= TARGET_SAMPLING_RATE * 70):
                        sounds.append(y)
    return sounds


def main():
    random.seed(42)

    if len(sys.argv) < 2:
        err_msg = 'The destination corpus directory is not specified!'
        raise ValueError(err_msg)
    if len(sys.argv) == 2:
        nonspeech_src_corpus_name = None
        dst_corpus_name = os.path.normpath(sys.argv[1])
    else:
        nonspeech_src_corpus_name = os.path.normpath(sys.argv[2])
        dst_corpus_name = os.path.normpath(sys.argv[1])
    if not os.path.isdir(dst_corpus_name):
        err_msg = f'The directory {dst_corpus_name} does not exist!'
        raise ValueError(err_msg)
    print(f'New dataset will be saved into the "{dst_corpus_name}".')
    if nonspeech_src_corpus_name is not None:
        if not os.path.isdir(nonspeech_src_corpus_name):
            err_msg = f'The directory {nonspeech_src_corpus_name} does not exist!'
            raise ValueError(err_msg)
        if not os.path.isfile(nonspeech_src_corpus_name + '.csv'):
            err_msg = f'The directory {nonspeech_src_corpus_name + ".csv"} does not exist!'
            raise ValueError(err_msg)
        print(f'The additional dataset "{nonspeech_src_corpus_name}" is used.')

    possible_languages = {
        'ru': epitran.Epitran('rus-Cyrl'),
        'tr': epitran.Epitran('tur-Latn'),
        'en': epitran.Epitran('eng-Latn'),
        'pl': epitran.Epitran('pol-Latn'),
        'uz': epitran.Epitran('uzb-Latn'),
        'uk': epitran.Epitran('ukr-Cyrl'),
        'pt': epitran.Epitran('por-Latn'),
    }

    set_of_validation_texts = set()
    set_of_test_texts = set()
    metadata = []
    counter = 1

    for cur_lang in sorted(list(possible_languages.keys())):
        audio_dataset = load_dataset('mozilla-foundation/common_voice_11_0', cur_lang, split='validation')
        audio_dataset = audio_dataset.remove_columns([
            'accent', 'age', 'client_id', 'down_votes',
            'gender', 'locale', 'segment', 'up_votes'
        ])
        if audio_dataset.features['audio'].sampling_rate != TARGET_SAMPLING_RATE:
            audio_dataset = audio_dataset.cast_column(
                'audio',
                Audio(sampling_rate=TARGET_SAMPLING_RATE)
            )
        for batch in tqdm(audio_dataset):
            sample = batch["audio"]
            assert sample["sampling_rate"] == TARGET_SAMPLING_RATE, f'{sample["sampling_rate"]}'
            speech_array = sample["array"]
            sampling_rate = sample["sampling_rate"]
            assert len(speech_array.shape) == 1, f'{speech_array.shape}'
            if (speech_array.shape[0] >= TARGET_SAMPLING_RATE // 3) and \
                    (speech_array.shape[0] <= TARGET_SAMPLING_RATE * 70):
                annotation = batch["sentence"]
                phonetic_transcription = apply_g2p(annotation, possible_languages[cur_lang])
                if len(phonetic_transcription) < 3:
                    assert phonetic_transcription == '<s> </s>', phonetic_transcription
                    annotation = '<SIL>'
                if (len(phonetic_transcription.split()) * 7) <= get_sound_frames_number(speech_array):
                    data_dir = os.path.join(dst_corpus_name, 'data')
                    if not os.path.isdir(data_dir):
                        os.mkdir(data_dir)
                    validation_dir = os.path.join(dst_corpus_name, 'data', 'validation')
                    if not os.path.isdir(validation_dir):
                        os.mkdir(validation_dir)
                    sound_fname = os.path.join(dst_corpus_name, 'data', 'validation', '{0:>06}.wav'.format(counter))
                    save_sound(speech_array=speech_array, fs=sampling_rate, fname=sound_fname)
                    metadata.append((
                        'data/validation/' + os.path.basename(sound_fname),
                        annotation,
                        phonetic_transcription,
                        cur_lang
                    ))
                    if annotation != '<SIL>':
                        set_of_validation_texts.add(' '.join(annotation.strip().split()))
                    counter += 1
        del audio_dataset

        audio_dataset = load_dataset('mozilla-foundation/common_voice_11_0', cur_lang, split='test')
        audio_dataset = audio_dataset.remove_columns([
            'accent', 'age', 'client_id', 'down_votes',
            'gender', 'locale', 'segment', 'up_votes'
        ])
        if audio_dataset.features['audio'].sampling_rate != TARGET_SAMPLING_RATE:
            audio_dataset = audio_dataset.cast_column(
                'audio',
                Audio(sampling_rate=TARGET_SAMPLING_RATE)
            )
        for batch in tqdm(audio_dataset):
            sample = batch["audio"]
            assert sample["sampling_rate"] == TARGET_SAMPLING_RATE, f'{sample["sampling_rate"]}'
            speech_array = sample["array"]
            sampling_rate = sample["sampling_rate"]
            assert len(speech_array.shape) == 1, f'{speech_array.shape}'
            if (speech_array.shape[0] >= TARGET_SAMPLING_RATE // 3) and \
                    (speech_array.shape[0] <= TARGET_SAMPLING_RATE * 70):
                annotation = batch["sentence"]
                phonetic_transcription = apply_g2p(annotation, possible_languages[cur_lang])
                if len(phonetic_transcription) < 3:
                    assert phonetic_transcription == '<s> </s>', phonetic_transcription
                    annotation = '<SIL>'
                if (len(phonetic_transcription.split()) * 7) <= get_sound_frames_number(speech_array):
                    data_dir = os.path.join(dst_corpus_name, 'data')
                    if not os.path.isdir(data_dir):
                        os.mkdir(data_dir)
                    test_dir = os.path.join(dst_corpus_name, 'data', 'test')
                    if not os.path.isdir(test_dir):
                        os.mkdir(test_dir)
                    sound_fname = os.path.join(dst_corpus_name, 'data', 'test', '{0:>06}.wav'.format(counter))
                    save_sound(speech_array=speech_array, fs=sampling_rate, fname=sound_fname)
                    metadata.append((
                        'data/test/' + os.path.basename(sound_fname),
                        annotation,
                        phonetic_transcription,
                        cur_lang
                    ))
                    if annotation != '<SIL>':
                        set_of_test_texts.add(' '.join(annotation.strip().split()))
                    counter += 1
        del audio_dataset

        audio_dataset = load_dataset('mozilla-foundation/common_voice_11_0', cur_lang, split='train')
        audio_dataset = audio_dataset.remove_columns([
            'accent', 'age', 'client_id', 'down_votes',
            'gender', 'locale', 'segment', 'up_votes'
        ])
        if audio_dataset.features['audio'].sampling_rate != TARGET_SAMPLING_RATE:
            audio_dataset = audio_dataset.cast_column(
                'audio',
                Audio(sampling_rate=TARGET_SAMPLING_RATE)
            )
        for batch in tqdm(audio_dataset):
            sample = batch["audio"]
            assert sample["sampling_rate"] == TARGET_SAMPLING_RATE, f'{sample["sampling_rate"]}'
            speech_array = sample["array"]
            sampling_rate = sample["sampling_rate"]
            assert len(speech_array.shape) == 1, f'{speech_array.shape}'
            if (speech_array.shape[0] >= TARGET_SAMPLING_RATE // 3) and \
                    (speech_array.shape[0] <= TARGET_SAMPLING_RATE * 70):
                annotation = batch["sentence"]
                phonetic_transcription = apply_g2p(annotation, possible_languages[cur_lang])
                if len(phonetic_transcription) < 3:
                    assert phonetic_transcription == '<s> </s>', phonetic_transcription
                    annotation = '<SIL>'
                if (len(phonetic_transcription.split()) * 7) <= get_sound_frames_number(speech_array):
                    data_dir = os.path.join(dst_corpus_name, 'data')
                    if not os.path.isdir(data_dir):
                        os.mkdir(data_dir)
                    train_dir = os.path.join(dst_corpus_name, 'data', 'train')
                    if not os.path.isdir(train_dir):
                        os.mkdir(train_dir)
                    if ((annotation not in set_of_validation_texts) and (annotation not in set_of_test_texts)) or \
                            (annotation == '<SIL>'):
                        sound_fname = os.path.join(dst_corpus_name, 'data', 'train', '{0:>06}.wav'.format(counter))
                        save_sound(speech_array=speech_array, fs=sampling_rate, fname=sound_fname)
                        metadata.append((
                            'data/train/' + os.path.basename(sound_fname),
                            annotation,
                            phonetic_transcription,
                            cur_lang
                        ))
                        counter += 1
        del audio_dataset

    print(f'There are {len(metadata)} speech sounds.')
    if nonspeech_src_corpus_name is not None:
        nonspeech_sounds = list(filter(
            lambda it: 15 <= get_sound_frames_number(it),
            load_nonspeech_corpus(nonspeech_src_corpus_name)
        ))
        print(f'There are {len(nonspeech_sounds)} nonspeech sounds.')
        random.shuffle(nonspeech_sounds)
        if len(nonspeech_sounds) > (len(metadata) // 5):
            nonspeech_sounds = nonspeech_sounds[:(len(metadata) // 5)]
        n = int(round(0.1 * len(nonspeech_sounds)))
        nonspeech_sounds_for_training = nonspeech_sounds[:-(2 * n)]
        nonspeech_sounds_for_validation = nonspeech_sounds[-(2 * n):-n]
        nonspeech_sounds_for_test = nonspeech_sounds[-n:]
        del nonspeech_sounds
        for cur_sound in nonspeech_sounds_for_training:
            sound_fname = os.path.join(dst_corpus_name, 'data', 'train', '{0:>06}.wav'.format(counter))
            save_sound(speech_array=cur_sound, fs=TARGET_SAMPLING_RATE, fname=sound_fname)
            metadata.append((
                'data/train/' + os.path.basename(sound_fname),
                '<SIL>',
                '<s> </s>',
                ''
            ))
            counter += 1
        for cur_sound in nonspeech_sounds_for_validation:
            sound_fname = os.path.join(dst_corpus_name, 'data', 'validation', '{0:>06}.wav'.format(counter))
            save_sound(speech_array=cur_sound, fs=TARGET_SAMPLING_RATE, fname=sound_fname)
            metadata.append((
                'data/validation/' + os.path.basename(sound_fname),
                '<SIL>',
                '<s> </s>',
                ''
            ))
            counter += 1
        for cur_sound in nonspeech_sounds_for_test:
            sound_fname = os.path.join(dst_corpus_name, 'data', 'test', '{0:>06}.wav'.format(counter))
            save_sound(speech_array=cur_sound, fs=TARGET_SAMPLING_RATE, fname=sound_fname)
            metadata.append((
                'data/test/' + os.path.basename(sound_fname),
                '<SIL>',
                '<s> </s>',
                ''
            ))
            counter += 1

    with codecs.open(os.path.join(dst_corpus_name, 'metadata.csv'), mode='w', encoding='utf-8') as fp:
        metadata_writer = csv.writer(fp, delimiter=',', quotechar='"')
        metadata_writer.writerow(['file_name', 'transcription', 'ipa', 'lang'])
        for prepared_fname, text, transcription, lang in metadata:
            metadata_writer.writerow([prepared_fname, text, transcription, lang])


if __name__ == '__main__':
    main()
