# %matplotlib inline
import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import numpy as np


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/nene.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint("./kazsa/G_10000.pth", net_g, None)

random_emotion_root = "./k_wav"
import random

def tts(txt, emotion):
    """emotion为参考情感音频路径 或random_sample（随机抽取）"""
    stn_tst = get_text(txt, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([0])
        if os.path.exists(f"{emotion}.emo.npy"):
            emo = torch.FloatTensor(np.load(f"{emotion}.emo.npy")).unsqueeze(0)
        elif emotion == "random_sample":
            while True:
                rand_wav = random.sample(os.listdir(random_emotion_root), 1)[0]
                if rand_wav.endswith('wav') and os.path.exists(f"{random_emotion_root}/{rand_wav}.emo.npy"):
                    break
            emo = torch.FloatTensor(np.load(f"{random_emotion_root}/{rand_wav}.emo.npy")).unsqueeze(0)
            print(f"{random_emotion_root}/{rand_wav}")
        elif emotion.endswith("wav"):
            import emotion_extract
            emo = torch.FloatTensor(emotion_extract.extract_wav(emotion))
        else:
            print("emotion参数不正确")

        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1, emo=emo)[0][0,0].data.float().numpy()
    ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

    # 随机选取使用训练数据中某一条数据的情感
# 随机抽取的音频文件路径可以用于使用该情感合成其他句子
txt = "なんでこんなに慣れてんのよ。私の方が先に好きだったのに"
tts(txt, emotion='random_sample')
tts(txt, emotion='random_sample')
tts(txt, emotion='random_sample')
tts(txt, emotion='random_sample')
tts(txt, emotion='random_sample')
tts(txt, emotion='random_sample')
txt = "こんにちは。私わあやちねねです。"
tts(txt, emotion="dataset/nene/nen116_030.wav")