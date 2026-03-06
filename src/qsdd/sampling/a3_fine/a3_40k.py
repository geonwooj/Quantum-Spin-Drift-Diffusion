# ============================================================
# QSDD (A=2.0, gain 없음, LP 유지, 랜덤 u^, 교대 학습)
# - UP step: flowers 배치 16, s=+1
# - DOWN step: leaf(PlantVillage) 배치 16, s=-1
# - Drift(η-타깃): r_t = s * γ_t * û,  γ_t = A * C_t
# - 전방: x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t) * ( ε + r_t )
# - 학습: L = || η̂(x_t,t,s) - (ε + r_t) ||^2
# - 역방향(VP-DDPM): x_{t-1} = ( x_t - (β_t/√(1-ᾱ_t)) * η̂ ) / √α_t + σ_t z
# - 초기화: x_T = √(1-ᾱ_T) * ( z + r_T )
# - 변경: u^를 16×16×3(uhat16) 랜덤(-1~1)으로 생성해 저장/로드 → bilinear 업샘플(H×W×3) + 정규화 후 사용
# ============================================================
import os, math, json, glob, subprocess, tarfile, re
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras import layers, Model, activations
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

def cosine_beta_schedule(K: int, s: float = 0.008):
    steps = K + 1
    t = tf.linspace(0.0, float(K), steps)
    f = tf.math.cos(((t/float(K))+s)/(1+s)*math.pi/2.0)**2
    alphabar = f/f[0]
    betas = 1.0 - (alphabar[1:]/alphabar[:-1])
    return tf.clip_by_value(betas, 1e-4, 1e-2)

Beta_table=cosine_beta_schedule()
print(Beta_table[0])

