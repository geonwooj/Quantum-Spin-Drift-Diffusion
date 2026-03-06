
# ============================================================
# QSDD (A=2.0, gain 없음, LP 유지, 랜덤 u^, 교대 학습)
# ============================================================
import os, math, json, glob, subprocess, tarfile, re
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras import layers, Model, activations
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

print("TF:", tf.__version__)

# -------------------------
# GPU & dtype
# -------------------------
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("float32")
SEED_T_EPS = 777  # 원하는 값으로 고정

# (권장) 가능한 한 결정론적으로
os.environ["TF_DETERMINISTIC_OPS"] = "1"
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# (권장) weight init 등도 동일하게
tf.keras.utils.set_random_seed(SEED_T_EPS)

# -------------------------
# Google Drive (optional)
# -------------------------
from google.colab import drive
drive.mount('/content/drive')

# -------------------------
# Save paths
# -------------------------
RUN_TAG     = "qsdd_A3_reweight"
SAVE_DIR    = f"/content/drive/MyDrive/diffusion_model/{RUN_TAG}"
WEIGHTS_DIR = os.path.join(SAVE_DIR, "weights")
CKPT_DIR    = os.path.join(SAVE_DIR, "tf_ckpt")
PROTO_PATH  = os.path.join(SAVE_DIR, "uhat16_flowers_leaf_diff_512.npz")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

# -------------------------
# Kaggle datasets (optional)
# -------------------------
os.environ['KAGGLE_USERNAME'] = os.environ.get("KAGGLE_USERNAME", "")
os.environ['KAGGLE_KEY']      = os.environ.get("KAGGLE_KEY", "")

def _kaggle_setup():
    try:
        import kaggle  # noqa
    except Exception:
        subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
    kagdir = "/root/.kaggle"; os.makedirs(kagdir, exist_ok=True)
    kj = os.path.join(kagdir, "kaggle.json")
    if not os.path.exists(kj) and os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        with open(kj, "w") as f:
            json.dump({"username": os.environ["KAGGLE_USERNAME"],
                       "key": os.environ["KAGGLE_KEY"]}, f)
        os.chmod(kj, 0o600)
        print("[kaggle] loaded credentials from env vars")
_kaggle_setup()

FLOWERS_DIR = "/content/sample_data/flowers"
CARS_DIR    = "/content/sample_data/cars"   # 여기엔 이제 PlantVillage leaf 도메인(약 8k장)이 들어감
os.makedirs(FLOWERS_DIR, exist_ok=True)
os.makedirs(CARS_DIR,    exist_ok=True)

# Flowers (기존 그대로)
!kaggle datasets download -d nunenuh/pytorch-challange-flower-dataset -p {FLOWERS_DIR} --unzip -q

# PlantVillage (leaf 도메인, emmarex/plantdisease)
!kaggle datasets download -d emmarex/plantdisease -p {CARS_DIR} --unzip -q

PLANT_ROOT = os.path.join(CARS_DIR, "PlantVillage")  # unzip 후 구조: CARS_DIR/PlantVillage/...

# leaf 도메인에 포함시킬 클래스들 (blight + healthy 위주)
PLANT_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
]

# ===== (순서만 조정) find_image_root에서 쓰이므로 먼저 선언 =====
ALLOW = {'.bmp', '.gif', '.jpeg', '.jpg', '.png'}

def find_image_root(base_dir: str):
    candidates = ["dataset/train","train","car_ims","cars_train/cars_train",".","images","image","jpg","dataset/jpg"]
    for rel in candidates:
        p = os.path.join(base_dir, rel)
        if os.path.isdir(p):
            try: files = os.listdir(p)
            except Exception: files = []
            if any(os.path.splitext(f)[1].lower() in ALLOW for f in files):
                print(f"[dataset] using (direct): {p}"); return p
    best_dir, best_cnt = None, 0
    for root, _, files in os.walk(base_dir):
        cnt = sum(1 for f in files if os.path.splitext(f)[1].lower() in ALLOW)
        if cnt > best_cnt: best_dir, best_cnt = root, cnt
    if not best_dir or best_cnt == 0: raise ValueError(f"No images found under {base_dir}")
    print(f"[dataset] using (recursive): {best_dir}  ({best_cnt} images)")
    return best_dir

flowers_root = os.path.join(FLOWERS_DIR, "dataset")
if not os.path.isdir(flowers_root):
    flowers_root = find_image_root(FLOWERS_DIR)

# =========================
# [SLICE 1] 전역 변수/데이터셋 선언부
# =========================

IMG_SIZE   = 128
CHANNELS   = 3

BATCH_DOMAIN = 16
BATCH_SIZE   = BATCH_DOMAIN * 2

AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(root_dir: str, batch_size: int, seed=42, shuffle_buf=8192):
    ds = image_dataset_from_directory(
        root_dir, labels=None, label_mode=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size, shuffle=False,   # ✅ 여기서 shuffle 끔
        interpolation="bilinear", seed=seed
    )
    opt = tf.data.Options()
    opt.experimental_deterministic = True
    ds = ds.with_options(opt)

    ds = ds.map(lambda x: tf.cast(x, tf.float32)/127.5 - 1.0,
                num_parallel_calls=1)           # ✅ AUTOTUNE 금지(완전 고정 목적)
    ds = ds.unbatch()
    ds = ds.shuffle(shuffle_buf, seed=seed, reshuffle_each_iteration=False)  # ✅ 핵심
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(1)                       # ✅ AUTOTUNE 금지

# flowers_root는 기존 로직 그대로 사용
flower_ds16 = make_dataset(flowers_root, batch_size=BATCH_DOMAIN)

# ===== (순서만 조정) LEAF_ROOT/파일 복사 루틴을 car_ds16 생성보다 먼저 =====
LEAF_ROOT    = os.path.join(CARS_DIR, "leaf_8000")
LEAF_ALL_DIR = os.path.join(LEAF_ROOT, "all")
os.makedirs(LEAF_ALL_DIR, exist_ok=True)

TARGET_COUNT = 8000
ALLOW_EXT = {'.bmp', '.gif', '.jpeg', '.jpg', '.png'}

existing = [
    f for f in os.listdir(LEAF_ALL_DIR)
    if os.path.splitext(f)[1].lower() in ALLOW_EXT
]
if len(existing) >= TARGET_COUNT:
    print(f"[leaf] reuse existing leaf domain: {len(existing)} images")
else:
    import random, shutil

    all_imgs = []
    for cls in PLANT_CLASSES:
        cls_dir = os.path.join(PLANT_ROOT, cls)
        if not os.path.isdir(cls_dir):
            print(f"[warn] missing class dir: {cls_dir}")
            continue
        files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in ALLOW_EXT
        ]
        all_imgs.extend(files)

    random.seed(42)
    random.shuffle(all_imgs)
    if len(all_imgs) > TARGET_COUNT:
        all_imgs = all_imgs[:TARGET_COUNT]

    print(f"[leaf] selected {len(all_imgs)} images for leaf domain")
    for i, src in enumerate(all_imgs):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(LEAF_ALL_DIR, f"leaf_{i:05d}{ext}")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

# leaf 도메인 (기존 car_root 변수명 유지)  <<< 이제 LEAF_ROOT가 이미 정의되어 있음
car_root = LEAF_ROOT
car_ds16 = make_dataset(car_root, batch_size=BATCH_DOMAIN, seed=42)

car_ds16 = car_ds16.unbatch().batch(BATCH_DOMAIN, drop_remainder=True).prefetch(AUTOTUNE)
# -------------------------
# Schedules
# -------------------------
def cosine_beta_schedule(K: int, s: float = 0.008):
    steps = K + 1
    t = tf.linspace(0.0, float(K), steps)
    f = tf.math.cos(((t/float(K))+s)/(1+s)*math.pi/2.0)**2
    alphabar = f/f[0]
    betas = 1.0 - (alphabar[1:]/alphabar[:-1])
    return tf.clip_by_value(betas, 1e-4, 1e-2)

def alpha_tables(betas: tf.Tensor):
    alphas    = 1.0 - betas
    alphabars = tf.math.cumprod(alphas, 0)
    sigma_star= tf.sqrt(1.0 - alphabars)
    return alphas, alphabars, sigma_star

def make_tau_cosine(K: int, tau0: float=1e-4):
    u = np.linspace(0, 1, K, endpoint=True)
    C = 0.5*(1 - np.cos(np.pi*u))
    C = (C - C[0]) / (C[-1]-C[0]+1e-12)
    tau = np.diff(np.concatenate([[0.0], C]))

    # ✅ t=0은 0으로 고정
    tau[0] = 0.0
    # ✅ 나머지에서만 최소 tau 적용
    tau[1:] = np.maximum(tau[1:], tau0)

    tau = tau / np.sum(tau)
    Ccum = np.cumsum(tau)
    return tau.astype(np.float32), Ccum.astype(np.float32)

# -------------------------
# UNet
# -------------------------
class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self, groups=32, axis=-1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs); self.groups=groups; self.axis=axis; self.epsilon=epsilon
    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.gamma = self.add_weight(shape=(dim,), initializer="ones", trainable=True)
        self.beta  = self.add_weight(shape=(dim,), initializer="zeros", trainable=True)
    def call(self, inputs):
        N,H,W,C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        G = tf.minimum(self.groups, C)
        x = tf.reshape(inputs, [N,H,W,G, C//G])
        mean, var = tf.nn.moments(x, [1,2,4], keepdims=True)
        x = (x-mean)/tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [N,H,W,C])
        return self.gamma*x + self.beta

class ResidualBlock(layers.Layer):
    def __init__(self, width, name=None):
        super().__init__(name=name); self.width=width
        self.gn=GroupNormalization(groups=32, axis=-1, epsilon=1e-5, name=f"{name}_gn")
        self.conv1=layers.Conv2D(width,3,padding="same",activation=activations.swish,name=f"{name}_conv1")
        self.conv2=layers.Conv2D(width,3,padding="same",name=f"{name}_conv2")
        self.proj =layers.Conv2D(width,1,name=f"{name}_proj")
        self.time_proj=layers.Dense(width,name=f"{name}_timeproj")
    def call(self,x,cond,training=False):
        res=x; x=self.gn(x,training=training); x=self.conv1(x); x=x + self.time_proj(cond)[:,None,None,:]
        x=self.conv2(x)
        if res.shape[-1]!=self.width: res=self.proj(res)
        return layers.Add()([x,res])

class SpatialSelfAttention(layers.Layer):
    def __init__(self, num_heads=4, dropout=0.0, window_size=8, name=None):
        super().__init__(name=name); self.num_heads=num_heads; self.dropout=dropout; self.window_size=window_size
    def build(self, input_shape):
        C=int(input_shape[-1]); key_dim=max(16, C//self.num_heads)
        self.norm=layers.LayerNormalization(epsilon=1e-5,name=f"{self.name}_ln")
        try:
            self.mha=layers.MultiHeadAttention(num_heads=self.num_heads,key_dim=key_dim,dropout=self.dropout,output_shape=C,name=f"{self.name}_mha")
            self.use_proj=False
        except TypeError:
            self.mha=layers.MultiHeadAttention(num_heads=self.num_heads,key_dim=key_dim,dropout=self.dropout,name=f"{self.name}_mha")
            self.proj=layers.Dense(C,name=f"{self.name}_proj"); self.use_proj=True
    def _mha_tokens(self,tokens,training=False):
        x=self.norm(tokens); out=self.mha(x,x,training=training)
        if getattr(self,"use_proj",False): out=self.proj(out)
        return tokens+out
    def call(self,x,training=False):
        B,H,W,C=tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3]; ws=self.window_size
        assert_op1=tf.debugging.assert_equal(H%ws,0); assert_op2=tf.debugging.assert_equal(W%ws,0)
        with tf.control_dependencies([assert_op1,assert_op2]):
            h_tiles=H//ws; w_tiles=W//ws
            x_blocks=tf.reshape(x,[B,h_tiles,ws,w_tiles,ws,C])
            x_blocks=tf.transpose(x_blocks,[0,1,3,2,4,5])
            x_blocks=tf.reshape(x_blocks,[-1,ws*ws,C])
            y_blocks=self._mha_tokens(x_blocks,training=training)
            y_blocks=tf.reshape(y_blocks,[B,h_tiles,w_tiles,ws,ws,C])
            y_blocks=tf.transpose(y_blocks,[0,1,3,2,4,5])
            return tf.reshape(y_blocks,[B,H,W,C])

def sinusoidal_time_embedding(t, dim=128):
    half=dim//2
    freq=tf.exp(tf.linspace(0.0, tf.math.log(10000.0), half)*(-1.0))
    args=tf.cast(tf.expand_dims(tf.cast(t,tf.float32),1),tf.float32)*tf.expand_dims(freq,0)
    emb=tf.concat([tf.sin(args),tf.cos(args)],axis=-1)
    if dim%2==1: emb=tf.pad(emb,[[0,0],[0,1]])
    return emb

class UNetDenoiser(Model):
    def __init__(self,use_attn_bot=True,use_attn_out=False):
        super().__init__(); self.use_attn_bot=use_attn_bot; self.use_attn_out=use_attn_out
        self.time_mlp=tf.keras.Sequential([layers.Dense(256,activation=activations.swish),
                                           layers.Dense(512,activation=activations.swish)])
        self.spin_mlp=tf.keras.Sequential([layers.Dense(128,activation=activations.swish),
                                           layers.Dense(512,activation=activations.swish)])
        self.e1=ResidualBlock(64,"e1"); self.down1=layers.AveragePooling2D(2)
        self.e2=ResidualBlock(128,"e2"); self.down2=layers.AveragePooling2D(2)
        self.e3=ResidualBlock(256,"e3"); self.down3=layers.AveragePooling2D(2)
        self.e4=ResidualBlock(512,"e4"); self.down4=layers.AveragePooling2D(2)
        self.b1=ResidualBlock(512,"b1")
        if self.use_attn_bot: self.attn_bot=SpatialSelfAttention(num_heads=4,window_size=8,name="attn_bot")
        self.up4=layers.UpSampling2D(2,interpolation="bilinear"); self.d4=ResidualBlock(512,"d4")
        self.up3=layers.UpSampling2D(2,interpolation="bilinear"); self.d3=ResidualBlock(256,"d3")
        self.up2=layers.UpSampling2D(2,interpolation="bilinear"); self.d2=ResidualBlock(128,"d2")
        self.up1=layers.UpSampling2D(2,interpolation="bilinear"); self.d1=ResidualBlock(64,"d1")
        self.final=layers.Conv2D(3,1,kernel_initializer="zeros",name="final_conv",dtype="float32")
    def call(self,x_t,t,s_scalar,training=False):
        temb=self.time_mlp(sinusoidal_time_embedding(t,128))
        semb=self.spin_mlp(tf.reshape(s_scalar,[-1,1]))  # s ∈ {+1,-1}
        cond=temb+semb
        e1=self.e1(x_t,cond,training=training); p1=self.down1(e1)
        e2=self.e2(p1,cond,training=training);  p2=self.down2(e2)
        e3=self.e3(p2,cond,training=training);  p3=self.down3(e3)
        e4=self.e4(p3,cond,training=training);  p4=self.down4(e4)
        b =self.b1(p4,cond,training=training)
        if self.use_attn_bot: b=self.attn_bot(b,training=training)
        u4=self.up4(b);  d4=self.d4(tf.concat([u4,e4],axis=-1),cond,training=training)
        u3=self.up3(d4); d3=self.d3(tf.concat([u3,e3],axis=-1),cond,training=training)
        u2=self.up2(d3); d2=self.d2(tf.concat([u2,e2],axis=-1),cond,training=training)
        u1=self.up1(d2); d1=self.d1(tf.concat([u1,e1],axis=-1),cond,training=training)
        return self.final(d1)   # η̂(x_t,t,s)

# -------------------------
# Drift: A=2.0, 랜덤 uhat16 + LP (LP는 현재 안 써도 무방)
# -------------------------
class TwoPhaseExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    step < start_decay: lr = base_lr
    step in [start_decay, total_steps]: lr = base_lr * exp(log(min_lr/base_lr) * tau)
      where tau = (step-start_decay)/(total_steps-start_decay) in [0,1]
    즉, start_decay에서 base_lr, total_steps에서 min_lr를 정확히 만족.
    """
    def __init__(self, base_lr=1e-4, start_decay=30000, total_steps=60000, min_lr=1e-5):
        super().__init__()
        self.base_lr = float(base_lr)
        self.start_decay = int(start_decay)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        base = tf.constant(self.base_lr, tf.float32)
        min_lr = tf.constant(self.min_lr, tf.float32)

        span = tf.maximum(1.0, tf.cast(self.total_steps - self.start_decay, tf.float32))
        tau = tf.clip_by_value((step - self.start_decay) / span, 0.0, 1.0)

        # exp interpolation: base -> min_lr
        log_ratio = tf.math.log(min_lr / base)
        decayed = base * tf.exp(log_ratio * tau)

        return tf.where(step < self.start_decay, base, decayed)



@dataclass
class DriftCfg:
    K:int; A:float=3.0; tau0:float=1e-4; lp_sigma:float=3.0
    freeze_prototypes: bool = True

def _to_unit_hwk(x, eps=1e-8):
    """
    랜덤 u^ 필드의 '픽셀별 L2 norm의 평균'이 target(≈0.6)이 되도록
    전체를 한 번에 스케일링.

    - 픽셀마다 크기는 다르게 유지 (랜덤성 유지)
    - 전체 평균 크기만 0.6 근처로 맞춤
    """
    x = tf.convert_to_tensor(x, tf.float32)

    # (H, W, C)에서 픽셀별 L2 norm: (H, W, 1)
    norms = tf.norm(x, axis=-1, keepdims=True)      # >= 0

    # 전체 픽셀의 평균 L2 norm 스칼라
    mean_norm = tf.reduce_mean(norms)

    target = 0.6  # "평균 0.6 정도에서 논다"는 기준
    scale = target / (mean_norm + eps)

    x_scaled = x * scale
    return tf.stop_gradient(x_scaled)

class DriftA_NoGain:
    def __init__(self, betas: tf.Tensor, cfg: DriftCfg):
        self.cfg=cfg
        self.betas=tf.cast(betas,tf.float32)
        self.alphas, self.alphabars, self.sigma_star = alpha_tables(self.betas)
        self.K=int(cfg.K)

        tau, Ccum = make_tau_cosine(self.K, tau0=cfg.tau0)
        self.tau = tf.constant(tau, tf.float32)          # sum=1
        self.C_table = tf.constant(Ccum, tf.float32)     # C_T=1
        self.A = tf.Variable(float(cfg.A), dtype=tf.float32, trainable=False)

        # γ_t = A * C_t  (노이즈 공간 계수)
        g = float(self.A.numpy()) * Ccum
        self.gamma_table = tf.constant(g.astype(np.float32), tf.float32)

        # 랜덤 uhat16 (고정 방향장)  ### MODIFIED
        self.uhat16 = None
        self._uhat_cache = {}
        self._lp_sigma=float(cfg.lp_sigma); self._lp_kernel=None

        # report r_T
        sigma_T=float(self.sigma_star[-1].numpy()); g_T=float(self.gamma_table[-1].numpy())
        self.r_T = 2.0*g_T/(sigma_T + 1e-12)
        print(f"[info] A={float(self.A.numpy()):.3f}, C_T=1 → γ_T={g_T:.4f}, σ(T)={sigma_T:.4f}, r_T={self.r_T:.3f}")

    # ---- Low-pass (현재 안 써도 되지만 남겨둠) ----
    def _ensure_lp_kernel(self,C:int,ksize:int=11,sigma:float=3.0):
        if self._lp_kernel is not None: return
        ax=np.arange(ksize)-(ksize-1)/2.0
        xx,yy=np.meshgrid(ax,ax)
        g=np.exp(-(xx**2+yy**2)/(2.0*sigma*sigma)); g/=np.sum(g)
        g4=np.zeros((ksize,ksize,C,1),dtype=np.float32)
        for c in range(C): g4[:,:,c,0]=g
        self._lp_kernel=tf.constant(g4,tf.float32)

    def _lowpass(self,x):
        B,H,W,C=x.shape; self._ensure_lp_kernel(int(C),ksize=11,sigma=self._lp_sigma)
        return tf.nn.depthwise_conv2d(x,self._lp_kernel,strides=[1,1,1,1],padding="SAME")

    # ---- 랜덤 uhat16 생성 및 저장/로드 ----
        # ---- 데이터 기반 uhat16 생성 및 저장/로드 ----
    def warmup_and_save_if_needed(self, flower_ds, car_ds, path: str, target_count=512):
        """
        IMAGE_UP  = flower dataset에서 512장
        IMAGE_DOWN= leaf dataset에서 512장

        각자 512장을 (16x16x3)로 downsample해서 평균을 계산:
          - u_up   ∈ R^{16x16x3}  (flowers 512장의 기대값)
          - u_down ∈ R^{16x16x3}  (leaf 512장의 기대값)

        최종 방향장:
          u_hat16 = (u_up - u_down) / ||u_up - u_down||_2 (픽셀별 L2 정규화)
        """
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            self.uhat16 = tf.Variable(data["uhat16"], dtype=tf.float32, trainable=False)
            print(f"[proto] loaded dataset-diff uhat16 ← {path} shape={self.uhat16.shape}")
            return

        print(f"[proto] building uhat16 from dataset images (flowers 512, leaf 512)...")

        def _mean_16x16(ds, n_imgs: int):
            """주어진 tf.data.Dataset에서 n_imgs장을 뽑아 16x16x3 평균을 계산."""
            sum_16 = np.zeros((16, 16, 3), dtype=np.float32)
            count  = 0

            for batch in ds:
                batch_np = batch.numpy()  # [-1,1], shape [B,128,128,3]
                B = batch_np.shape[0]

                for k in range(B):
                    img = batch_np[k:k+1]  # [1,128,128,3]
                    img16 = tf.image.resize(img, (16, 16), method="area").numpy()[0]  # [16,16,3]
                    sum_16 += img16
                    count  += 1
                    if count >= n_imgs:
                        break

                if count >= n_imgs:
                    break

            if count == 0:
                raise ValueError("[proto] no images collected in _mean_16x16")

            if count < n_imgs:
                print(f"[proto][warn] only {count} images collected (requested {n_imgs})")

            return sum_16 / float(count)

        # 1) flowers에서 512장 평균
        u_up16   = _mean_16x16(flower_ds, target_count)  # [16,16,3]
        # 2) leaf(PlantVillage)에서 512장 평균
        u_down16 = _mean_16x16(car_ds,    target_count)  # [16,16,3]

        # 3) 차이 벡터 및 픽셀별 L2 정규화
        diff = u_up16 - u_down16                          # [16,16,3]
        uhat16 = diff.astype(np.float32)           # [16,16,3]
        self.uhat16 = tf.Variable(uhat16, dtype=tf.float32, trainable=False)

        np.savez(
            path,
            uhat16=self.uhat16.numpy(),
            u_up16=u_up16.astype(np.float32),
            u_down16=u_down16.astype(np.float32),
            A=float(self.A.numpy()),
            C_table=self.C_table.numpy(),
            gamma_table=self.gamma_table.numpy()
        )
        print(f"[proto] saved dataset-diff uhat16 → {path}")
    def _uhat_full(self, H:int, W:int):
        key=(int(H), int(W))
        if key in self._uhat_cache:
            return self._uhat_cache[key]
        if self.uhat16 is None:
            raise RuntimeError("uhat16 not initialized; call warmup_and_save_if_needed first.")
        u = tf.image.resize(self.uhat16[None,...], (H, W), method="bilinear")[0]
        u = _to_unit_hwk(u)  # 업샘플 후 다시 정규화
        self._uhat_cache[key]=u
        return u

    def direction(self, x: tf.Tensor):
      B = tf.shape(x)[0]
      H = tf.shape(x)[1]
      W = tf.shape(x)[2]

      # 해상도 고정(예: 128)이라면 IMG_SIZE 써도 되고,
      # 일반화하려면 H, W 그대로 넘겨도 됨.
      u = self._uhat_full(int(H), int(W))   # [H, W, 3], 평균 norm ≈ 0.6
      u_batch = tf.tile(u[None, ...], [B, 1, 1, 1])
      return tf.stop_gradient(u_batch)


    # r_t = s * γ_t * û  (노이즈 공간)
    def c_t_batch(self, x:tf.Tensor, t_vec:tf.Tensor, s_vec:tf.Tensor):
        uhat=self.direction(x)
        g=tf.gather(self.gamma_table, tf.cast(t_vec,tf.int32))
        #coeff = tf.reshape(s_vec * g, [-1,1,1,1])  # sign 제거
        coeff=tf.reshape(tf.sign(s_vec)*g, [-1,1,1,1])
        return coeff*uhat

# -------------------------
# Build / sample / viz
# -------------------------
def build_model():
    m=UNetDenoiser(use_attn_bot=True,use_attn_out=False)
    _=m(tf.zeros([1,IMG_SIZE,IMG_SIZE,3]), tf.zeros([1],tf.int32), tf.zeros([1],tf.float32), training=False)
    m.summary(); return m

def ddpm_reverse_mean_eps(y_t, t_int, eps_hat, betas, alphas, alphabars):
    """
    y_t = x_t - sqrt(1 - alphabar_t) * r_t  에 대해
    eps_hat ≈ ε  를 넣었을 때의 DDPM 역평균.
    """
    beta_t     = betas[t_int]
    alpha_t    = alphas[t_int]
    alphabar_t = alphabars[t_int]
    return (y_t - (beta_t / tf.sqrt(1.0 - alphabar_t)) * eps_hat) / tf.sqrt(alpha_t)


def sample_images_tf(model, drift: DriftA_NoGain, tables, n=8, s_scalar=+1.0, shape=(128,128,3)):
    betas, alphas, alphabars = tables
    K = int(betas.shape[0])
    s_b = tf.ones([n], tf.float32) * float(s_scalar)

    # --- (1) 초기화: x_T = √(1-ᾱ_T) * ( z + r_T ) ---
    z    = tf.random.normal([n, *shape], dtype=tf.float32)
    t_T  = tf.fill([n], K-1)
    r_T  = drift.c_t_batch(tf.zeros_like(z), t_T, s_b)   # r_T = s*γ_T*û
    sqrt1m_T = tf.sqrt(1.0 - alphabars[-1])
    x = sqrt1m_T * (z + r_T)                             # x_T
    #x=z+sqrt1m_T*(r_T)
    # --- (2) 역과정 루프 (y-world) ---
    for t_int in reversed(range(K)):
        t_vec = tf.fill([n], t_int)

        # r_t (현재 시점), r_{t-1} (다음 시점) 미리 계산
        r_t = drift.c_t_batch(tf.zeros_like(x), t_vec, s_b)

        alphabar_t = alphabars[t_int]
        sqrt1m_t   = tf.sqrt(1.0 - alphabar_t)

        # (2-1) y_t = x_t - sqrt(1 - alphabar_t) * r_t
        y_t = x - sqrt1m_t * r_t

        # (2-2) 모델 출력: η̂(x_t,t,s) ≈ ε + r_t  (입력은 그대로 x_t)
        eta_hat = model(x, t_vec, s_b, training=False)

        # (2-3) y-world noise: ε̂ = η̂ - r_t
        eps_hat = eta_hat - r_t

        # (2-4) y_{t-1}^{mean} = DDPM 역평균(ε-target)
        y_mean = ddpm_reverse_mean_eps(y_t, t_int, eps_hat,
                                       betas, alphas, alphabars)

        if t_int > 0:
            # VP-DDPM variance (beta_tilde)
            alphabar_tm1 = alphabars[t_int - 1]
            beta_t       = betas[t_int]
            beta_tilde   = beta_t * (1.0 - alphabar_tm1) / (1.0 - alphabar_t)

            # y_{t-1} 샘플링
            z = tf.random.normal(tf.shape(x), dtype=x.dtype)
            y_tm1 = y_mean + tf.sqrt(beta_tilde) * z

            # r_{t-1}, x_{t-1} = y_{t-1} + √(1-ᾱ_{t-1}) * r_{t-1}
            t_prev_vec = tf.fill([n], t_int - 1)
            r_tm1      = drift.c_t_batch(tf.zeros_like(x), t_prev_vec, s_b)
            sqrt1m_tm1 = tf.sqrt(1.0 - alphabar_tm1)

            x = y_tm1 + sqrt1m_tm1 * r_tm1
        else:
            # t=0: ᾱ_0=1 → √(1-ᾱ_0)=0 → x_0 = y_0
            x = y_mean

    return tf.clip_by_value(x, -1.0, 1.0)
def sample_with_snapshots_tf(
    model, drift: DriftA_NoGain, tables,
    n=8, s_scalar=+1.0, shape=(128,128,3),
    snapshot_ts=(999,700,500,300,100),
    seed=777,
    init_mode="your_init",  # "your_init" or "ddpm_init"
    return_stats=True
):
    betas, alphas, alphabars = tables
    K = int(betas.shape[0])
    snapshot_ts = sorted(set([int(t) for t in snapshot_ts if 0 <= t < K]), reverse=True)

    tf.random.set_seed(int(seed))
    s_b = tf.ones([n], tf.float32) * float(s_scalar)

    # ---- init x_T ----
    z = tf.random.normal([n, *shape], dtype=tf.float32)
    t_T = tf.fill([n], K-1)
    r_T = drift.c_t_batch(tf.zeros_like(z), t_T, s_b)
    sigma_T = tf.sqrt(1.0 - alphabars[-1])

    if init_mode == "ddpm_init":
        x = sigma_T * (z + r_T)
    else:
        x = z + sigma_T * r_T

    snaps = {}      # t -> x_t (RAW, unclipped)
    stats = {}      # t -> dict of stats

    def _stat(x):
        # x: [n,H,W,3]
        absx = tf.abs(x)
        return {
            "min": float(tf.reduce_min(x).numpy()),
            "max": float(tf.reduce_max(x).numpy()),
            "mean_abs": float(tf.reduce_mean(absx).numpy()),
            "sat_gt1": float(tf.reduce_mean(tf.cast(absx > 1.0, tf.float32)).numpy()),
            "sat_gt2": float(tf.reduce_mean(tf.cast(absx > 2.0, tf.float32)).numpy()),
        }

    # ---- reverse loop ----
    for t_int in reversed(range(K)):  # 999 -> 0
        t_vec = tf.fill([n], t_int)

        r_t = drift.c_t_batch(tf.zeros_like(x), t_vec, s_b)

        alphabar_t = alphabars[t_int]
        sigma_t    = tf.sqrt(1.0 - alphabar_t)

        y_t = x - sigma_t * r_t

        eta_hat = model(x, t_vec, s_b, training=False)
        eps_hat = eta_hat - r_t

        y_mean = ddpm_reverse_mean_eps(y_t, t_int, eps_hat, betas, alphas, alphabars)

        if t_int > 0:
            alphabar_tm1 = alphabars[t_int - 1]
            beta_t       = betas[t_int]
            beta_tilde   = beta_t * (1.0 - alphabar_tm1) / (1.0 - alphabar_t)

            z2 = tf.random.normal(tf.shape(x), dtype=x.dtype)
            y_tm1 = y_mean + tf.sqrt(beta_tilde) * z2

            t_prev = tf.fill([n], t_int - 1)
            r_tm1  = drift.c_t_batch(tf.zeros_like(x), t_prev, s_b)
            sigma_tm1 = tf.sqrt(1.0 - alphabar_tm1)

            x = y_tm1 + sigma_tm1 * r_tm1
        else:
            x = y_mean

        # ---- snapshot: store RAW (NO CLIP) ----
        if t_int in snapshot_ts:
            snaps[t_int] = tf.identity(x)
            if return_stats:
                stats[t_int] = _stat(x)

    if return_stats:
        return snaps, stats
    return snaps

import numpy as np
import matplotlib.pyplot as plt

def show_grid_autoscale(imgs, cols=4, title="", q=0.01):
    """
    imgs: tf.Tensor [N,H,W,3] (RAW, unclipped)
    q: 0.01이면 1%~99% 구간으로 자동 스케일
    """
    x = imgs.numpy()  # float
    lo = np.quantile(x, q)
    hi = np.quantile(x, 1.0 - q)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    x = (x * 255).astype(np.uint8)

    rows = int(np.ceil(len(x)/cols))
    plt.figure(figsize=(cols*2.5, rows*2.5))
    for i, im in enumerate(x):
        plt.subplot(rows, cols, i+1)
        plt.imshow(im)
        plt.axis("off")
    plt.suptitle(title + f"  (autoscale q={q})")
    plt.tight_layout()
    plt.show()

def show_snapshots_autoscale(snaps_dict, title_prefix="", cols=4, q=0.01):
    for t in sorted(snaps_dict.keys(), reverse=True):
        show_grid_autoscale(snaps_dict[t], cols=cols, title=f"{title_prefix} x_{t}", q=q)


def show_grid(imgs, cols=4, title="samples"):
    imgs=(imgs.numpy()*127.5+127.5).astype(np.uint8); rows=int(np.ceil(len(imgs)/cols))
    plt.figure(figsize=(cols*2.5, rows*2.5))
    for i,im in enumerate(imgs):
        plt.subplot(rows,cols,i+1); plt.imshow(im); plt.axis('off')
    plt.suptitle(title); plt.tight_layout(); plt.show()

# =========================
# [SLICE 2] train 함수만 교체 (A=4, 16+16 혼합 배치 학습)
# =========================
@dataclass
class TrainConfig:
    K:int=1000; lr:float=1e-4; grad_clip:float=3.0
    total_steps:int=12000; save_every:int=1000; resume:bool=True
    use_ema: bool = False

def stratified_t_3bins(B, K=1000,
                       low_max=300, mid_max=700,
                       n_low=10, n_mid=11):
    # n_high는 자동 계산
    n_high = B - n_low - n_mid  # 11 (B=32일 때)
    assert n_high >= 0

    t_low  = tf.random.uniform([n_low],  0,       low_max, dtype=tf.int32)
    t_mid  = tf.random.uniform([n_mid],  low_max, mid_max, dtype=tf.int32)
    t_high = tf.random.uniform([n_high], mid_max, K,       dtype=tf.int32)

    t = tf.concat([t_low, t_mid, t_high], axis=0)
    return tf.random.shuffle(t)

def ema_decay_schedule(step: int) -> float:
    # step: 0부터 시작한다고 가정
    if step < 30000:
        return 0
    else:
      return 0.99



def train_alt(flower_ds16, car_ds16, cfg: TrainConfig):
    betas = cosine_beta_schedule(cfg.K)
    alphas, alphabars, sigma_star = alpha_tables(betas)

    drift = DriftA_NoGain(
        betas, DriftCfg(K=cfg.K, A=3.0, tau0=1e-4, lp_sigma=3.0, freeze_prototypes=False)
    )

    # proto(uhat16) 만들기/로드: train에서 쓰는 ds16을 그대로 넘겨도 OK
    drift.warmup_and_save_if_needed(flower_ds16, car_ds16, PROTO_PATH, target_count=512)

    u = drift._uhat_full(IMG_SIZE, IMG_SIZE)   # [128,128,3]
    norms = tf.norm(u, axis=-1)                # [128,128]
    print("u^ pixel L2 mean:", float(tf.reduce_mean(norms).numpy()))
    print("min, max:", float(tf.reduce_min(norms).numpy()),
                    float(tf.reduce_max(norms).numpy()))

    model = build_model()
    # ✅ t/eps 전용 RNG (결정론적 난수열)
    rng = tf.random.Generator.from_seed(SEED_T_EPS)

    LAMBDA_RW = 1.0  # ✅ lambda=1
    lr_schedule = TwoPhaseExpDecay(
        base_lr=cfg.lr,          # 1e-4
        start_decay=30000,       # 30K
        total_steps=cfg.total_steps,  # 60K
        min_lr=1e-5              # ✅ 60K에서 1e-5
    )
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
    def _as_tensor(v):
      if hasattr(v, "read_value"):
          return v.read_value()
      if hasattr(v, "value"):
          try:
              return v.value()
          except TypeError:
              return v.value
      return tf.convert_to_tensor(v)

    ema_vars = None
    if cfg.use_ema:
        ema_vars = [tf.Variable(_as_tensor(v), trainable=False, dtype=v.dtype)
                    for v in model.trainable_variables]
    #print(f"[ema] enabled | decay=SCHEDULE(0.9→0.99→30K) | n_vars={len(ema_vars)}")
    step_var = tf.Variable(0, dtype=tf.int64, name="global_step")
    ckpt = tf.train.Checkpoint(step=step_var, model=model, optimizer=opt, rng=rng)

    if cfg.resume:
        latest = tf.train.latest_checkpoint(CKPT_DIR)
        if latest:
            ckpt.restore(latest).expect_partial()
            if cfg.use_ema:
                for ev, v in zip(ema_vars, model.trainable_variables):
                    ev.assign(v)   # resume 시 EMA를 현재 모델로 리셋/동기화

            print(f"[ckpt] resumed {latest} (step={int(step_var.numpy())})")

    fl_it  = iter(flower_ds16.repeat())
    car_it = iter(car_ds16.repeat())

    step = int(step_var.numpy())
        # ---- grad/clip 모니터링용 ----
    clip_hits = 0
    gn_max = 0.0
    gn_ema = 0.0    # grad_norm EMA(추세 보기용)
    gn_ema_decay = 0.999

        # ---- 모니터링용(per-var 기준 hit 확인) ----
    pervar_hits = 0
    max_pervar_max = 0.0

    win_per_hits = 0
    win_cnt = 0


    while step < cfg.total_steps:
        # ----- 1) 한 step에 16(꽃, +1) + 16(잎, -1) -----
        x_up = next(fl_it)    # [16,H,W,3]
        x_dn = next(car_it)   # [16,H,W,3]

        x0 = tf.concat([x_up, x_dn], axis=0)   # [32,H,W,3]
        x0 = tf.convert_to_tensor(x0, tf.float32)

        # s: 앞 16개 +1, 뒤 16개 -1  (섞지 않음: 요청사항)
        sZ = tf.concat([
            tf.ones([BATCH_DOMAIN], tf.float32),
            -tf.ones([BATCH_DOMAIN], tf.float32)
        ], axis=0)  # [32]

        B = tf.shape(x0)[0]  # 32

        t = rng.uniform([BATCH_SIZE], minval=0, maxval=cfg.K, dtype=tf.int32)
        eps = rng.normal(tf.shape(x0), dtype=x0.dtype)

        # ----- 2) 전방 (η-타깃) -----
        alphabar_t = tf.gather(alphabars, t)                      # [B]
        sqrt1m     = tf.sqrt(1.0 - alphabar_t)[:, None, None, None]
        r_map      = drift.c_t_batch(x0, t, sZ)                   # r_t = s*γ_t*û
        x_t        = tf.sqrt(alphabar_t)[:, None, None, None] * x0 + sqrt1m * (eps + r_map)

        # ----- 3) 학습 (η-타깃) -----
        with tf.GradientTape() as tape:
            eta_hat = model(x_t, t, sZ, training=True)
            target = eps + r_map
            err = eta_hat - target  # [B,H,W,3]
            # (1) 샘플별 MSE: [B]
            mse_per = tf.reduce_mean(tf.square(err), axis=[1,2,3])
            # ✅ r^2 = E_{H,W} [ sum_c r^2 ]  (픽셀 L2^2 평균, 채널 평균 제거)
            r2_per = tf.reduce_mean(tf.reduce_sum(tf.square(r_map), axis=-1), axis=[1,2])  # [B]
            w = 1.0 / (1.0 + LAMBDA_RW * r2_per)
            w = tf.stop_gradient(w)
            # (권장) 평균 1 정규화: reweight 때문에 실효 LR이 줄어드는 착시 방지
            w = w / (tf.reduce_mean(w) + 1e-12)
            loss = tf.reduce_mean(w * mse_per)
        vars_ = model.trainable_variables
        grads = tape.gradient(loss, vars_)

        '''pairs = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        if(step%1000==0):
          g_norms = [tf.norm(g) for g, _ in pairs]
          idx = int(tf.argmax(tf.stack(g_norms)).numpy())
          gmax, vmax = pairs[idx]
          print("There is Time step:",step)
          print("WORST var name:", vmax.name)
          print("WORST grad norm:", float(tf.norm(gmax).numpy()))
          print("GLOBAL grad norm:", float(tf.linalg.global_norm([g for g,_ in pairs]).numpy()))
          print("non-null grads step:",step,"/", len(pairs), "/", len(vars_))
          norms = [(tf.norm(g).numpy(), v.name) for g, v in pairs]
          norms.sort(reverse=True)
          print("top-10 grad norms:")
          for n, name in norms[:10]:
              print(f"{n:10.4f}  {name}")
          gn = tf.linalg.global_norm([g for g,_ in pairs]).numpy()
          mx = norms[0][0]
          print("max/gn =", mx/(gn+1e-12), " gn=", gn, " max=", mx)'''


                # ---- per-var 기준: "각 텐서 grad norm"의 최대값 ----
        grads_nonnull = [g for g in grads if g is not None]
        if grads_nonnull:
            per_norms = [tf.norm(g) for g in grads_nonnull]
            max_per = tf.reduce_max(tf.stack(per_norms))
        else:
            max_per = tf.constant(0.0, tf.float32)

        max_per_v = float(max_per.numpy())
        max_pervar_max = max(max_pervar_max, max_per_v)

        # per-var clip=cfg.grad_clip였다면 hit?
        hit_per = (cfg.grad_clip and (max_per_v > float(cfg.grad_clip)))
        if hit_per:
            pervar_hits += 1
            win_per_hits += 1
        win_cnt += 1


        # ---- global grad norm (before clipping) ----
        grads_nonnull = [g for g in grads if g is not None]
        gn = tf.linalg.global_norm(grads_nonnull) if len(grads_nonnull) > 0 else tf.constant(0.0, tf.float32)

        # ---- global clip (방향 유지) ----
        if cfg.grad_clip and float(cfg.grad_clip) > 0:
            clipped_nonnull, _ = tf.clip_by_global_norm(grads_nonnull, cfg.grad_clip)
            it = iter(clipped_nonnull)
            grads = [next(it) if g is not None else None for g in grads]

            # clip hit count
            if bool((gn > cfg.grad_clip).numpy()):
              clip_hits+=1
              # 1) worst grad 변수 찾기
              grads_nonnull = [g for g in grads if g is not None]
              vars_nonnull  = [v for (v,g) in zip(model.trainable_variables, grads) if g is not None]
              per_norms = tf.stack([tf.norm(g) for g in grads_nonnull])
              idx = int(tf.argmax(per_norms).numpy())
              worst_name = vars_nonnull[idx].name
              worst_norm = float(per_norms[idx].numpy())

              # 2) t 분포 요약
              t_min = int(tf.reduce_min(t).numpy())
              t_max = int(tf.reduce_max(t).numpy())
              t_mean = float(tf.reduce_mean(tf.cast(t, tf.float32)).numpy())

              # 3) r_map, x_t 스케일 요약
              r_abs = float(tf.reduce_mean(tf.abs(r_map)).numpy())
              xt_abs = float(tf.reduce_mean(tf.abs(x_t)).numpy())

              print(f"[SPIKE] step={step} gn={float(gn.numpy()):.3f} worst={worst_name} |norm|={worst_norm:.3f} "
                    f"t(min/mean/max)={t_min}/{t_mean:.1f}/{t_max} mean|r|={r_abs:.4f} mean|xt|={xt_abs:.4f}")

        opt.apply_gradients(zip(grads, model.trainable_variables))

        # ---- grad norm stats ----
        gnv = float(gn.numpy())
        gn_max = max(gn_max, gnv)
        gn_ema = gn_ema_decay * gn_ema + (1.0 - gn_ema_decay) * gnv

        # ===== EMA 업데이트 =====
        # ===== EMA 업데이트 (스케줄 적용 + 조건부 스킵) =====
        if cfg.use_ema:
            cur_decay = ema_decay_schedule(step)  # step 기반
            d = tf.constant(cur_decay, tf.float32)
            one_minus = tf.constant(1.0 - cur_decay, tf.float32)
            for ev, v in zip(ema_vars, model.trainable_variables):
                ev.assign(d * ev + one_minus * v)


        step += 1
        step_var.assign(step)

        if (step % 100) == 0:
            win_rate_per = win_per_hits / max(1, win_cnt)
            print(
                f"step {step:6d} | loss {float(loss.numpy()):.12f} | r_T {drift.r_T:.3f} | "
                f"gn {gnv:8.3f} | gn_ema {gn_ema:8.3f} | gn_max {gn_max:8.3f} | "
                f"clip_hits {clip_hits} | clip_rate {clip_hits/max(1,step):.4f} | "
                f"max_per {max_per_v:8.3f} | max_per_max {max_pervar_max:8.3f} | "
                f"per_hits {pervar_hits} | win_per_rate {win_rate_per:.4f}"
            )
            win_per_hits = 0
            win_cnt = 0

        if (step % 300) == 0:
            w_min = float(tf.reduce_min(w).numpy())
            w_mean = float(tf.reduce_mean(w).numpy())
            w_max = float(tf.reduce_max(w).numpy())
            r2_mean = float(tf.reduce_mean(r2_per).numpy())
            print(f"| w(min/mean/max)={w_min:.3f}/{w_mean:.3f}/{w_max:.3f} | r2_mean={r2_mean:.4f}", end="")


        # ----- 4) 저장 -----
        if (step % cfg.save_every) == 0 or step == cfg.total_steps:
          # 1) ckpt는 유지 (resume/optimizer 위해)
          ckpt_path = ckpt.save(os.path.join(CKPT_DIR, "ckpt"))

          # 2) raw weights 저장은 생략 (원하면 완전 제거)
          w_path = os.path.join(WEIGHTS_DIR, f"denoise_fn_step{step:07d}.weights.h5")
          model.save_weights(w_path)

          # 3) EMA weights만 저장
          if cfg.use_ema:
              ema_w_path = os.path.join(WEIGHTS_DIR, f"denoise_fn_step{step:07d}_ema.weights.h5")

              # (중복 방지) 이미 있으면 삭제
              if tf.io.gfile.exists(ema_w_path):
                  tf.io.gfile.remove(ema_w_path)

              # 현재 weights 백업 -> EMA로 스왑 -> 저장 -> 복구
              backup = [tf.identity(_as_tensor(v)) for v in model.trainable_variables]
              for v, ev in zip(model.trainable_variables, ema_vars):
                  v.assign(ev)

              model.save_weights(ema_w_path)

              for v, b in zip(model.trainable_variables, backup):
                  v.assign(b)

              print(f"[save] step {step} | ckpt→{ckpt_path} | EMA weights→{ema_w_path}")
          else:
              print(f"[save] step {step} | ckpt→{ckpt_path}")

    print("[done] training finished.")
    return model, drift, (betas, alphas, alphabars)

# ===========================
# Load latest & Sample
# ===========================
def build_and_load_latest(K=1000):
    model = build_model()
    betas = cosine_beta_schedule(K)
    alphas, alphabars, sigma_star = alpha_tables(betas)

    drift = DriftA_NoGain(
        betas, DriftCfg(K=K, A=3.0, tau0=1e-4, lp_sigma=3.0, freeze_prototypes=True)
    )
    drift.warmup_and_save_if_needed(flower_ds16, car_ds16, PROTO_PATH, target_count=512)

    wlist = sorted(glob.glob(os.path.join(WEIGHTS_DIR, "denoise_fn_step0035000.weights.h5")))
    wlist = [p for p in wlist if "_ema" not in os.path.basename(p)]
    assert wlist, "raw weights가 없습니다."

    latest_w = wlist[-1]
    print("[weights] loading:", latest_w)
    model.load_weights(latest_w)

    return model, drift, (betas, alphas, alphabars)

    #load때 쓰기 wlist=sorted(glob.glob(os.path.join(WEIGHTS_DIR,"denoise_fn_step0025000.weights.h5")))


# ===========================
# Run (Train & Sample)
# ===========================
cfg = TrainConfig(K=1000, lr=1e-4, grad_clip=100, total_steps=60000, save_every=5000, resume=True)
model, drift, tables = train_alt(flower_ds16, car_ds16, cfg)
model, drift, (betas, alphas, alphabars) = build_and_load_latest(K=cfg.K)
 #샘플링 (동일 시드로 스핀만 바꿔 비교하려면 seed 고정 후 두 번 호출)
imgs_up = sample_images_tf(model, drift, (betas, alphas, alphabars), n=8, s_scalar=1.0)
imgs_dn = sample_images_tf(model, drift, (betas, alphas, alphabars), n=8, s_scalar=-1.0)
show_grid(imgs_up, title="spin UP samples (A=4,35K, C=always1,fine û)")
show_grid(imgs_dn, title="spin DOWN samples (A=4,35K,C=always1, fine û)")
'''
def load_w(model, path):
    print("[load]", path)
    model.load_weights(path)

def pick_path(weights_dir, step, ema=False):
    tag = "_ema" if ema else ""
    return os.path.join(weights_dir, f"denoise_fn_step{int(step):07d}{tag}.weights.h5")

@tf.function
def forward_eta(model, x_t, t_vec, s_vec):
    return model(x_t, t_vec, s_vec, training=False)

def compare_raw_vs_ema_eta(model, drift, tables, step=55000, n=8, seed=777, t_list=(999,700,300,100)):
    betas, alphas, alphabars = tables
    K = int(betas.shape[0])

    # 고정 입력 만들기: x0, eps, s, t 고정
    tf.random.set_seed(seed)
    x0 = tf.random.normal([n, 128,128,3], dtype=tf.float32)
    eps = tf.random.normal([n, 128,128,3], dtype=tf.float32)
    s_vec = tf.concat([tf.ones([n//2]), -tf.ones([n-n//2])], axis=0)  # 섞어도 되고 고정해도 됨

    raw_path = pick_path(WEIGHTS_DIR, step, ema=False)
    ema_path = pick_path(WEIGHTS_DIR, step, ema=True)

    for t_int in t_list:
        t_vec = tf.fill([n], int(t_int))
        alphabar_t = tf.gather(alphabars, t_vec)
        sqrt1m = tf.sqrt(1.0 - alphabar_t)[:,None,None,None]
        r_map  = drift.c_t_batch(x0, t_vec, s_vec)
        x_t    = tf.sqrt(alphabar_t)[:,None,None,None]*x0 + sqrt1m*(eps + r_map)

        # RAW
        load_w(model, raw_path)
        eta_raw = forward_eta(model, x_t, t_vec, s_vec)

        # EMA
        load_w(model, ema_path)
        eta_ema = forward_eta(model, x_t, t_vec, s_vec)

        # 비교 지표
        d = float(tf.reduce_mean(tf.abs(eta_raw - eta_ema)).numpy())
        a = float(tf.reduce_mean(tf.abs(eta_raw)).numpy())
        b = float(tf.reduce_mean(tf.abs(eta_ema)).numpy())
        print(f"[t={t_int}] mean|eta_raw-eta_ema|={d:.6f} | mean|eta_raw|={a:.6f} | mean|eta_ema|={b:.6f} | rel={d/(a+1e-12):.6f}")
compare_raw_vs_ema_eta(model, drift, (betas, alphas, alphabars), step=55000)
def real_stats(ds, n_batches=50):  # ds: [-1,1] 배치 데이터셋
    abs_means = []
    for i, x in enumerate(ds):
        if i >= n_batches: break
        abs_means.append(tf.reduce_mean(tf.abs(x)).numpy())
    return float(np.mean(abs_means)), float(np.std(abs_means))

flower_mean, flower_std = real_stats(flower_ds16, n_batches=50)
leaf_mean, leaf_std     = real_stats(car_ds16, n_batches=50)

print("REAL flower mean_abs:", flower_mean, "+/-", flower_std)
print("REAL leaf   mean_abs:", leaf_mean, "+/-", leaf_std)
def gen_mean_abs_for_spin(model, drift, tables, s, n=64, seed=777):
    betas, alphas, alphabars = tables
    snaps, stats = sample_with_snapshots_tf(
        model, drift, (betas, alphas, alphabars),
        n=n, s_scalar=float(s),
        snapshot_ts=(0,),
        seed=seed,
        init_mode="your_init",
        return_stats=True
    )
    return stats[0]["mean_abs"], stats[0]["sat_gt1"]

# RAW weights 로드 후
model.load_weights(os.path.join(WEIGHTS_DIR, "denoise_fn_step0060000.weights.h5"))

mabs_p, sat_p = gen_mean_abs_for_spin(model, drift, (betas, alphas, alphabars), s=+1)
mabs_m, sat_m = gen_mean_abs_for_spin(model, drift, (betas, alphas, alphabars), s=-1)

print("[GEN RAW] mean_abs (+1) =", mabs_p, "sat_gt1=", sat_p)
print("[GEN RAW] mean_abs (-1) =", mabs_m, "sat_gt1=", sat_m)
print("[REAL ] flower mean_abs =", flower_mean, "+/-", flower_std)
print("[REAL ] leaf   mean_abs =", leaf_mean,   "+/-", leaf_std)
'''
'''snapshot_ts = (999, 700, 300, 100, 50, 10, 0)
seed = 777

def get_snaps_and_stats(weight_path):
    model.load_weights(weight_path)
    snaps, stats = sample_with_snapshots_tf(
        model, drift, (betas, alphas, alphabars),
        n=8, s_scalar=+1.0,
        snapshot_ts=snapshot_ts,
        seed=seed,
        init_mode="your_init",
        return_stats=True
    )
    return snaps, stats

raw_path = os.path.join(WEIGHTS_DIR, "denoise_fn_step0055000.weights.h5")
ema_path = os.path.join(WEIGHTS_DIR, "denoise_fn_step0055000_ema.weights.h5")

raw_snaps, raw_stats = get_snaps_and_stats(raw_path)
ema_snaps, ema_stats = get_snaps_and_stats(ema_path)

print("\n[t]  mean|x_raw-x_ema|   sat_gt1(raw/ema)   mean_abs(raw/ema)")
for t in snapshot_ts:
    d = float(tf.reduce_mean(tf.abs(raw_snaps[t] - ema_snaps[t])).numpy())
    sr, se = raw_stats[t]["sat_gt1"], ema_stats[t]["sat_gt1"]
    ar, ae = raw_stats[t]["mean_abs"], ema_stats[t]["mean_abs"]
    print(f"{t:4d}  {d:.6f}         {sr:.4f}/{se:.4f}        {ar:.4f}/{ae:.4f}")

'''
'''

#이 코드들은, spin_mlp가 얼마나 깨졌는지 보기 위한 코드이다.
s_plus  = tf.constant([[+1.0]], tf.float32)
s_minus = tf.constant([[-1.0]], tf.float32)
semb_p = model.spin_mlp(s_plus,  training=False)  # [1,512]
semb_m = model.spin_mlp(s_minus, training=False)

diff_l1  = float(tf.reduce_mean(tf.abs(semb_p - semb_m)).numpy())
cos_sim  = float(tf.reduce_sum(semb_p*semb_m).numpy() / (tf.norm(semb_p).numpy()*tf.norm(semb_m).numpy() + 1e-12))

print(f"[ema spin_mlp check] mean|semb(+)-semb(-)| = {diff_l1:.6f}")
print(f"[ema spin_mlp check] cos(semb(+),semb(-))  = {cos_sim:.6f}")

def quick_cond_strength(model, t_list=(999,700,300,100)):
    # model.call 내부와 동일한 흐름으로 temb/semb를 뽑습니다.
    for t in t_list:
        t_vec = tf.constant([t], tf.int32)

        temb = model.time_mlp(sinusoidal_time_embedding(t_vec,128))   # [1,512]
        semb_p = model.spin_mlp(tf.constant([[+1.0]], tf.float32))    # [1,512]
        semb_m = model.spin_mlp(tf.constant([[-1.0]], tf.float32))    # [1,512]

        # 평균 절댓값(스케일 감 잡기)
        ta = float(tf.reduce_mean(tf.abs(temb)).numpy())
        sp = float(tf.reduce_mean(tf.abs(semb_p)).numpy())
        sm = float(tf.reduce_mean(tf.abs(semb_m)).numpy())
        diff = float(tf.reduce_mean(tf.abs(semb_p - semb_m)).numpy())

        # cond 차이 = (temb+semb_p) - (temb+semb_m) = semb_p - semb_m
        ratio = diff / (ta + 1e-12)

        print(f"t={t:4d} | mean|temb|={ta:.6f} | mean|semb(+)|={sp:.6f} | mean|semb(-)|={sm:.6f} "
              f"| mean|Δsemb|={diff:.6f} | Δ/|temb|={ratio:.6f}")

quick_cond_strength(model)
'''



'''
debug_eta_on_fixed_xT_multi_t(
    model,
    betas,
    alphabars,
    PROTO_PATH,
    A_list=(2.0, 2.1),
    t_list=(999, 700, 300, 100),
    n=8,
    s_scalar=+1.0,
    seed=777,
    mode="your_init"
)
def debug_eta_on_fixed_xT_multi_t(
    model, betas, alphabars, proto_path,
    A_list=(2.0, 2.1),
    t_list=(999, 700, 300, 100),
    n=8, s_scalar=+1.0, shape=(128,128,3),
    seed=1234, mode="your_init"
):
    """
    샘플링 없이,
    - 동일 seed z 고정
    - 여러 t에서 eta_hat / r_t / eps_hat 비교
    """

    K = int(betas.shape[0])
    s_b = tf.ones([n], tf.float32) * float(s_scalar)

    # --- seed 고정 z ---
    tf.random.set_seed(int(seed))
    z = tf.random.normal([n, *shape], dtype=tf.float32)

    # A별 drift 준비
    drifts = []
    for A in A_list:
        d = DriftA_NoGain(
            betas,
            DriftCfg(K=K, A=float(A), tau0=1e-4, lp_sigma=3.0, freeze_prototypes=True)
        )
        d.warmup_and_save_if_needed(flower_ds16, car_ds16, proto_path, target_count=512)
        drifts.append(d)

    # 공통 random normal (eta - rnd) 비교용
    tf.random.set_seed(int(seed) + 999)
    rnd = tf.random.normal([n, *shape], dtype=tf.float32)

    def mean_abs(x):
        return float(tf.reduce_mean(tf.abs(x)).numpy())

    print("\n===== DEBUG eta_hat @ multi-t (no sampling) =====")
    print(f"mode={mode} | n={n} | s={s_scalar} | seed={seed}")
    print("------------------------------------------------")

    for t_val in t_list:
        print(f"\n--- t = {t_val} ---")
        t_vec = tf.fill([n], int(t_val))

        sqrt1m_t = tf.sqrt(1.0 - alphabars[t_val])

        for A, d in zip(A_list, drifts):
            r_t = d.c_t_batch(tf.zeros_like(z), t_vec, s_b)

            if mode == "ddpm_init":
                x_t = sqrt1m_t * (z + r_t)
            else:  # your_init
                x_t = z + sqrt1m_t * r_t

            eta = model(x_t, t_vec, s_b, training=False)

            epshat = eta - r_t

            print(
                f"A={A:>4} | "
                f"mean|eta|={mean_abs(eta):.6f} | "
                f"mean|eta-rnd|={mean_abs(eta - rnd):.6f} | "
                f"mean|r_t|={mean_abs(r_t):.6f} | "
                f"mean|epshat|={mean_abs(epshat):.6f}"
            )

    print("\n===============================================\n")
snapshot_ts = (999, 700,600, 500,400, 300, 100)

snaps_up, stats_up = sample_with_snapshots_tf(
    model, drift, (betas, alphas, alphabars),
    n=8, s_scalar=+1.0, snapshot_ts=snapshot_ts,
    seed=777, init_mode="your_init", return_stats=True
)
snaps_dn, stats_dn = sample_with_snapshots_tf(
    model, drift, (betas, alphas, alphabars),
    n=8, s_scalar=-1.0, snapshot_ts=snapshot_ts,
    seed=777, init_mode="your_init", return_stats=True
)

# 시각화(클립 없이 raw를 autoscale로)
show_snapshots_autoscale(snaps_up, title_prefix="spin +1", cols=4, q=0.01)
show_snapshots_autoscale(snaps_dn, title_prefix="spin -1", cols=4, q=0.01)

# 포화율/최소최대 확인(이게 진짜 핵심)
print("\n[stats] spin +1")
for t in sorted(stats_up.keys(), reverse=True):
    s = stats_up[t]
    print(t, s)

print("\n[stats] spin -1")
for t in sorted(stats_dn.keys(), reverse=True):
    s = stats_dn[t]
    print(t, s)





def show_snapshots(snaps_dict, title_prefix="", cols=8):
    # snaps_dict: {t: [N,H,W,3]}  (t bigger -> noisier)
    for t in sorted(snaps_dict.keys(), reverse=True):
        show_grid(snaps_dict[t], cols=cols, title=f"{title_prefix}  x_{t}")










'''



# ---- 실행 예시 ----
'''
snapshot_ts = (999, 700, 500, 300, 100)

# spin +1
snaps_up = sample_with_snapshots_tf(
    model, drift, (betas, alphas, alphabars),
    n=8, s_scalar=+1.0, snapshot_ts=snapshot_ts,
    seed=777, init_mode="your_init"
)
show_snapshots(snaps_up, title_prefix="spin +1", cols=4)

# spin -1 (같은 seed 유지해서 비교)
snaps_dn = sample_with_snapshots_tf(
    model, drift, (betas, alphas, alphabars),
    n=8, s_scalar=-1.0, snapshot_ts=snapshot_ts,
    seed=777, init_mode="your_init"
)
show_snapshots(snaps_dn, title_prefix="spin -1", cols=4)

def sample_with_snapshots_tf(
    model, drift: DriftA_NoGain, tables,
    n=8, s_scalar=+1.0, shape=(128,128,3),
    snapshot_ts=(999,700,500,300,100),
    seed=777,
    init_mode="your_init"  # "your_init" or "ddpm_init"
):
    betas, alphas, alphabars = tables
    K = int(betas.shape[0])
    snapshot_ts = sorted(set([int(t) for t in snapshot_ts if 0 <= t < K]), reverse=True)

    # seed 고정
    tf.random.set_seed(int(seed))

    s_b = tf.ones([n], tf.float32) * float(s_scalar)

    # ---- init x_T ----
    z = tf.random.normal([n, *shape], dtype=tf.float32)
    t_T = tf.fill([n], K-1)
    r_T = drift.c_t_batch(tf.zeros_like(z), t_T, s_b)
    sigma_T = tf.sqrt(1.0 - alphabars[-1])

    if init_mode == "ddpm_init":
        x = sigma_T * (z + r_T)
    else:
        x = z + sigma_T * r_T

    snaps = {}  # t -> x_t

    # ---- reverse loop ----
    for t_int in reversed(range(K)):  # 999 -> 0
        t_vec = tf.fill([n], t_int)

        # r_t
        r_t = drift.c_t_batch(tf.zeros_like(x), t_vec, s_b)

        alphabar_t = alphabars[t_int]
        sigma_t    = tf.sqrt(1.0 - alphabar_t)

        # y_t
        y_t = x - sigma_t * r_t

        # eta_hat ≈ eps + r_t
        eta_hat = model(x, t_vec, s_b, training=False)
        eps_hat = eta_hat - r_t

        # y_mean
        y_mean = ddpm_reverse_mean_eps(y_t, t_int, eps_hat, betas, alphas, alphabars)

        if t_int > 0:
            alphabar_tm1 = alphabars[t_int - 1]
            beta_t       = betas[t_int]
            beta_tilde   = beta_t * (1.0 - alphabar_tm1) / (1.0 - alphabar_t)

            z2 = tf.random.normal(tf.shape(x), dtype=x.dtype)
            y_tm1 = y_mean + tf.sqrt(beta_tilde) * z2

            t_prev = tf.fill([n], t_int - 1)
            r_tm1  = drift.c_t_batch(tf.zeros_like(x), t_prev, s_b)
            sigma_tm1 = tf.sqrt(1.0 - alphabar_tm1)

            x = y_tm1 + sigma_tm1 * r_tm1
        else:
            x = y_mean

        # ---- snapshot: store x_t AFTER update so key matches current x ----
        if t_int in snapshot_ts:
            snaps[t_int] = tf.clip_by_value(x, -1.0, 1.0)

    return snaps  # dict: {t: x_t}'''


