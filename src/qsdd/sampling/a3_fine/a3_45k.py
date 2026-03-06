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

print("TF:", tf.__version__)

# -------------------------
# GPU & dtype
# -------------------------
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("float32")

# -------------------------
# Google Drive (optional)
# -------------------------
from google.colab import drive
drive.mount('/content/drive')

# -------------------------
# Save paths
# -------------------------
RUN_TAG     = "qsdd_A2_randomU_LP_eta_target_UP_flower_DOWN_leaf"
SAVE_DIR    = f"/content/drive/MyDrive/diffusion_model/{RUN_TAG}"
WEIGHTS_DIR = os.path.join(SAVE_DIR, "weights")
CKPT_DIR    = os.path.join(SAVE_DIR, "tf_ckpt")
PROTO_PATH  = os.path.join(SAVE_DIR, "uhat16_random.npz")
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
IMG_SIZE=128
FLOWERS_DIR = "/content/sample_data/flowers"
CARS_DIR    = "/content/sample_data/cars"   # 여기엔 이제 PlantVillage leaf 도메인(약 8k장)이 들어감
os.makedirs(FLOWERS_DIR, exist_ok=True)
os.makedirs(CARS_DIR,    exist_ok=True)

# Flowers (기존 그대로)
#/kaggle datasets download -d nunenuh/pytorch-challange-flower-dataset -p {FLOWERS_DIR} --unzip -q

# PlantVillage (leaf 도메인, emmarex/plantdisease)
#kaggle datasets download -d emmarex/plantdisease -p {CARS_DIR} --unzip -q

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

# leaf 8,000장만 모을 디렉토리 (image_dataset_from_directory용 루트)
'''
LEAF_ROOT    = os.path.join(CARS_DIR, "leaf_8000")
LEAF_ALL_DIR = os.path.join(LEAF_ROOT, "all")
os.makedirs(LEAF_ALL_DIR, exist_ok=True)

TARGET_COUNT = 8000
ALLOW_EXT = {'.bmp', '.gif', '.jpeg', '.jpg', '.png'}

# 이미 만들어진 경우 재사용
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

# -------------------------
# Datasets (배치 16, step별 도메인 교체)
# -------------------------
ALLOW = {'.bmp', '.gif', '.jpeg', '.jpg', '.png'}
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE   = 128
BATCH_SIZE = 32
CHANNELS  = 3

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

def make_dataset(root_dir: str):
    ds = image_dataset_from_directory(
        root_dir, labels=None, label_mode=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True,
        interpolation="bilinear", seed=42
    )
    ds = ds.map(lambda x: tf.cast(x, tf.float32)/127.5 - 1.0, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

flower_ds = make_dataset(flowers_root)

# 여기서 car_root는 사실 leaf 도메인(root=LEAF_ROOT) 이지만 변수명은 car_root/car_ds 유지
car_root = LEAF_ROOT
car_ds   = image_dataset_from_directory(
    car_root, labels=None, label_mode=None,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, shuffle=True,
    interpolation="bilinear", seed=42
).map(lambda x: tf.cast(x, tf.float32)/127.5 - 1.0,
      num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
      '''


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
    C = 0.5*(1 - np.cos(np.pi*u))          # C_0=0, C_T=1 (정규화)
    C = (C - C[0]) / (C[-1]-C[0]+1e-12)
    tau = np.diff(np.concatenate([[0.0], C]))
    tau = np.maximum(tau, tau0)
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
@dataclass
class DriftCfg:
    K:int; A:float=2.5; tau0:float=1e-4; lp_sigma:float=3.0
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


def sample_images_tf(
    model,
    drift: DriftA_NoGain,
    tables,
    n=8,
    s_scalar=+1.0,
    shape=(128,128,3),
    z=None,
    start_idx=0,                 # 배치 offset (노트북의 saved)
    base_seed=(1234, 5678),       # 이것만 바꿔 끼우면 전체 샘플이 바뀜
):
    betas, alphas, alphabars = tables
    K = int(betas.shape[0])

    # --- seed 준비 ---
    base = tf.constant([base_seed[0], base_seed[1]], dtype=tf.int32)
    start_idx_tf = tf.cast(start_idx, tf.int32)

    # --- n_eff 결정 + s_b 생성 ---
    if z is None:
        n_eff = n
    else:
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        n_eff = tf.shape(z)[0]

    s_b = tf.ones([n_eff], tf.float32) * float(s_scalar)

    # --- (1) z0 생성 (stateless) 또는 주입 ---
    if z is None:
        # z0는 (base_seed, start_idx)로만 결정 -> 배치 분할이 같으면 항상 동일
        seed_z = tf.random.experimental.stateless_fold_in(base, start_idx_tf)
        seed_z = tf.random.experimental.stateless_fold_in(seed_z, tf.constant(99991, tf.int32))  # z용 구분자
        z0 = tf.random.stateless_normal([n_eff, *shape], seed=seed_z, dtype=tf.float32)
    else:
        z0 = z

    # --- (2) x_T 초기화 ---
    t_T  = tf.fill([n_eff], K-1)
    r_T  = drift.c_t_batch(tf.zeros_like(z0), t_T, s_b)
    sqrt1m_T = tf.sqrt(1.0 - alphabars[-1])
    x = z0 + sqrt1m_T * r_T

    # --- (3) 역과정 루프 ---
    for t_int in reversed(range(K)):
        t_vec = tf.fill([n_eff], t_int)

        r_t = drift.c_t_batch(tf.zeros_like(x), t_vec, s_b)

        alphabar_t = alphabars[t_int]
        sqrt1m_t   = tf.sqrt(1.0 - alphabar_t)

        y_t = x - sqrt1m_t * r_t

        eta_hat = model(x, t_vec, s_b, training=False)
        eps_hat = eta_hat - r_t

        y_mean = ddpm_reverse_mean_eps(
            y_t, t_int, eps_hat,
            betas, alphas, alphabars
        )

        if t_int > 0:
            alphabar_tm1 = alphabars[t_int - 1]
            beta_t       = betas[t_int]
            beta_tilde   = beta_t * (1.0 - alphabar_tm1) / (1.0 - alphabar_t)

            # --- (핵심) step-noise도 stateless로 고정 ---
            seed_t = tf.random.experimental.stateless_fold_in(base, tf.cast(t_int, tf.int32))
            seed_t = tf.random.experimental.stateless_fold_in(seed_t, start_idx_tf)
            seed_t = tf.random.experimental.stateless_fold_in(seed_t, tf.constant(424242, tf.int32))  # eps용 구분자
            noise = tf.random.stateless_normal(tf.shape(x), seed=seed_t, dtype=x.dtype)

            y_tm1 = y_mean + tf.sqrt(beta_tilde) * noise

            t_prev_vec = tf.fill([n_eff], t_int - 1)
            r_tm1      = drift.c_t_batch(tf.zeros_like(x), t_prev_vec, s_b)
            sqrt1m_tm1 = tf.sqrt(1.0 - alphabar_tm1)

            x = y_tm1 + sqrt1m_tm1 * r_tm1
        else:
            x = y_mean

    return tf.clip_by_value(x, -1.0, 1.0)


def show_grid(imgs, cols=4, title="samples"):
    imgs=(imgs.numpy()*127.5+127.5).astype(np.uint8); rows=int(np.ceil(len(imgs)/cols))
    plt.figure(figsize=(cols*2.5, rows*2.5))
    for i,im in enumerate(imgs):
        plt.subplot(rows,cols,i+1); plt.imshow(im); plt.axis('off')
    plt.suptitle(title); plt.tight_layout(); plt.show()

# -------------------------
# Training (랜덤 uhat16 고정, 도메인 번갈아)
# -------------------------
@dataclass
class TrainConfig:
    K:int=1000; lr:float=1e-4; grad_clip:float=1.0
    total_steps:int=12000; save_every:int=1000; resume:bool=True

def train_alt(flower_ds, car_ds, cfg: TrainConfig):
    betas = cosine_beta_schedule(cfg.K)
    alphas, alphabars, sigma_star = alpha_tables(betas)

    drift = DriftA_NoGain(
        betas, DriftCfg(K=cfg.K, A=3.0, tau0=1e-4, lp_sigma=3.0, freeze_prototypes=False)
    )

    drift.warmup_and_save_if_needed(flower_ds, car_ds, PROTO_PATH, target_count=512)
    u = drift._uhat_full(IMG_SIZE, IMG_SIZE)   # [128,128,3]
    norms = tf.norm(u, axis=-1)               # [128,128]
    print("u^ pixel L2 mean:", float(tf.reduce_mean(norms).numpy()))
    print("min, max:", float(tf.reduce_min(norms).numpy()),
                    float(tf.reduce_max(norms).numpy()))
    model = build_model()
    opt = tf.keras.optimizers.Adam(cfg.lr)
    step_var = tf.Variable(0, dtype=tf.int64, name="global_step")
    ckpt = tf.train.Checkpoint(step=step_var, model=model, optimizer=opt)

    if cfg.resume:
        latest = tf.train.latest_checkpoint(CKPT_DIR)
        if latest:
            ckpt.restore(latest).expect_partial()
            print(f"[ckpt] resumed {latest} (step={int(step_var.numpy())})")

    fl_it = iter(flower_ds.repeat())
    car_it = iter(car_ds.repeat())

    step = int(step_var.numpy())

    while step < cfg.total_steps:
        # ----- 1. 데이터 선택 (교대학습) -----
        if (step % 2) == 0:
            x0 = next(fl_it); s_scalar = +1.0
        else:
            x0 = next(car_it); s_scalar = -1.0

        x0 = tf.convert_to_tensor(x0, tf.float32)
        B = tf.shape(x0)[0]
        t = tf.random.uniform([B], minval=0, maxval=cfg.K, dtype=tf.int32)
        eps = tf.random.normal(tf.shape(x0), dtype=x0.dtype)
        sZ = tf.ones([B], tf.float32) * float(s_scalar)

        # ----- 2. 전방 (η-타깃) -----
        alphabar_t = tf.gather(alphabars, t)                      # [B]
        sqrt1m = tf.sqrt(1.0 - alphabar_t)[:, None, None, None]
        r_map  = drift.c_t_batch(x0, t, sZ)                       # r_t = s*γ_t*û
        x_t = tf.sqrt(alphabar_t)[:, None, None, None] * x0 + sqrt1m * (eps + r_map)

        # ----- 3. 학습 (η-타깃) -----
        with tf.GradientTape() as tape:
            eta_hat = model(x_t, t, training=True)            # η̂(x_t,t,s)
            loss = tf.reduce_mean(tf.square(eta_hat - (eps + r_map)))
        grads = tape.gradient(loss, model.trainable_variables)
        if cfg.grad_clip:
            grads = [tf.clip_by_norm(g, cfg.grad_clip) if g is not None else None for g in grads]
        opt.apply_gradients(zip(grads, model.trainable_variables))

        step += 1
        step_var.assign(step)

        if (step % 100) == 0:
            print(f"step {step:6d} | loss {float(loss.numpy()):.12f} | r_T {drift.r_T:.3f}")

        # ----- 4. (EMA 프로토타입 갱신 제거됨) -----

        # ----- 5. 저장 -----
        if (step % cfg.save_every) == 0 or step == cfg.total_steps:
            ckpt_path = ckpt.save(os.path.join(CKPT_DIR, "ckpt"))
            w_path = os.path.join(WEIGHTS_DIR, f"denoise_fn_step{step:07d}.weights.h5")
            model.save_weights(w_path)
            print(f"[save] step {step} | ckpt→{ckpt_path} | weights→{w_path}")

    print("[done] training finished.")
    return model, drift, (betas, alphas, alphabars)

# ===========================
# Load latest & Sample
# ===========================
def build_and_load_latest(model_dir, K=1000):
    global WEIGHTS_DIR, PROTO_PATH   # 필요하면 유지

    # 1) model_dir 기반으로 경로 설정
    WEIGHTS_DIR = os.path.join(model_dir, "weights")
    PROTO_PATH  = os.path.join(model_dir, "uhat16_flowers_leaf_diff_512.npz")

    # 2) 모델/테이블 구성
    model = build_model()
    betas = cosine_beta_schedule(K)
    alphas, alphabars, sigma_star = alpha_tables(betas)

    # 3) drift 생성 (프로토타입은 파일만 쓰니까 ds는 None)
    drift = DriftA_NoGain(
        betas,
        DriftCfg(K=K, A=3.0, tau0=1e-4, lp_sigma=3.0, freeze_prototypes=True)
    )
    drift.warmup_and_save_if_needed(None, None, PROTO_PATH, target_count=512)

    # 4) 최신 가중치 로드
    wlist = sorted(glob.glob(os.path.join(WEIGHTS_DIR, "denoise_fn_step0045000.weights.h5")))
    assert len(wlist) > 0, "가중치 파일이 없습니다."

    latest_w = wlist[-1]
    print("[weights] loading:", latest_w)
    model.load_weights(latest_w)

    return model, drift, (betas, alphas, alphabars)

