#!/usr/bin/env python3
"""Convert MDX-Net (ONNX) and MDX23C (PyTorch) TFC-TDF checkpoints to GGUF.

python tools/convert_mdx.py --preset kim_vocal_2 models/   # -> mdx_net_kim_vocal_2-17M-v1.0-F32.gguf
python tools/convert_mdx.py --preset drumsep models/       # -> mdx23c_drumsep-0.1B-v1.0-F32.gguf

Both are U-Nets of TFC-TDF blocks over a complex spectrogram, so one runtime architecture
(general.architecture = "tfc_tdf", src/tfc_tdf.cpp) runs both; tfc_tdf.variant selects the block:

  mdx_net  KUIELab's ConvTDFNet as UVR ships it, exported to ONNX: post-activation
           Conv+BN+ReLU, BN folded into the convolutions by the exporter, TDF BatchNorms after
           each frequency Linear (kept as per-channel scale/shift), multiplicative skips.
  mdx23c   ZFTurbo's TFC_TDF_net v3: pre-activation norm/act/conv, InstanceNorm, 1x1 shortcuts,
           concatenated skips, subband channels, every stem from one pass.

Kim_Vocal_2 has no PyTorch source: the converter walks the ONNX graph node by node, checks
every op, attribute and connection against the ConvTDFNet pattern, folds the remaining
BatchNorms and names the weights structurally (src/tfc_tdf.cpp documents the names). Its STFT
settings and volume compensation come from UVR's MDX model_data.json. The DrumSep checkpoint
keeps its PyTorch tensor names; its settings come from the training YAML.

ConvTranspose weights are stored as [out, kh, kw, in] (ggml: [in, kw, kh, out]) so the runtime
runs the stride == kernel transposed convolution as one matmul. Needs numpy and gguf, plus onnx
for kim_vocal_2 or torch and pyyaml for drumsep. Sources are downloaded to models/src/ and
verified by SHA256; --src reuses a local file.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path
import urllib.request

import numpy as np
from gguf import GGUFWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_meta  # noqa: E402

UVR_REV = "a5f88453bfb2b38b05a965bcf67727243e0cbf19"
MSST_WEBUI_REV = "90b617b15bd0dc0b784f3d361faca1b51173fe44"          # Hugging Face weights mirror
MSST_WEBUI_GIT = "39287d46072d2d1d91d9f1518abb1dd9a930e3d7"          # GitHub configs
BN_EPS = 1e-5

PRESETS = {
    # Kim_Vocal_2 as UVR's MDX-Net list publishes it, from the seanghay mirror the VR presets
    # also use. UVR's model_data.json (key 970b3f94..., the MD5 of the file's last 10 MB):
    # compensate 1.009, dim_f 3072, dim_t 2**8, n_fft 7680, primary stem Vocals. UVR's
    # secondary stem is the mix minus the compensated vocals. No license is stated upstream.
    "kim_vocal_2": dict(
        variant="mdx_net", name="mdx_net_kim_vocal_2", file="Kim_Vocal_2.onnx",
        url="https://huggingface.co/seanghay/uvr_models/resolve/6f4fc0cfb0717c9033ffed12471a53008f145b20/Kim_Vocal_2.onnx",
        sha256="ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b",
        source_url="https://huggingface.co/seanghay/uvr_models", organization="UVR (Kim)",
        n_fft=7680, hop=1024, dim_f=3072, dim_t=256, compensate=1.009,
        sources=["vocals"], complement="instrumental"),
    # aufr33 & jarredou's 6-stem drum separation (MDX23C), as mirrored by MSST-WebUI with its
    # training YAML. The original release (github.com/jarredou/models) is gone and no license
    # was ever stated for the weights.
    "drumsep": dict(
        variant="mdx23c", name="mdx23c_drumsep",
        file="aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt",
        url=f"https://huggingface.co/Sucial/MSST-WebUI/resolve/{MSST_WEBUI_REV}/All_Models/multi_stem_models/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt",
        sha256="d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0",
        config_url=f"https://raw.githubusercontent.com/SUC-DriverOld/MSST-WebUI/{MSST_WEBUI_GIT}/configs_backup/multi_stem_models/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt.yaml",
        config_sha256="1f019da093523d34b95912add62c5ab5467d5640c3361c0cbf7a3dcab2bdd4ab",
        source_url="https://huggingface.co/Sucial/MSST-WebUI", organization="aufr33 & jarredou"),
}


def fetch(url, path, sha256):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 and digest != sha256:
        raise ValueError(f"{path}: SHA256 {digest} does not match the preset")
    return digest


def channel(v):
    """A per-channel vector broadcast over [W, H, C] in ggml."""
    return np.asarray(v, np.float32).reshape(-1, 1, 1)


def convt_layout(w):
    """torch ConvTranspose2d [in, out, kh, kw] -> [out, kh, kw, in] (see the module docstring)."""
    return np.ascontiguousarray(w.transpose(1, 2, 3, 0))


# --- mdx_net: walk the ONNX graph ------------------------------------------------------------

class OnnxWalk:
    """Consumes nodes in order and checks each against the expected op and data flow."""

    def __init__(self, model):
        from onnx import helper, numpy_helper
        g = model.graph
        self.nodes = list(g.node)
        self.i = 0
        self.inits = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in g.initializer}
        self.attr = lambda n: {a.name: helper.get_attribute_value(a) for a in n.attribute}
        self.used = set()
        self.cur = g.input[0].name

    def peek(self):
        return self.nodes[self.i] if self.i < len(self.nodes) else None

    def take(self, op, *, chain=True, **attrs):
        n = self.peek()
        if n is None or n.op_type != op:
            raise ValueError(f"node {self.i}: expected {op}, found {n.op_type if n else 'end of graph'}")
        if chain and n.input[0] != self.cur:
            raise ValueError(f"node {self.i} ({n.name}): input is not the previous output")
        a = self.attr(n)
        for k, v in attrs.items():
            if list(a.get(k, [])) != list(v) if isinstance(v, list) else a.get(k) != v:
                raise ValueError(f"node {self.i} ({n.name}): {k}={a.get(k)}, expected {v}")
        self.i += 1
        self.cur = n.output[0]
        return n

    def w(self, name):
        self.used.add(name)
        return self.inits[name]

    def conv(self, k, stride=1, pad=0):
        n = self.take("Conv", kernel_shape=[k, k], strides=[stride, stride], pads=[pad] * 4,
                      dilations=[1, 1], group=1)
        if len(n.input) != 3:
            raise ValueError(f"{n.name}: expected a folded bias")
        return self.w(n.input[1]), self.w(n.input[2])

    def bn(self):
        n = self.take("BatchNormalization")
        eps = self.attr(n).get("epsilon", 1e-5)
        g, b, m, v = (self.w(x) for x in n.input[1:5])
        scale = g / np.sqrt(v + np.float32(eps))
        return scale, b - m * scale

    def relu(self):
        self.take("Relu")


def walk_mdx_net(model):
    """ConvTDFNet as exported: returns (tensors, config) or raises on any deviation."""
    g = OnnxWalk(model)
    t, cfg = {}, {}

    def tfc_tdf(p):
        layers = 0
        while g.peek().op_type == "Conv":
            w, b = g.conv(3, 1, 1)
            g.relu()
            t[f"{p}.tfc.{layers}.weight"], t[f"{p}.tfc.{layers}.bias"] = w, channel(b)
            layers += 1
        if layers == 0:
            raise ValueError(f"{p}: empty TFC")
        x = g.cur
        for j in (1, 2):
            n = g.take("MatMul")
            t[f"{p}.tdf{j}.weight"] = np.ascontiguousarray(g.w(n.input[1]).T)   # x @ B == Linear(B.T)
            scale, shift = g.bn()
            t[f"{p}.tdf{j}.scale"], t[f"{p}.tdf{j}.shift"] = channel(scale), channel(shift)
            g.relu()
        tdf = g.cur
        n = g.take("Add", chain=False)
        if sorted(n.input) != sorted([x, tdf]):
            raise ValueError(f"{p}: TDF residual does not add the TFC output")
        return layers

    w, b = g.conv(1)
    t["first.weight"], t["first.bias"] = w, channel(b)
    g.relu()
    g.take("Transpose", perm=[0, 1, 3, 2])
    skips, layers = [], set()
    enc = 0
    while True:
        p = f"enc.{enc}"
        layers.add(tfc_tdf(p))
        if g.peek().op_type == "ConvTranspose":
            break
        skips.append(g.cur)
        w, b = g.conv(2, 2, 0)
        g.relu()
        t[f"{p}.down.weight"], t[f"{p}.down.bias"] = w, channel(b)
        enc += 1
    # The last parsed block had no downsampling: it is the bottleneck.
    for k in [k for k in t if k.startswith(f"enc.{enc}.")]:
        t["mid." + k[len(f"enc.{enc}."):]] = t.pop(k)
    n_scales = enc
    for d in range(n_scales):
        p = f"dec.{d}"
        n = g.take("ConvTranspose", kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0],
                   dilations=[1, 1], group=1)
        w, b = g.w(n.input[1]), g.w(n.input[2])
        scale, shift = g.bn()
        g.relu()
        t[f"{p}.up.weight"] = convt_layout(w * scale[None, :, None, None])
        t[f"{p}.up.bias"] = channel(b * scale + shift)
        n = g.take("Mul")
        if skips[-1] not in n.input:
            raise ValueError(f"{p}: skip connection does not multiply the matching encoder output")
        skips.pop()
        layers.add(tfc_tdf(p))
    g.take("Transpose", perm=[0, 1, 3, 2])
    w, b = g.conv(1)
    t["final.weight"], t["final.bias"] = w, channel(b)
    if g.peek() is not None or g.cur != model.graph.output[0].name:
        raise ValueError("unexpected nodes after the final convolution")
    if set(g.inits) - g.used:
        raise ValueError(f"unused ONNX initializers: {sorted(set(g.inits) - g.used)[:8]}")
    if len(layers) != 1:
        raise ValueError("TFC blocks differ in depth")
    c = t["first.weight"].shape[0]
    cfg.update(num_scales=n_scales, num_blocks=layers.pop(), num_channels=c,
               growth=t["enc.0.down.weight"].shape[0] - c,
               bottleneck=t["enc.0.tdf1.weight"].shape[1] // t["enc.0.tdf1.weight"].shape[0],
               in_channels=t["first.weight"].shape[1])
    n_params = sum(int(np.prod(v.shape)) for v in g.inits.values())
    return t, cfg, n_params


def convert_mdx_net(p, src):
    import onnx
    model = onnx.load(str(src))
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if dims[1:] != [4, p["dim_f"], p["dim_t"]]:
        raise ValueError(f"ONNX input {dims} does not match dim_f/dim_t of the preset")
    t, cfg, n_params = walk_mdx_net(model)
    cfg.update(n_fft=p["n_fft"], hop=p["hop"], dim_f=p["dim_f"], dim_t=p["dim_t"], num_subbands=1)
    return t, cfg, n_params


# --- mdx23c: PyTorch checkpoint + YAML -------------------------------------------------------

def convert_mdx23c(p, src, config_path):
    import torch
    import yaml
    cfg_y = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
    a, m, tr, inf = cfg_y["audio"], cfg_y["model"], cfg_y["training"], cfg_y["inference"]
    if m["norm"] != "InstanceNorm" or m["act"] not in ("gelu", "relu") or list(m["scale"]) != [2, 2]:
        raise ValueError("only InstanceNorm, gelu/relu and 2x2 scaling are supported")
    if tr.get("target_instrument") or a["num_channels"] != 2:
        raise ValueError("expected a stereo model that predicts every instrument")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k: v.detach().float().numpy() for k, v in sd.items()}
    t = {}
    for k, v in sd.items():
        if v.ndim == 1:
            t[k] = channel(v)                                    # norm weight/bias
        elif ".upscale.conv.2." in k:
            t[k] = convt_layout(v)
        else:
            t[k] = v
    k = m["num_subbands"]
    cfg = dict(n_fft=a["n_fft"], hop=a["hop_length"], dim_f=a["dim_f"], dim_t=a["dim_t"],
               num_subbands=k, num_scales=m["num_scales"], num_blocks=m["num_blocks_per_scale"],
               num_channels=m["num_channels"], growth=m["growth"], bottleneck=m["bottleneck_factor"],
               in_channels=k * a["num_channels"] * 2, act=m["act"],
               chunk_size=a["chunk_size"], num_overlap=inf["num_overlap"])
    if a["chunk_size"] != a["hop_length"] * (a["dim_t"] - 1) or inf.get("dim_t", a["dim_t"]) != a["dim_t"]:
        raise ValueError("chunk_size must be hop_length * (dim_t - 1)")
    if a["sample_rate"] != 44100:
        raise ValueError("expected 44.1 kHz")
    n_params = sum(int(np.prod(v.shape)) for v in sd.values())
    return t, cfg, n_params, list(tr["instruments"])


def expected_shapes(variant, cfg, n_stems):
    """The v1 tensor contract, mirrored by TFCTDF::validate_weights()."""
    s = {}
    n, l, c0, g, bn = cfg["num_scales"], cfg["num_blocks"], cfg["num_channels"], cfg["growth"], cfg["bottleneck"]
    cin = cfg["in_channels"]
    f0 = cfg["dim_f"] // cfg["num_subbands"]
    if variant == "mdx_net":
        def block(p, c, f):
            for j in range(l):
                s[f"{p}.tfc.{j}.weight"], s[f"{p}.tfc.{j}.bias"] = (c, c, 3, 3), (c, 1, 1)
            s[f"{p}.tdf1.weight"], s[f"{p}.tdf2.weight"] = (f // bn, f), (f, f // bn)
            for j in (1, 2):
                s[f"{p}.tdf{j}.scale"] = s[f"{p}.tdf{j}.shift"] = (c, 1, 1)
        s["first.weight"], s["first.bias"] = (c0, cin, 1, 1), (c0, 1, 1)
        c, f = c0, f0
        for i in range(n):
            block(f"enc.{i}", c, f)
            s[f"enc.{i}.down.weight"], s[f"enc.{i}.down.bias"] = (c + g, c, 2, 2), (c + g, 1, 1)
            c, f = c + g, f // 2
        block("mid", c, f)
        for i in range(n):
            s[f"dec.{i}.up.weight"], s[f"dec.{i}.up.bias"] = (c - g, 2, 2, c), (c - g, 1, 1)
            c, f = c - g, f * 2
            block(f"dec.{i}", c, f)
        s["final.weight"], s["final.bias"] = (cin * n_stems, c, 1, 1), (cin * n_stems, 1, 1)
    else:
        def block(p, cin_b, c, f):
            for j in range(l):
                q = f"{p}.blocks.{j}"
                ci = cin_b if j == 0 else c
                s[f"{q}.tfc1.0.weight"] = s[f"{q}.tfc1.0.bias"] = (ci, 1, 1)
                s[f"{q}.tfc1.2.weight"] = (c, ci, 3, 3)
                for k in ("tdf.0", "tdf.3", "tfc2.0"):
                    s[f"{q}.{k}.weight"] = s[f"{q}.{k}.bias"] = (c, 1, 1)
                s[f"{q}.tdf.2.weight"], s[f"{q}.tdf.5.weight"] = (f // bn, f), (f, f // bn)
                s[f"{q}.tfc2.2.weight"] = (c, c, 3, 3)
                s[f"{q}.shortcut.weight"] = (c, ci, 1, 1)
        s["first_conv.weight"] = (c0, cin, 1, 1)
        c, f = c0, f0
        for i in range(n):
            block(f"encoder_blocks.{i}.tfc_tdf", c, c, f)
            q = f"encoder_blocks.{i}.downscale.conv"
            s[f"{q}.0.weight"] = s[f"{q}.0.bias"] = (c, 1, 1)
            s[f"{q}.2.weight"] = (c + g, c, 2, 2)
            c, f = c + g, f // 2
        block("bottleneck_block", c, c, f)
        for i in range(n):
            q = f"decoder_blocks.{i}.upscale.conv"
            s[f"{q}.0.weight"] = s[f"{q}.0.bias"] = (c, 1, 1)
            s[f"{q}.2.weight"] = (c - g, 2, 2, c)
            c, f = c - g, f * 2
            block(f"decoder_blocks.{i}.tfc_tdf", 2 * c, c, f)
        s["final_conv.0.weight"] = (c, c + cin, 1, 1)
        s["final_conv.2.weight"] = (n_stems * cin, c, 1, 1)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--preset", choices=PRESETS, required=True)
    ap.add_argument("--src", type=Path, help="local copy of the preset's checkpoint (still SHA256-checked)")
    ap.add_argument("--config", type=Path, help="local copy of the MDX23C training YAML")
    ap.add_argument("--license", default="other", help="exact weight license; no license assumed")
    args = ap.parse_args()
    p = PRESETS[args.preset]
    src = args.src or Path("models/src") / p["file"]
    digest = fetch(p["url"], src, p["sha256"])
    complement = None
    if p["variant"] == "mdx_net":
        t, cfg, n_params = convert_mdx_net(p, src)
        sources, complement = p["sources"], p["complement"]
        cfg.update(act="relu", compensate=p["compensate"])
    else:
        config = args.config or src.with_name(src.name + ".yaml")
        fetch(p["config_url"], config, p["config_sha256"])
        t, cfg, n_params, sources = convert_mdx23c(p, src, config)
        cfg.update(compensate=1.0)
    want = expected_shapes(p["variant"], cfg, 1 if complement else len(sources))
    if set(t) != set(want):
        raise ValueError(f"tensor names differ from the contract: {sorted(set(t) ^ set(want))[:8]}")
    for k, shape in want.items():
        if tuple(t[k].shape) != shape or not np.isfinite(t[k]).all():
            raise ValueError(f"invalid tensor {k}: expected {shape}, got {t[k].shape}")

    name = p["name"]
    w = GGUFWriter(None, "tfc_tdf")
    w.add_string("stems.model", name)
    w.add_array("stems.sources", sources)
    if complement:
        w.add_string("stems.complement", complement)
    w.add_uint32("stems.samplerate", 44100)
    w.add_uint32("stems.audio_channels", 2)
    w.add_uint32("tfc_tdf.schema_version", 1)
    w.add_string("tfc_tdf.variant", p["variant"])
    for k in ("n_fft", "hop", "dim_f", "dim_t", "num_subbands", "num_scales", "num_blocks",
              "num_channels", "growth", "bottleneck"):
        w.add_uint32("tfc_tdf." + k, int(cfg[k]))
    w.add_string("tfc_tdf.act", cfg["act"])
    w.add_float32("tfc_tdf.compensate", float(cfg["compensate"]))
    if p["variant"] == "mdx_net":
        # UVR's MDX path: zero the lowest bins of the network input, hann-windowed chunks.
        w.add_string("tfc_tdf.chunking", "uvr_mdx")
        w.add_uint32("tfc_tdf.zero_bins", 3)
    else:
        w.add_string("tfc_tdf.chunking", "msst")
        w.add_uint32("tfc_tdf.num_overlap", int(cfg["num_overlap"]))
    for k, a in t.items():
        w.add_tensor(k, np.ascontiguousarray(a, dtype=np.float32))
    gguf_meta.add_general(w, name, n_params, args.license)
    w.add_source_url(p["source_url"])
    gguf_meta.add_source(w, p["file"], p["organization"], p["source_url"], "sha256:" + digest, [p["file"]])
    label = gguf_meta.size_label(n_params)
    out = args.out / gguf_meta.filename(name, label, "F32") if args.out.is_dir() else args.out
    w.write_header_to_file(out)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{name}: {p['variant']}, {cfg['num_scales']} scales x {cfg['num_blocks']} blocks, "
          f"{cfg['num_channels']}+{cfg['growth']} channels, {n_params/1e6:.1f}M params -> {out}")


if __name__ == "__main__":
    main()
