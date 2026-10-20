"""U-Net tham số hóa cho ablation, giữ nguyên phong cách UNET.py gốc
(padding=1, BatchNorm, ConvTranspose2d, concat skip, interpolate khi lệch shape).

Tham số:
  features     : số kênh từng mức encoder, ví dụ (64,128,256,512) -> depth 4
  bottleneck   : số kênh ở đáy (mặc định 2*features[-1])
  n_convs      : số conv 3x3 trong mỗi khối (gốc = 2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_convs=2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UNetAblate(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 features=(64, 128, 256, 512), bottleneck=None, n_convs=2):
        super().__init__()
        features = list(features)
        bottleneck = bottleneck or features[-1] * 2
        self.pool = nn.MaxPool2d(2, 2)

        self.downs = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(ConvBlock(ch, f, n_convs))
            ch = f

        self.bottleneck = ConvBlock(features[-1], bottleneck, n_convs)

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        ch = bottleneck
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, 2))   # up-conv: ch -> f
            self.dec_blocks.append(ConvBlock(f * 2, f, n_convs))   # concat 2f -> f
            ch = f

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.depth = len(features)
        self.n_convs = n_convs

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = dec(torch.cat((skip, x), dim=1))
        return self.final_conv(x)


VARIANTS = {
    # name           : kwargs
    "baseline":        dict(features=(64, 128, 256, 512), n_convs=2),
    "conv1_per_block": dict(features=(64, 128, 256, 512), n_convs=1),
    "conv3_per_block": dict(features=(64, 128, 256, 512), n_convs=3),
    "depth3":          dict(features=(64, 128, 256), n_convs=2),
    "depth5":          dict(features=(64, 128, 256, 512, 1024), n_convs=2),
    "depth5_capped":   dict(features=(64, 128, 256, 512, 512), bottleneck=1024, n_convs=2),
}


def build(name, **kw):
    return UNetAblate(**VARIANTS[name], **kw)


def receptive_field(depth, n_convs):
    """Receptive field (pixel) của 1 pixel ở bottleneck và ở output cuối.
    Tính analytic: conv3x3 thêm 2*jump, maxpool2 thêm 1*jump rồi nhân đôi jump,
    convT2 chia đôi jump. Baseline ra 200 = 1 + 184 (tổng biên mất của paper 572->388) + 15 (pool)."""
    rf, jump = 1, 1
    for _ in range(depth):
        rf += n_convs * 2 * jump
        rf += jump          # maxpool 2x2
        jump *= 2
    rf += n_convs * 2 * jump  # bottleneck
    rf_bottom = rf
    for _ in range(depth):
        jump //= 2          # convT 2x2 stride 2: mỗi pixel ra phụ thuộc đúng 1 pixel vào -> +0
        rf += n_convs * 2 * jump
    return rf_bottom, rf


def count_macs(model, size=512):
    macs = 0

    def hook(m, inp, out):
        nonlocal macs
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            macs += out.numel() * k
        elif isinstance(m, nn.ConvTranspose2d):
            macs += inp[0].numel() * m.out_channels * m.kernel_size[0] * m.kernel_size[1]

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, size, size))
    for h in hs:
        h.remove()
    return macs


if __name__ == "__main__":
    import time, json
    torch.manual_seed(0)
    rows = []
    for name, kw in VARIANTS.items():
        m = build(name)
        params = sum(p.numel() for p in m.parameters())
        n_conv = sum(isinstance(x, (nn.Conv2d, nn.ConvTranspose2d)) for x in m.modules())
        rf_b, rf_o = receptive_field(m.depth, m.n_convs)
        gmacs = count_macs(m, 512) / 1e9
        x = torch.randn(1, 1, 512, 512)
        m.eval()
        with torch.no_grad():
            m(x)
            t0 = time.perf_counter()
            for _ in range(2):
                m(x)
            lat = (time.perf_counter() - t0) / 2
        rows.append(dict(variant=name, depth=m.depth, n_convs=m.n_convs, conv_layers=n_conv,
                         params_M=round(params / 1e6, 2), GMACs_512=round(gmacs, 1),
                         rf_bottleneck=rf_b, rf_output=rf_o, input_multiple=2 ** m.depth,
                         cpu_latency_512_s=round(lat, 2)))
        print(rows[-1], flush=True)
    json.dump(rows, open("structure.json", "w"), indent=2)
