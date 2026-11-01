"""Dense SonarVoxNet: dynamic voxel feature encoder -> dense 3D conv middle stage ->
2D BEV encoder/decoder -> CenterPoint head with a continuous 6D SO(3) rotation output.

The grid (120 x 100 x 10 at the paper's settings) is small enough that dense 3D convs are
cheap, and the paper's ablation shows dense == sparse on this data.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import CORNER_TEMPLATE, gram_schmidt_6d_torch
from .targets import REG_DIM



# ----------------------------------------------------------------------------- grid coordinates
def grid_polar(g: dict) -> float:
    """deg_per_unit > 0 switches the BEV grid to (range, bearing): u = range [m], v = bearing / deg_per_unit."""
    return float(g.get("polar_deg_per_unit", 0.0) or 0.0)


def world_to_grid_torch(xyz: torch.Tensor, g: dict) -> torch.Tensor:
    k = grid_polar(g)
    if k <= 0:
        return xyz
    r = torch.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    th = torch.rad2deg(torch.atan2(xyz[:, 1], xyz[:, 0])) / k
    return torch.stack([r, th, xyz[:, 2]], 1)


def grid_to_world_torch(u: torch.Tensor, v: torch.Tensor, g: dict):
    k = grid_polar(g)
    if k <= 0:
        return u, v
    th = torch.deg2rad(v * k)
    return u * torch.cos(th), u * torch.sin(th)


# ----------------------------------------------------------------------------- beam visibility
BEAM_DEFAULT = {"bear_min": -45.0, "bear_step": 0.353, "n_bear": 256,
                "elev_min": -20.0, "elev_step": 0.635, "n_elev": 64, "margin": 0.15, "r_max": 20.0}


def beam_index_torch(xyz: torch.Tensor, beam: dict):
    """(N, 3) sonar-frame points -> (bearing idx, elevation idx, range, inside-FOV mask)."""
    r = torch.linalg.norm(xyz, dim=1)
    bear = torch.rad2deg(torch.atan2(xyz[:, 1], xyz[:, 0]))
    elev = torch.rad2deg(torch.atan2(xyz[:, 2], torch.hypot(xyz[:, 0], xyz[:, 1])))
    bi = torch.round((bear - beam["bear_min"]) / beam["bear_step"]).long()
    ei = torch.round((elev - beam["elev_min"]) / beam["elev_step"]).long()
    ok = (bi >= 0) & (bi < beam["n_bear"]) & (ei >= 0) & (ei < beam["n_elev"])
    return bi, ei, r, ok


@torch.no_grad()
def visibility_volume(points: torch.Tensor, batch_size: int, vox, beam: dict, n_feat_time: bool):
    """Ray-cast occupancy from the regular sonar beam grid (one return per beam), by marching
    every beam through the voxel grid. Returns (B, 3, D, H, W) channels: free fraction (beam
    samples that passed through the voxel before their return), shadow fraction (samples behind
    the beam's return), and observed (any beam sample fell in the voxel). Only current-frame
    points (time channel == 0) define the returns when frames are stacked."""
    dev = points.device
    b = points[:, 0].long()
    xyz = points[:, 1:4]
    if n_feat_time and points.shape[1] >= 6:
        cur = points[:, 5] == 0
        b, xyz = b[cur], xyz[cur]
    nb, ne = beam["n_bear"], beam["n_elev"]
    bi, ei, r, ok = beam_index_torch(xyz, beam)
    beam_r = torch.full((batch_size * nb * ne,), beam["r_max"], device=dev)
    lin = (b[ok] * nb + bi[ok]) * ne + ei[ok]
    beam_r.scatter_reduce_(0, lin, r[ok], reduce="amin", include_self=True)          # (B*nb*ne,)
    # beam directions
    bear = torch.deg2rad(beam["bear_min"] + beam["bear_step"] * torch.arange(nb, device=dev, dtype=torch.float32))
    elev = torch.deg2rad(beam["elev_min"] + beam["elev_step"] * torch.arange(ne, device=dev, dtype=torch.float32))
    cb, sb = torch.cos(bear)[:, None], torch.sin(bear)[:, None]
    ce, se = torch.cos(elev)[None, :], torch.sin(elev)[None, :]
    dirs = torch.stack([cb * ce, sb * ce, (se * torch.ones_like(cb))], -1).view(-1, 3)   # (nb*ne, 3)
    step = min(vox.vx, vox.vz)
    t_max = float(beam.get("r_max", 20.0))
    ts = torch.arange(step / 2, t_max, step, device=dev)                               # (T,)
    T = ts.numel()
    D, H, W = vox.D, vox.H, vox.W
    free = torch.zeros(batch_size * D * H * W, device=dev)
    shadow = torch.zeros_like(free)
    count = torch.zeros_like(free)
    m = beam["margin"]
    for bb in range(batch_size):
        br = beam_r[bb * nb * ne:(bb + 1) * nb * ne]                                   # (nb*ne,)
        pts = dirs[:, None, :] * ts[None, :, None]                                     # (nb*ne, T, 3)
        g = world_to_grid_torch(pts.view(-1, 3), vox.grid)
        xi = torch.floor((g[:, 0] - vox.xmin) / vox.vx).long()
        yi = torch.floor((g[:, 1] - vox.ymin) / vox.vy).long()
        zi = torch.floor((g[:, 2] - vox.zmin) / vox.vz).long()
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & (zi >= 0) & (zi < D)
        tt = ts[None, :].expand(nb * ne, T).reshape(-1)
        rr = br[:, None].expand(nb * ne, T).reshape(-1)
        is_free = (tt < rr - m) & inside
        is_shadow = (tt > rr + m) & inside
        idx = ((bb * D + zi) * H + yi) * W + xi
        count.index_add_(0, idx[inside], torch.ones(int(inside.sum()), device=dev))
        free.index_add_(0, idx[is_free], torch.ones(int(is_free.sum()), device=dev))
        shadow.index_add_(0, idx[is_shadow], torch.ones(int(is_shadow.sum()), device=dev))
    denom = count.clamp(min=1.0)
    vis = torch.stack([free / denom, shadow / denom, (count > 0).float()], 0)           # (3, B*D*H*W)
    return vis.view(3, batch_size, D, H, W).transpose(0, 1).contiguous()



# ----------------------------------------------------------------------------- range-view branch
class RangeViewBranch(nn.Module):
    """2D U-Net on the sonar's native beam image (n_elev x n_bear). Input channels per beam:
    range/12, intensity, x/12, y/6, z/3, valid. Output: out_ch features per beam, gathered back to
    each current-frame point by its beam index (previous-frame points get zeros)."""

    def __init__(self, beam: dict, ch: int = 32, out_ch: int = 32):
        super().__init__()
        self.beam = beam
        self.nb, self.ne = beam["n_bear"], beam["n_elev"]
        self.out_ch = out_ch
        self.enc1 = nn.Sequential(conv2d_bn(6, ch), conv2d_bn(ch, ch))
        self.enc2 = nn.Sequential(conv2d_bn(ch, 2 * ch, 3, 2, 1), conv2d_bn(2 * ch, 2 * ch))
        self.enc3 = nn.Sequential(conv2d_bn(2 * ch, 4 * ch, 3, 2, 1), conv2d_bn(4 * ch, 4 * ch))
        self.dec2 = nn.Sequential(conv2d_bn(4 * ch + 2 * ch, 2 * ch), conv2d_bn(2 * ch, 2 * ch))
        self.dec1 = nn.Sequential(conv2d_bn(2 * ch + ch, ch), conv2d_bn(ch, out_ch))

    def rasterise(self, points: torch.Tensor, batch_size: int, has_time: bool):
        """-> image (B, 6, ne, nb), and per-point beam linear index (-1 = not a current-frame in-FOV point)."""
        b = points[:, 0].long()
        xyz = points[:, 1:4]
        inten = points[:, 4]
        cur = (points[:, 5] == 0) if (has_time and points.shape[1] >= 6) else torch.ones_like(b, dtype=torch.bool)
        bi, ei, r, ok = beam_index_torch(xyz, self.beam)
        ok = ok & cur
        lin = torch.where(ok, (b * self.ne + ei.clamp(0, self.ne - 1)) * self.nb + bi.clamp(0, self.nb - 1), torch.full_like(b, -1))
        img = points.new_zeros((batch_size * self.ne * self.nb, 6))
        idx = lin[ok]
        vals = torch.stack([r[ok] / 12.0, inten[ok], xyz[ok, 0] / 12.0, xyz[ok, 1] / 6.0, xyz[ok, 2] / 3.0, torch.ones_like(r[ok])], 1)
        img.index_copy_(0, idx, vals)
        img = img.view(batch_size, self.ne, self.nb, 6).permute(0, 3, 1, 2).contiguous()
        return img, lin

    def forward(self, points: torch.Tensor, batch_size: int, has_time: bool):
        img, lin = self.rasterise(points, batch_size, has_time)
        e1 = self.enc1(img)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        feat = d1.permute(0, 2, 3, 1).reshape(-1, self.out_ch)
        per_point = points.new_zeros((points.shape[0], self.out_ch))
        ok = lin >= 0
        per_point[ok] = feat[lin[ok]]
        return per_point

# ----------------------------------------------------------------------------- voxelisation
class Voxelizer:
    def __init__(self, grid: dict):
        self.xmin, self.ymin, self.zmin = grid["xmin"], grid["ymin"], grid["zmin"]
        self.vx, self.vy, self.vz = grid["vx"], grid["vy"], grid["vz"]
        self.W = int(round((grid["xmax"] - grid["xmin"]) / grid["vx"]))
        self.H = int(round((grid["ymax"] - grid["ymin"]) / grid["vy"]))
        self.D = int(round((grid["zmax"] - grid["zmin"]) / grid["vz"]))
        self.grid = grid

    @torch.no_grad()
    def __call__(self, points: torch.Tensor, batch_size: int):
        """points (M, 5) = [b, x, y, z, i] -> per-point features, voxel index, #voxels."""
        b = points[:, 0].long()
        xyz = points[:, 1:4]
        inten = points[:, 4:]                                  # intensity (+ time channel, ...)
        uvz = world_to_grid_torch(xyz, self.grid)              # == xyz unless the grid is polar
        xi = torch.floor((uvz[:, 0] - self.xmin) / self.vx).long()
        yi = torch.floor((uvz[:, 1] - self.ymin) / self.vy).long()
        zi = torch.floor((uvz[:, 2] - self.zmin) / self.vz).long()
        ok = (xi >= 0) & (xi < self.W) & (yi >= 0) & (yi < self.H) & (zi >= 0) & (zi < self.D)
        b, xyz, uvz, inten, xi, yi, zi = b[ok], xyz[ok], uvz[ok], inten[ok], xi[ok], yi[ok], zi[ok]
        lin = ((b * self.D + zi) * self.H + yi) * self.W + xi
        uniq, inv = torch.unique(lin, return_inverse=True)
        V = uniq.numel()
        counts = torch.zeros(V, device=points.device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float32))
        mean = torch.zeros(V, 3, device=points.device).index_add_(0, inv, xyz) / counts[:, None]
        vcenter = torch.stack([(xi.float() + 0.5) * self.vx + self.xmin,
                               (yi.float() + 0.5) * self.vy + self.ymin,
                               (zi.float() + 0.5) * self.vz + self.zmin], 1)
        if getattr(self, "drop_xyz", False):                      # GBlobs-style: no absolute coordinates
            feats = torch.cat([inten, xyz - mean[inv], uvz - vcenter], 1)      # (M, 6 + n_feat)
        elif getattr(self, "geo_feats", True):
            feats = torch.cat([xyz, inten, xyz - mean[inv], uvz - vcenter], 1)  # (M, 9 + n_feat)
        else:                                                     # range-view-only: xyz + extra columns
            feats = torch.cat([xyz, inten], 1)
        return feats, inv, uniq, V


def scatter_max(x: torch.Tensor, inv: torch.Tensor, V: int) -> torch.Tensor:
    out = x.new_full((V, x.shape[1]), float("-inf"))
    out.scatter_reduce_(0, inv[:, None].expand(-1, x.shape[1]), x, reduce="amax", include_self=True)
    return out


class DynamicVFE(nn.Module):
    """VoxelNet-style VFE without a max-points-per-voxel cap."""

    def __init__(self, in_ch: int = 10, chs=(64, 128)):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(in_ch, chs[0]), nn.BatchNorm1d(chs[0]), nn.ReLU(inplace=True))
        self.l2 = nn.Sequential(nn.Linear(2 * chs[0], chs[1]), nn.BatchNorm1d(chs[1]), nn.ReLU(inplace=True))
        self.out_ch = chs[1]

    def forward(self, feats, inv, V):
        f = self.l1(feats)
        f = torch.cat([f, scatter_max(f, inv, V)[inv]], 1)
        f = self.l2(f)
        return scatter_max(f, inv, V)


# ----------------------------------------------------------------------------- backbones
def conv3d_bn(i, o, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv3d(i, o, k, s, p, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True))


def conv2d_bn(i, o, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Dense3DMiddle(nn.Module):
    """Collapses the z axis D -> 2 with strided 3D convs, then flattens z into channels."""

    def __init__(self, in_ch: int, ch: int, D: int):
        super().__init__()
        layers = [conv3d_bn(in_ch, ch), conv3d_bn(ch, ch)]
        d = D
        while d > 2:
            if d >= 4:
                layers += [conv3d_bn(ch, ch, 3, (2, 1, 1), 1)]
                d = (d + 2 - 3) // 2 + 1
            else:  # d == 3
                layers += [conv3d_bn(ch, ch, (2, 3, 3), (1, 1, 1), (0, 1, 1))]
                d = d - 1
            layers += [conv3d_bn(ch, ch)]
        self.net = nn.Sequential(*layers)
        self.out_ch = ch * d
        self.d_out = d

    def forward(self, x):  # (B, C, D, H, W)
        x = self.net(x)
        B, C, D, H, W = x.shape
        return x.reshape(B, C * D, H, W)


class BEVNeck(nn.Module):
    """SECOND-style multi-scale 2D encoder; every scale is resized back to stride `out_stride`
    (relative to the voxel grid) and concatenated."""

    def __init__(self, in_ch: int, ch: int = 128, out_stride: int = 2, n_blocks=(4, 6, 6), up_ch: int = 256):
        super().__init__()
        self.out_stride = out_stride
        chs = [ch, ch * 2, ch * 2]
        blocks, ups = [], []
        prev = in_ch
        for bi, (c, n) in enumerate(zip(chs, n_blocks)):
            layers = [conv2d_bn(prev, c, 3, 2, 1)] + [conv2d_bn(c, c) for _ in range(n - 1)]
            blocks.append(nn.Sequential(*layers))
            ups.append(conv2d_bn(c, up_ch, 1, 1, 0))
            prev = c
        self.blocks = nn.ModuleList(blocks)
        self.ups = nn.ModuleList(ups)
        self.out_ch = up_ch * len(chs)

    def forward(self, x):
        H, W = x.shape[-2] // self.out_stride, x.shape[-1] // self.out_stride
        outs = []
        for blk, up in zip(self.blocks, self.ups):
            x = blk(x)
            y = up(x)
            if y.shape[-2:] != (H, W):
                y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(y)
        return torch.cat(outs, 1)


class CenterHead(nn.Module):
    def __init__(self, in_ch: int, head_ch: int = 64, z_bins: int = 0):
        super().__init__()
        self.shared = conv2d_bn(in_ch, head_ch)
        self.hm = nn.Sequential(conv2d_bn(head_ch, head_ch), nn.Conv2d(head_ch, 1, 1))
        self.hm[-1].bias.data.fill_(-2.19)
        self.branches = nn.ModuleDict({
            name: nn.Sequential(conv2d_bn(head_ch, head_ch), nn.Conv2d(head_ch, n, 1))
            for name, n in (("offset", 2), ("z", 1), ("dim", 3), ("rot", 6))
        })
        self.iou = nn.Sequential(conv2d_bn(head_ch, head_ch), nn.Conv2d(head_ch, 1, 1))
        # sonar-native elevation decoding: classify z over bins, regress the residual inside the bin
        self.zcls = nn.Sequential(conv2d_bn(head_ch, head_ch), nn.Conv2d(head_ch, z_bins, 1)) if z_bins > 0 else None

    def forward(self, x):
        x = self.shared(x)
        reg = torch.cat([self.branches[k](x) for k in ("offset", "z", "dim", "rot")], 1)
        out = {"hm": self.hm(x), "reg": reg, "iou": self.iou(x)}
        if self.zcls is not None:
            out["zcls"] = self.zcls(x)
        return out


def z_bin_centres(cfg: dict, device):
    nb = int(cfg.get("model", {}).get("z_bins", 0))
    g = cfg["grid"]
    bw = (g["zmax"] - g["zmin"]) / nb
    return g["zmin"] + (torch.arange(nb, device=device, dtype=torch.float32) + 0.5) * bw, bw


class SonarVoxNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get("model", {})
        self.vox = Voxelizer(cfg["grid"])
        self.has_time = cfg.get("data", {}).get("n_prev", 0) > 0
        n_feat = 1 + (1 if self.has_time else 0)
        n_feat += 10 if int(cfg.get("data", {}).get("local_feats", 0)) > 0 else 0   # sonar-native local structure channels
        self.beam = {**BEAM_DEFAULT, **cfg.get("data", {}).get("beam", {})}
        rv = m.get("rv", {}) or {}
        self.rv_mode = rv.get("mode", "off")                      # off | fuse | only
        self.rv = RangeViewBranch(self.beam, int(rv.get("ch", 32)), int(rv.get("out_ch", 32))) if self.rv_mode != "off" else None
        rv_ch = self.rv.out_ch if self.rv is not None else 0
        if self.rv_mode == "only":
            self.vox.geo_feats = False
            self.vfe = DynamicVFE(3 + n_feat + rv_ch, tuple(m.get("vfe_ch", [64, 128])))
        else:
            if m.get("drop_xyz", False):
                self.vox.drop_xyz = True
                self.vfe = DynamicVFE(6 + n_feat + rv_ch, tuple(m.get("vfe_ch", [64, 128])))
            else:
                self.vfe = DynamicVFE(9 + n_feat + rv_ch, tuple(m.get("vfe_ch", [64, 128])))
        self.visibility = bool(m.get("visibility", False))
        self.z_relative = bool(cfg.get("data", {}).get("z_relative", False))
        self.out_stride = cfg["out_stride"]
        self.mid = Dense3DMiddle(self.vfe.out_ch + (3 if self.visibility else 0), m.get("bb3d_ch", 64), self.vox.D)
        self.neck = BEVNeck(self.mid.out_ch, m.get("neck_ch", 128), cfg["out_stride"],
                            tuple(m.get("neck_blocks", [4, 6, 6])), m.get("neck_up_ch", 256))
        self.head = CenterHead(self.neck.out_ch, m.get("head_ch", 64), int(m.get("z_bins", 0)))
        self.head_drop = nn.Dropout2d(m.get("head_dropout", 0.0)) if m.get("head_dropout", 0.0) > 0 else nn.Identity()

    def forward(self, points: torch.Tensor, batch_size: int):
        if self.rv is not None:                                   # append per-point beam features as extra columns
            points = torch.cat([points, self.rv(points, batch_size, self.has_time)], 1)
        feats, inv, uniq, V = self.vox(points, batch_size)
        C = self.vfe.out_ch
        D, H, W = self.vox.D, self.vox.H, self.vox.W
        dense = points.new_zeros((batch_size * D * H * W, C))
        if V > 0:
            dense[uniq] = self.vfe(feats, inv, V)
        dense = dense.view(batch_size, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        if self.visibility:
            dense = torch.cat([dense, visibility_volume(points, batch_size, self.vox, self.beam, self.has_time)], 1)
        bev = self.mid(dense)
        bev = self.neck(bev)
        out = self.head(self.head_drop(bev))
        if self.z_relative:
            out["zref"] = self.zref_map(points, batch_size)
        return out

    @torch.no_grad()
    def zref_map(self, points: torch.Tensor, batch_size: int) -> torch.Tensor:
        """(B, 1, Ho, Wo) mean z of current-frame returns in the 3x3 output-cell neighbourhood;
        0 where there are none (same definition as targets.zref_map)."""
        s = self.out_stride
        sx, sy = self.vox.vx * s, self.vox.vy * s
        Wo, Ho = self.vox.W // s, self.vox.H // s
        b = points[:, 0].long()
        xyz = points[:, 1:4]
        if self.has_time and points.shape[1] >= 6:
            cur = points[:, 5] == 0
            b, xyz = b[cur], xyz[cur]
        g = world_to_grid_torch(xyz, self.vox.grid)
        xi = torch.floor((g[:, 0] - self.vox.xmin) / sx).long()
        yi = torch.floor((g[:, 1] - self.vox.ymin) / sy).long()
        ok = (xi >= 0) & (xi < Wo) & (yi >= 0) & (yi < Ho)
        lin = (b[ok] * Ho + yi[ok]) * Wo + xi[ok]
        ssum = torch.zeros(batch_size * Ho * Wo, device=points.device).index_add_(0, lin, xyz[ok, 2])
        cnt = torch.zeros_like(ssum).index_add_(0, lin, torch.ones_like(lin, dtype=torch.float32))
        ssum = F.avg_pool2d(ssum.view(batch_size, 1, Ho, Wo), 3, 1, 1, divisor_override=1)
        cnt = F.avg_pool2d(cnt.view(batch_size, 1, Ho, Wo), 3, 1, 1, divisor_override=1)
        return torch.where(cnt > 0, ssum / cnt.clamp(min=1), torch.zeros_like(ssum))


# ----------------------------------------------------------------------------- loss
def gaussian_focal_loss(pred_logits: torch.Tensor, gt: torch.Tensor, alpha=2.0, beta=4.0):
    p = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg_w = (1 - gt).pow(beta)
    pos_loss = torch.log(p) * (1 - p).pow(alpha) * pos
    neg_loss = torch.log(1 - p) * p.pow(alpha) * neg_w * (1 - pos)
    num_pos = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos.clamp(min=1.0)


def gather_feat(feat: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
    """feat (B, C, H, W), inds (B, K) -> (B, K, C)."""
    B, C = feat.shape[:2]
    flat = feat.view(B, C, -1).permute(0, 2, 1)
    return flat.gather(1, inds[:, :, None].expand(-1, -1, C))


def detection_loss(out: dict, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    lcfg = cfg["loss"]
    hm_loss = gaussian_focal_loss(out["hm"], batch["hm"], lcfg.get("alpha", 2.0), lcfg.get("beta", 4.0))
    pred = gather_feat(out["reg"], batch["inds"])            # (B, K, 12)
    tgt, mask = batch["reg"], batch["mask"]
    zcls_loss = None
    if "zcls" in out:                                        # z = bin centre + residual
        centres, bw = z_bin_centres(cfg, pred.device)
        zb = ((tgt[..., 2] - cfg["grid"]["zmin"]) / bw).floor().long().clamp(0, centres.numel() - 1)
        tgt = tgt.clone()
        tgt[..., 2] = tgt[..., 2] - centres[zb]
        logits = gather_feat(out["zcls"], batch["inds"])     # (B, K, nb)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), zb.reshape(-1), reduction="none").view_as(zb)
        zcls_loss = (ce * mask).sum() / mask.sum().clamp(min=1.0)
    l_geo = (pred[..., :6] - tgt[..., :6]).abs().sum(-1)
    l_pos = (pred[..., 6:] - tgt[..., 6:]).abs().sum(-1)
    if lcfg.get("rot_sym", True):                            # Rz(pi) maps the cuboid onto itself
        l_rot = torch.minimum(l_pos, (pred[..., 6:] + tgt[..., 6:]).abs().sum(-1))
    else:
        l_rot = l_pos
    n = mask.sum().clamp(min=1.0)
    geo = (l_geo * mask).sum() / n
    rot = (l_rot * mask).sum() / n
    w_rot = lcfg.get("rot_weight", 1.0)
    reg_loss = geo + w_rot * rot
    total = lcfg.get("hm_weight", 1.0) * hm_loss + lcfg.get("reg_weight", 1.0) * reg_loss
    parts = {"hm": hm_loss.item(), "geo": geo.item(), "rot": rot.item()}
    if zcls_loss is not None:
        total = total + lcfg.get("zcls_weight", 1.0) * zcls_loss
        parts["zcls"] = zcls_loss.item()
    w_iou = lcfg.get("iou_weight", 0.0)
    if w_iou > 0 and "iou" in out:
        from .evalio import official
        ev = official(cfg["data_root"])
        g, s = cfg["grid"], cfg["out_stride"]
        sx, sy = g["vx"] * s, g["vy"] * s
        Wo = int(round((g["xmax"] - g["xmin"]) / sx))
        inds = batch["inds"]
        cx = (inds % Wo).float()
        cy = (inds // Wo).float()
        with torch.no_grad():
            def np_boxes(t):
                c = torch.stack([(cx + t[..., 0]) * sx + g["xmin"], (cy + t[..., 1]) * sy + g["ymin"], t[..., 2]], -1)
                d = t[..., 3:6].clamp(-4.0, 3.0).exp()
                R = gram_schmidt_6d_torch(t[..., 6:12].reshape(-1, 6)).view(*t.shape[:2], 3, 3)
                return c.double().cpu().numpy(), d.double().cpu().numpy(), R.double().cpu().numpy()
            pc, pd, pR = np_boxes(pred)
            gc_, gd, gR = np_boxes(tgt)
            m_np = mask.cpu().numpy()
            tgt_iou = torch.zeros_like(mask)
            for b in range(mask.shape[0]):
                for k in range(mask.shape[1]):
                    if m_np[b, k] > 0:
                        try:
                            tgt_iou[b, k] = ev.iou_3d(ev.Box(center=pc[b, k], dims=pd[b, k], R=ev.project_to_so3(pR[b, k])),
                                                      ev.Box(center=gc_[b, k], dims=gd[b, k], R=ev.project_to_so3(gR[b, k])))
                        except Exception:
                            tgt_iou[b, k] = 0.0
        iou_pred = gather_feat(out["iou"], inds)[..., 0]
        iou_loss = (torch.nn.functional.binary_cross_entropy_with_logits(iou_pred, tgt_iou, reduction="none") * mask).sum() / n
        total = total + w_iou * iou_loss
        parts["iou"] = iou_loss.item()
    w_corner = lcfg.get("corner_weight", 0.0)
    if w_corner > 0:
        # L1 on the 8 box corners in the sonar frame, folding the Rz(pi) symmetry (corner
        # permutation); couples centre, extents and rotation the way 3D IoU does
        g, s = cfg["grid"], cfg["out_stride"]
        sx, sy = g["vx"] * s, g["vy"] * s
        Wo = int(round((g["xmax"] - g["xmin"]) / sx))
        inds = batch["inds"]
        cx = (inds % Wo).float()
        cy = (inds // Wo).float()

        def boxes(t):
            c = torch.stack([(cx + t[..., 0]) * sx + g["xmin"], (cy + t[..., 1]) * sy + g["ymin"], t[..., 2]], -1)
            d = t[..., 3:6].clamp(-4.0, 3.0).exp()
            R = gram_schmidt_6d_torch(t[..., 6:12].reshape(-1, 6)).view(*t.shape[:2], 3, 3)
            tmpl = torch.tensor(CORNER_TEMPLATE, dtype=t.dtype, device=t.device)        # (8, 3)
            corners = torch.einsum("bkij,bkcj->bkci", R, tmpl[None, None] * d[:, :, None, :]) + c[:, :, None, :]
            return corners                                                                 # (B, K, 8, 3)

        pc, gc = boxes(pred), boxes(tgt)
        perm = [2, 3, 0, 1, 6, 7, 4, 5]                                                   # Rz(pi) corner order
        l1 = (pc - gc).abs().mean((-1, -2))
        l2 = (pc[:, :, perm] - gc).abs().mean((-1, -2))
        corner = (torch.minimum(l1, l2) * mask).sum() / n
        total = total + w_corner * corner
        parts["corner"] = corner.item()
    return total, parts


# ----------------------------------------------------------------------------- decode
@torch.no_grad()
def decode(out: dict, cfg: dict, K: int = 100):
    """-> per-sample dict of tensors: score (K,), center (K,3), dims (K,3), R (K,3,3)."""
    g, s = cfg["grid"], cfg["out_stride"]
    sx, sy = g["vx"] * s, g["vy"] * s
    hm = torch.sigmoid(out["hm"])
    B, _, Ho, Wo = hm.shape
    keep = (F.max_pool2d(hm, 3, 1, 1) == hm).float()
    hm = hm * keep
    scores, inds = hm.view(B, -1).topk(min(K, Ho * Wo))
    ys, xs = (inds // Wo).float(), (inds % Wo).float()
    reg = gather_feat(out["reg"], inds)                      # (B, K, 12)
    x = (xs + reg[..., 0]) * sx + g["xmin"]
    y = (ys + reg[..., 1]) * sy + g["ymin"]
    x, y = grid_to_world_torch(x, y, g)
    if "zcls" in out:
        centres, bw = z_bin_centres(cfg, hm.device)
        zb = gather_feat(out["zcls"], inds).argmax(-1)
        z = (centres[zb] + reg[..., 2].clamp(-bw, bw)).clamp(g["zmin"] - 2.0, g["zmax"] + 2.0)
    else:
        z = reg[..., 2]
        if "zref" in out:                                 # evidence-relative z target
            z = z + gather_feat(out["zref"], inds)[..., 0]
        z = z.clamp(g["zmin"] - 2.0, g["zmax"] + 2.0)
    center = torch.stack([x, y, z], -1)
    dims = reg[..., 3:6].clamp(-4.0, 3.0).exp()           # keep garbage peaks finite
    R = gram_schmidt_6d_torch(reg[..., 6:12].reshape(-1, 6)).view(B, -1, 3, 3)
    alpha = cfg["decode"].get("iou_alpha", 0.0)
    if alpha > 0 and "iou" in out:
        iou = torch.sigmoid(gather_feat(out["iou"], inds)[..., 0]).clamp(1e-4, 1.0)
        scores = scores.clamp(1e-4, 1.0).pow(1 - alpha) * iou.pow(alpha)
    return [{"score": scores[b], "center": center[b], "dims": dims[b], "R": R[b]} for b in range(B)]
