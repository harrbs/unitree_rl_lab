"""Export Baseline E (PointCloudActorCritic) checkpoint to ONNX.

Standalone script — Isaac Lab / AppLauncher 불필요.

ONNX 인터페이스:
  inputs : policy      [1, 47]    (고유감각: ang_vel 3 + gravity 3 + cmd 3 + jpos 12 + jvel 12 + action 12 + gait 2)
           point_cloud [1, 288]   (지형 pointcloud: 96 points × 3)
           h_in        [1, 1, 256] (GRU hidden state)
  outputs: actions [1, 12]
           h_out   [1, 1, 256]

사용법:
  python scripts/rsl_rl/export_stair_e.py \\
      logs/rsl_rl/go2_stair_E_pointcloud/2026-03-29_00-15-27/model_1300.pt
"""

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

# PointCloudEncoder 파일만 직접 import (Isaac Lab / pxr 없이 동작)
_ENCODER_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/policies",
)
sys.path.insert(0, os.path.abspath(_ENCODER_PATH))
from point_cloud_encoder import PointCloudEncoder  # noqa: E402

# ── 아키텍처 상수 (학습과 반드시 일치) ──────────────────────────────────────
PROP_DIM    = 47
PC_DIM      = 288           # 96 points × 3
GRU_HIDDEN  = 256
PC_OUT      = 64
GRU_IN_DIM  = PROP_DIM + PC_OUT    # 111  — concat(policy, z_pc) → GRU 입력
FUSED_DIM   = GRU_HIDDEN + PC_OUT  # 320  — concat(z_rnn, z_pc)  → fusion 입력
NUM_ACTIONS = 12


def _make_mlp(in_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    return nn.Sequential(*layers)


class StairEOnnxWrapper(nn.Module):
    """ONNX export용 래퍼 — actor branch 전용.

    C++ OrtRunner 인터페이스:
        policy      [1, 47]    → GRU 입력 (고유감각)
        point_cloud [1, 288]   → PointNet 입력 (지형 pointcloud)
        h_in        [1, 1, 256] → GRU 히든 스테이트 입력
        ──────────────────────────────────────────
        actions [1, 12]
        h_out   [1, 1, 256] → 다음 스텝에 h_in 으로 전달
    """

    def __init__(self) -> None:
        super().__init__()
        self.gru     = nn.GRU(GRU_IN_DIM, GRU_HIDDEN, num_layers=1, batch_first=False)
        self.pc_enc  = PointCloudEncoder(num_points=96, out_dim=PC_OUT)
        self.fusion  = _make_mlp(FUSED_DIM, [256, 128])
        self.head    = nn.Linear(128, NUM_ACTIONS)

    def forward(
        self,
        policy: torch.Tensor,       # [1, 47]
        point_cloud: torch.Tensor,  # [1, 288]
        h_in: torch.Tensor,         # [1, 1, 256]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # PointNet: pointcloud → z_pc
        z_pc = self.pc_enc(point_cloud)                              # [1, 64]

        # GRU: input = concat(policy, z_pc), seq=1, batch=1, in=111
        gru_in = torch.cat([policy, z_pc], dim=-1)                   # [1, 111]
        z_rnn, h_out = self.gru(gru_in.unsqueeze(0), h_in)          # [1,1,256], [1,1,256]
        z_rnn = z_rnn.squeeze(0)                                     # [1, 256]

        # Fusion: concat(z_rnn, z_pc) → actions
        fused   = torch.cat([z_rnn, z_pc], dim=-1)                   # [1, 320]
        actions = self.head(self.fusion(fused))                       # [1, 12]
        return actions, h_out


def load_weights(wrapper: StairEOnnxWrapper, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    model_sd = state["model_state_dict"]

    # wrapper 파라미터 이름 → checkpoint 키 매핑
    mapping = {
        "gru":    "memory_a.rnn",
        "pc_enc": "pc_enc_a",
        "fusion": "fusion_a",
        "head":   "actor_head",
    }

    new_sd: dict[str, torch.Tensor] = {}
    for wrapper_prefix, ckpt_prefix in mapping.items():
        for k, v in model_sd.items():
            if k.startswith(ckpt_prefix + "."):
                new_key = wrapper_prefix + k[len(ckpt_prefix):]
                new_sd[new_key] = v

    missing, unexpected = wrapper.load_state_dict(new_sd, strict=True)
    if missing:
        print(f"[WARNING] missing keys: {missing}")
    if unexpected:
        print(f"[WARNING] unexpected keys: {unexpected}")


def export(ckpt_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "policy.onnx")

    wrapper = StairEOnnxWrapper()
    load_weights(wrapper, ckpt_path)
    wrapper.eval()
    wrapper.cpu()

    policy_dummy = torch.zeros(1, PROP_DIM)      # [1, 47]
    pc_dummy     = torch.zeros(1, PC_DIM)        # [1, 288]
    h_dummy      = torch.zeros(1, 1, GRU_HIDDEN) # [1, 1, 256]

    torch.onnx.export(
        wrapper,
        (policy_dummy, pc_dummy, h_dummy),
        out_path,
        opset_version=18,
        input_names=["policy", "point_cloud", "h_in"],
        output_names=["actions", "h_out"],
        dynamic_axes={},
    )
    print(f"[OK] exported → {out_path}")

    # 검증
    try:
        import onnx
        model = onnx.load(out_path)
        onnx.checker.check_model(model)
        for inp in model.graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  input  {inp.name}: {dims}")
        for out in model.graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  output {out.name}: {dims}")
    except ImportError:
        print("  (onnx 패키지 없음 — 검증 생략)")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Baseline E to ONNX")
    parser.add_argument("checkpoint", help="model_XXXX.pt 경로")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="출력 디렉토리 (기본: <run_dir>/exported)",
    )
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    run_dir   = os.path.dirname(ckpt_path)          # model_XXXX.pt가 있는 run 디렉토리
    out_dir   = args.out_dir or os.path.join(run_dir, "exported")

    export(ckpt_path, out_dir)


if __name__ == "__main__":
    main()
