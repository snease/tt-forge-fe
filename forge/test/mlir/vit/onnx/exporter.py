import torch
from pathlib import Path
import sys
import itertools
     import Encoder


def export_model(
    output_dir: Path,
    heads: int,
    batch: int,
    dim: int,
    seq: int,
    layers: int,
    mlp_size_multiple: int,
):
    mlp_dim = dim * mlp_size_multiple
    model = Encoder(
        seq_length=seq,
        num_layers=layers,
        num_heads=heads,
        hidden_dim=dim,
        mlp_dim=mlp_dim,
        dropout=0.0,
        attention_dropout=0.0,
    ).eval()

    model_name = (
        f"vit_encoder_b{batch}_s{seq}_d{dim}_m{mlp_dim}_h{heads}_l{layers}.onnx"
    )
    output_path = output_dir / model_name

    input_tensor = torch.randn(batch, seq, dim)

    torch.onnx.export(
        model,
        (input_tensor),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        do_constant_folding=True,
        verbose=False,
    )

    return output_path


def generate_models(output_dir: Path):
    batch = [1, 4, 8]
    dim = [512, 1024, 1536, 2048]
    seq = [1024, 2048, 3072]
    heads = 16
    layers = 12
    mlp_size_multiple = 4

    for b, d, s in itertools.product(batch, dim, seq):
        model_path = export_model(
            output_dir=output_dir,
            heads=heads,
            batch=b,
            dim=d,
            seq=s,
            layers=layers,
            mlp_size_multiple=mlp_size_multiple,
        )
        print(f"Saved to {model_path}")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python exporter.py <output_dir>"
    output_dir = Path(sys.argv[1])

    assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    generate_models(output_dir)
