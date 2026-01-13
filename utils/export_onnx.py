#!/usr/bin/env python3
"""
Export trained PyTorch model to ONNX format.

This script converts the trained ArrowCNN model to ONNX format.

Usage:
    python -m utils.export_onnx [--model-path PATH] [--output PATH]

Example:
    python -m utils.export_onnx --model-path artifacts/trained_model.pth --output docs/model.onnx
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.onnx

# Import model architecture
from train_model import ArrowCNN


def export_to_onnx(model_path, output_path, seq_length=198):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model_path: Path to trained PyTorch model (.pth file)
        output_path: Path to save ONNX model (.onnx file)
        seq_length: Expected sequence length (default: 198)
    """
    print("="*70)
    print("EXPORT PYTORCH MODEL TO ONNX")
    print("="*70)
    
    # Load trained model
    print(f"\n[1/4] Loading trained model from: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize model
    model = ArrowCNN(input_channels=9, seq_length=seq_length)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("  ✓ Model loaded successfully")
    print(f"  - Architecture: ArrowCNN")
    print(f"  - Input channels: 9")
    print(f"  - Sequence length: {seq_length}")
    
    # Create dummy input for tracing
    print("\n[2/4] Creating dummy input for model tracing...")
    dummy_input = torch.randn(1, 9, seq_length)
    print(f"  - Input shape: {dummy_input.shape} (batch_size, channels, time_steps)")
    
    # Test forward pass
    print("\n[3/4] Testing forward pass...")
    with torch.no_grad():
        arrows_out = model(dummy_input)
        print(f"  ✓ Forward pass successful")
        print(f"  - Arrows output shape: {arrows_out.shape}")
    
    # Export to ONNX
    print(f"\n[4/4] Exporting to ONNX format: {output_path}")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,                          # Model to export
        dummy_input,                    # Example input
        output_path,                    # Output file path
        export_params=True,             # Store trained parameters
        opset_version=11,               # ONNX opset version
        do_constant_folding=True,       # Optimize constant folding
        input_names=['input'],          # Input tensor name
        output_names=['arrows'],        # Output tensor name
        dynamic_axes={
            'input': {0: 'batch_size'},     # Variable batch size
            'arrows': {0: 'batch_size'}
        }
    )
    
    print(f"  ✓ Model exported successfully")
    print(f"  - File size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    
    # Verify ONNX model
    print("\n[Verification] Loading ONNX model for verification...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model is valid")
        
        # Print model info
        print("\n[Model Info]")
        print(f"  - Producer: {onnx_model.producer_name}")
        print(f"  - IR version: {onnx_model.ir_version}")
        print(f"  - Opset version: {onnx_model.opset_import[0].version}")
        
    except ImportError:
        print("  ⚠ onnx package not installed, skipping verification")
        print("    Install with: pip install onnx")
    
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Copy {output_path} to your web server or CDN")
    print(f"2. Update inference.js to load the ONNX model")
    print(f"3. Test in browser with real sensor data")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Export PyTorch model to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default paths
  python -m utils.export_onnx

  # Export with custom paths
  python -m utils.export_onnx --model-path my_model.pth --output docs/model.onnx
  
  # Export with custom sequence length
  python -m utils.export_onnx --seq-length 256
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='artifacts/trained_model.pth',
        help='Path to trained PyTorch model (default: artifacts/trained_model.pth)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='docs/model.onnx',
        help='Path to save ONNX model (default: docs/model.onnx)'
    )
    
    parser.add_argument(
        '--seq-length',
        type=int,
        default=198,
        help='Expected sequence length (default: 198)'
    )
    
    args = parser.parse_args()
    
    try:
        export_to_onnx(args.model_path, args.output, args.seq_length)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
