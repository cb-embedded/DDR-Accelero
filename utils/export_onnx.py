#!/usr/bin/env python3
"""
Export trained Keras model to ONNX format.

This script converts the trained Keras model to ONNX format.
Note: Requires tf2onnx package (pip install tf2onnx)

Usage:
    python -m utils.export_onnx [--model-path PATH] [--output PATH]

Example:
    python -m utils.export_onnx --model-path artifacts/trained_model.h5 --output docs/model.onnx
"""

import argparse
import sys
from pathlib import Path

try:
    import tf2onnx
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("Error: tf2onnx is required for ONNX export")
    print("Install with: pip install tf2onnx")
    sys.exit(1)


def export_to_onnx(model_path, output_path):
    """
    Export Keras model to ONNX format.
    
    Args:
        model_path: Path to trained Keras model (.h5 file)
        output_path: Path to save ONNX model (.onnx file)
    """
    print("="*70)
    print("EXPORT KERAS MODEL TO ONNX")
    print("="*70)
    
    # Load trained model
    print(f"\n[1/3] Loading trained model from: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    print("  ✓ Model loaded successfully")
    print(f"  - Input shape: {model.input_shape}")
    print(f"  - Output shape: {model.output_shape}")
    
    # Test forward pass
    print("\n[2/3] Testing forward pass...")
    import numpy as np
    dummy_input = np.random.randn(1, 198, 9).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"  ✓ Forward pass successful")
    print(f"  - Output shape: {output.shape}")
    
    # Export to ONNX
    print(f"\n[3/3] Exporting to ONNX format: {output_path}")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert using tf2onnx
    spec = (tf.TensorSpec((None, 198, 9), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    # Save ONNX model
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print(f"  ✓ ONNX model saved successfully")
    
    # Get file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  - File size: {size_mb:.2f} MB")
    
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\n✓ ONNX model saved to: {output_path}")
    print("\nYou can now use this model with ONNX Runtime:")
    print("  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{output_path}')")
    print("  output = session.run(None, {'input': input_data})")


def main():
    parser = argparse.ArgumentParser(
        description='Export Keras model to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default paths
  python -m utils.export_onnx

  # Export with custom paths
  python -m utils.export_onnx --model-path my_model.h5 --output docs/model.onnx
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='artifacts/trained_model.h5',
        help='Path to trained Keras model (default: artifacts/trained_model.h5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='docs/model.onnx',
        help='Path to save ONNX model (default: docs/model.onnx)'
    )
    
    args = parser.parse_args()
    
    try:
        export_to_onnx(args.model_path, args.output)
    except Exception as e:
        print(f"\n❌ Error during export: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
