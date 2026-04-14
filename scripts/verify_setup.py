#!/usr/bin/env python3
"""
Startup verification script - runs before server starts.
Tests that all pipeline components are correctly installed.
Exit 0 = pass, Exit 1 = fail
"""
import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("Checking package imports...")
    packages = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy"),
        ("faster_whisper", "Faster Whisper"),
        ("pyannote.audio", "PyAnnote Audio"),
        ("speechbrain", "SpeechBrain"),
        ("noisereduce", "NoiseReduce"),
        ("transformers", "Transformers"),
    ]
    failed = []
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)
    if failed:
        print(f"\nFailed imports: {', '.join(failed)}")
        return False
    return True

def test_torch_cuda():
    """Test that PyTorch can access CUDA."""
    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("  ✗ CUDA not available (CPU mode)")
            return False
    except Exception as e:
        print(f"  ✗ CUDA check failed: {e}")
        return False

def test_diarization_load():
    """Test that pyannote diarization pipeline can load (without running)."""
    print("\nChecking diarization model...")
    try:
        from pyannote.audio import Pipeline
        import inspect
        sig = inspect.signature(Pipeline.from_pretrained)
        params = list(sig.parameters.keys())
        print(f"  ✓ Pipeline.from_pretrained params: {params}")
        # Check for token/auth parameter
        token_params = [p for p in params if 'auth' in p.lower() or 'token' in p.lower()]
        if token_params:
            print(f"    Token param: {token_params[0]}")
        else:
            print("    Note: No auth token param found (may use HF_TOKEN env var)")
        return True
    except Exception as e:
        print(f"  ✗ Diarization check failed: {e}")
        return False

def test_whisper_import():
    """Test that faster-whisper can be imported."""
    print("\nChecking Whisper...")
    try:
        from faster_whisper import WhisperModel
        print("  ✓ Faster Whisper imported")
        return True
    except Exception as e:
        print(f"  ✗ Whisper check failed: {e}")
        return False

def main():
    print("=" * 50)
    print("Voice AI Pipeline - Startup Verification")
    print("=" * 50)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_torch_cuda()))
    results.append(("Whisper", test_whisper_import()))
    results.append(("Diarization", test_diarization_load()))

    print("\n" + "=" * 50)
    print("Summary:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✓ All checks passed - server can start")
        sys.exit(0)
    else:
        print("\n✗ Some checks failed - fix before starting server")
        sys.exit(1)

if __name__ == "__main__":
    main()
