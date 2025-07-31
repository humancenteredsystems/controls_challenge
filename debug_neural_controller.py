#!/usr/bin/env python3
"""
Debug Neural Controller Initialization
Isolates the Tournament #3 failure root cause
"""
import sys
sys.path.append('.')

import onnxruntime as ort
from pathlib import Path
import traceback

def test_neural_controller_init():
    print('🔍 Neural Controller Debug Test')
    print('=' * 50)
    
    # Test 1: Basic imports
    print('\n1. Testing imports...')
    try:
        from controllers.neural_blended import Controller as NeuralController
        print('✅ Neural controller import successful')
    except Exception as e:
        print(f'❌ Neural controller import failed: {e}')
        traceback.print_exc()
        return
    
    # Test 2: Check available ONNX Runtime providers
    print('\n2. Checking ONNX Runtime providers...')
    available_providers = ort.get_available_providers()
    print(f'Available providers: {available_providers}')
    
    # Test 3: Check neural models availability
    print('\n3. Checking neural models...')
    models_dir = Path('models')
    blender_models = list(models_dir.glob('blender_*.onnx'))
    print(f'Found {len(blender_models)} blender models')
    if blender_models:
        test_model = str(blender_models[0])
        print(f'Test model: {test_model}')
    else:
        print('❌ No blender models found!')
        return
    
    # Test 4: Try ONNX Runtime session creation directly
    print('\n4. Testing ONNX Runtime session creation...')
    
    # Test with CUDA first
    print('   Testing CUDA provider...')
    try:
        session_options = ort.SessionOptions()
        cuda_session = ort.InferenceSession(
            test_model,
            sess_options=session_options,
            providers=['CUDAExecutionProvider']
        )
        print('   ✅ CUDA provider successful')
    except Exception as e:
        print(f'   ⚠️ CUDA provider failed: {e}')
        
    # Test with CPU fallback
    print('   Testing CPU provider...')
    try:
        session_options = ort.SessionOptions()
        cpu_session = ort.InferenceSession(
            test_model,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        print('   ✅ CPU provider successful')
    except Exception as e:
        print(f'   ❌ CPU provider failed: {e}')
        traceback.print_exc()
        return
    
    # Test 5: Try neural controller initialization without model path (should use fallback)
    print('\n5. Testing neural controller init (no model)...')
    try:
        controller_no_model = NeuralController()
        print('✅ Neural controller init (no model) successful')
        print(f'   Blender session: {controller_no_model.blender_session}')
    except Exception as e:
        print(f'❌ Neural controller init (no model) failed: {e}')
        traceback.print_exc()
    
    # Test 6: Try neural controller initialization with specific model
    print('\n6. Testing neural controller init (with model)...')
    try:
        controller_with_model = NeuralController(blender_model_path=test_model)
        print('✅ Neural controller init (with model) successful')
        print(f'   Blender session: {controller_with_model.blender_session}')
    except Exception as e:
        print(f'❌ Neural controller init (with model) failed: {e}')
        traceback.print_exc()
    
    # Test 7: Try tournament archive loading
    print('\n7. Testing tournament archive loading...')
    try:
        import json
        with open('plans/tournament_archive.json', 'r') as f:
            archive = json.load(f)
        print('✅ Tournament archive loaded successfully')
        best = min(archive['archive'], key=lambda x: x.get('stats', {}).get('avg_total_cost', float('inf')))
        print(f'   Best cost: {best["stats"]["avg_total_cost"]:.2f}')
        print(f'   Low gains: {best["low_gains"]}')
        print(f'   High gains: {best["high_gains"]}')
    except Exception as e:
        print(f'❌ Tournament archive loading failed: {e}')
        traceback.print_exc()

if __name__ == '__main__':
    test_neural_controller_init()