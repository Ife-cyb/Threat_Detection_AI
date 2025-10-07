#!/usr/bin/env python3
"""
Integration Test for Threat Detection AI Enhancements

This script validates all the new features without requiring actual data or trained models.
It tests the code structure, imports, and basic functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    tests = []
    
    # Test train.py imports
    try:
        from train import train, build_model, load_processed, get_features_targets
        tests.append(("train.py imports", True))
    except Exception as e:
        tests.append(("train.py imports", False, str(e)))
    
    # Test retrain.py imports
    try:
        from retrain import retrain, load_new_data, combine_datasets, evaluate_model
        tests.append(("retrain.py imports", True))
    except Exception as e:
        tests.append(("retrain.py imports", False, str(e)))
    
    # Test multi_dataset.py imports
    try:
        from multi_dataset import (
            DatasetAdapter, CICIDS2017Adapter, NSLKDDAdapter, 
            get_adapter, prepare_dataset
        )
        tests.append(("multi_dataset.py imports", True))
    except Exception as e:
        tests.append(("multi_dataset.py imports", False, str(e)))
    
    # Test serve_api.py imports
    try:
        from serve_api import (
            app, FlowRecord, PredictRequest, PredictResponse, 
            BatchPredictResponse
        )
        tests.append(("serve_api.py imports", True))
    except Exception as e:
        tests.append(("serve_api.py imports", False, str(e)))
    
    return tests


def test_dataset_adapters():
    """Test dataset adapter creation."""
    print("\nTesting dataset adapters...")
    
    tests = []
    
    try:
        from multi_dataset import get_adapter, DatasetAdapter
        
        # Test CICIDS2017 adapter
        try:
            adapter = get_adapter('cicids2017')
            assert adapter.name == 'cicids2017'
            tests.append(("CICIDS2017 adapter creation", True))
        except Exception as e:
            tests.append(("CICIDS2017 adapter creation", False, str(e)))
        
        # Test NSL-KDD adapter
        try:
            adapter = get_adapter('nsl_kdd')
            assert adapter.name == 'nsl_kdd'
            tests.append(("NSL-KDD adapter creation", True))
        except Exception as e:
            tests.append(("NSL-KDD adapter creation", False, str(e)))
        
        # Test invalid adapter
        try:
            adapter = get_adapter('invalid')
            tests.append(("Invalid adapter handling", False, "Should raise ValueError"))
        except ValueError:
            tests.append(("Invalid adapter handling", True))
        except Exception as e:
            tests.append(("Invalid adapter handling", False, str(e)))
            
    except Exception as e:
        tests.append(("Dataset adapter tests", False, str(e)))
    
    return tests


def test_model_builders():
    """Test model building functions."""
    print("\nTesting model builders...")
    
    tests = []
    
    try:
        from train import build_model
        
        # Test RandomForest
        try:
            model = build_model('random_forest')
            tests.append(("RandomForest model creation", True))
        except Exception as e:
            tests.append(("RandomForest model creation", False, str(e)))
        
        # Test XGBoost
        try:
            model = build_model('xgboost')
            tests.append(("XGBoost model creation", True))
        except Exception as e:
            tests.append(("XGBoost model creation", False, str(e)))
            
    except Exception as e:
        tests.append(("Model builder tests", False, str(e)))
    
    return tests


def test_api_models():
    """Test API request/response models."""
    print("\nTesting API models...")
    
    tests = []
    
    try:
        from serve_api import FlowRecord, PredictRequest, PredictResponse, BatchPredictResponse
        
        # Test FlowRecord
        try:
            record = FlowRecord(proto="tcp", service="http", spkts=10)
            tests.append(("FlowRecord creation", True))
        except Exception as e:
            tests.append(("FlowRecord creation", False, str(e)))
        
        # Test PredictRequest
        try:
            request = PredictRequest(
                model_name="random_forest",
                records=[FlowRecord(proto="tcp")]
            )
            tests.append(("PredictRequest creation", True))
        except Exception as e:
            tests.append(("PredictRequest creation", False, str(e)))
        
        # Test PredictResponse
        try:
            response = PredictResponse(
                prediction=[0, 1],
                threat_probability=[0.2, 0.8]
            )
            tests.append(("PredictResponse creation", True))
        except Exception as e:
            tests.append(("PredictResponse creation", False, str(e)))
        
        # Test BatchPredictResponse
        try:
            response = BatchPredictResponse(
                predictions=[0, 1, 0],
                threat_probabilities=[0.1, 0.9, 0.2],
                total_records=3,
                total_threats=1
            )
            tests.append(("BatchPredictResponse creation", True))
        except Exception as e:
            tests.append(("BatchPredictResponse creation", False, str(e)))
            
    except Exception as e:
        tests.append(("API model tests", False, str(e)))
    
    return tests


def print_results(test_name, results):
    """Print test results."""
    print(f"\n{test_name}:")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for result in results:
        if len(result) == 2:
            name, status = result
            if status:
                print(f"  ✓ {name}")
                passed += 1
            else:
                print(f"  ✗ {name}")
                failed += 1
        else:
            name, status, error = result
            if status:
                print(f"  ✓ {name}")
                passed += 1
            else:
                print(f"  ✗ {name}: {error}")
                failed += 1
    
    print(f"\nPassed: {passed}, Failed: {failed}")
    return passed, failed


def main():
    """Run all tests."""
    print("=" * 60)
    print("Threat Detection AI - Integration Tests")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # Run import tests
    results = test_imports()
    passed, failed = print_results("Import Tests", results)
    total_passed += passed
    total_failed += failed
    
    # Run dataset adapter tests
    results = test_dataset_adapters()
    passed, failed = print_results("Dataset Adapter Tests", results)
    total_passed += passed
    total_failed += failed
    
    # Run model builder tests
    results = test_model_builders()
    passed, failed = print_results("Model Builder Tests", results)
    total_passed += passed
    total_failed += failed
    
    # Run API model tests
    results = test_api_models()
    passed, failed = print_results("API Model Tests", results)
    total_passed += passed
    total_failed += failed
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} test(s) failed")
        print("Note: Some failures are expected if dependencies are not installed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
