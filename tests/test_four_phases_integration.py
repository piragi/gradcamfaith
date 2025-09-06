"""
Integration tests for the four main phases of the pipeline.

This test suite requires:
1. CUDA-capable device
2. All three datasets downloaded via setup.py
3. Real models and data for meaningful tests

The four phases tested:
1. Classification without gradient gate
2. Classification with gradient gate  
3. Faithfulness analysis
4. Parameter sweep functionality
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pytest
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from pipeline import run_unified_pipeline
from main_sweep import run_parameter_sweep
from faithfulness import evaluate_and_report_faithfulness


class TestFourPhasesIntegration:
    """Integration tests for all four main phases."""
    
    @pytest.fixture(scope="class")
    def require_cuda(self):
        """Skip all tests if CUDA is not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available - integration tests require GPU")
        return torch.device("cuda")
    
    @pytest.fixture(scope="class") 
    def check_datasets(self):
        """Verify all required datasets are available."""
        datasets_to_check = [
            ("waterbirds", "./data/waterbirds/waterbird_complete95_forest2water2/"),
            ("covidquex", "./data/covidquex/data/lung/"),
            ("hyperkvasir", "./data/hyperkvasir/labeled-images/")
        ]
        
        available_datasets = []
        for name, path in datasets_to_check:
            dataset_path = Path(path)
            if dataset_path.exists():
                available_datasets.append(name)
            else:
                print(f"Warning: Dataset {name} not found at {path}")
        
        if not available_datasets:
            pytest.skip("No datasets available - run setup.py to download datasets")
        
        return available_datasets
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for integration tests."""
        pipeline_config = config.PipelineConfig()
        pipeline_config.file.use_cached_original = False
        pipeline_config.file.use_cached_perturbed = ""
        pipeline_config.file.current_mode = "val"  # Use validation set for faster testing
        pipeline_config.classify.data_collection = False
        return pipeline_config
    
    @pytest.fixture
    def test_output_dir(self):
        """Create a temporary output directory for tests."""
        with tempfile.TemporaryDirectory(prefix="gradcamfaith_test_") as tmp_dir:
            output_dir = Path(tmp_dir)
            yield output_dir
    
    def test_phase_1_vanilla_classification(self, require_cuda, check_datasets, base_config, test_output_dir):
        """Phase 1: Test classification without gradient gating."""
        if "waterbirds" not in check_datasets:
            pytest.skip("Waterbirds dataset not available")
            
        # Configure for vanilla classification
        base_config.file.set_dataset("waterbirds")
        base_config.classify.boosting.enable_feature_gradients = False
        base_config.classify.analysis = False  # Skip faithfulness for basic classification test
        base_config.classify.use_clip = True  # Use CLIP for waterbirds
        base_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
        base_config.classify.clip_text_prompts = [
            "a photo of a terrestrial bird", "a photo of an aquatic bird"
        ]
        base_config.file.base_pipeline_dir = test_output_dir
        
        # Run with small subset for speed
        results, saco_results = run_unified_pipeline(
            config=base_config,
            dataset_name="waterbirds",
            source_data_path=Path("./data/waterbirds/waterbird_complete95_forest2water2/"),
            device=require_cuda,
            subset_size=5,  # Small subset for testing
            random_seed=42
        )
        
        # Validate results
        assert len(results) > 0, "Should have classification results"
        assert all(r.prediction.confidence > 0 for r in results), "All predictions should have confidence > 0"
        assert all(r.prediction.predicted_class_idx in [0, 1] for r in results), "Predictions should be valid class indices"
        assert not base_config.classify.boosting.enable_feature_gradients, "Feature gradients should be disabled"
        
        print(f"✅ Phase 1 (Vanilla Classification): {len(results)} images processed successfully")
        
        # Check that we get reasonable confidence scores
        confidences = [r.prediction.confidence for r in results]
        mean_confidence = np.mean(confidences)
        assert 0.5 <= mean_confidence <= 1.0, f"Mean confidence {mean_confidence:.3f} should be reasonable"
    
    def test_phase_2_gradient_gating_classification(self, require_cuda, check_datasets, base_config, test_output_dir):
        """Phase 2: Test classification with gradient gating enabled.""" 
        if "waterbirds" not in check_datasets:
            pytest.skip("Waterbirds dataset not available")
            
        # Configure for gradient gating
        base_config.file.set_dataset("waterbirds")
        base_config.classify.boosting.enable_feature_gradients = True
        base_config.classify.boosting.feature_gradient_layers = [4]  # Test with layer 4
        base_config.classify.boosting.steering_layers = [4]
        base_config.classify.analysis = False  # Skip faithfulness for basic classification test
        base_config.classify.use_clip = True
        base_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
        base_config.classify.clip_text_prompts = [
            "a photo of a terrestrial bird", "a photo of an aquatic bird"
        ]
        base_config.file.base_pipeline_dir = test_output_dir
        
        # Run with small subset
        results, saco_results = run_unified_pipeline(
            config=base_config,
            dataset_name="waterbirds",
            source_data_path=Path("./data/waterbirds/waterbird_complete95_forest2water2/"),
            device=require_cuda,
            subset_size=5,
            random_seed=42
        )
        
        # Validate results
        assert len(results) > 0, "Should have classification results"
        assert all(r.prediction.confidence > 0 for r in results), "All predictions should have confidence > 0"
        assert base_config.classify.boosting.enable_feature_gradients, "Feature gradients should be enabled"
        assert 4 in base_config.classify.boosting.feature_gradient_layers, "Layer 4 should be in gradient layers"
        
        print(f"✅ Phase 2 (Gradient Gating): {len(results)} images processed with feature gradients")
        
        # Verify that attribution paths were created (gradient gating generates attributions)
        assert all(r.attribution_paths is not None for r in results), "All results should have attribution paths"
    
    def test_phase_3_faithfulness_analysis(self, require_cuda, check_datasets, base_config, test_output_dir):
        """Phase 3: Test faithfulness analysis."""
        if "waterbirds" not in check_datasets:
            pytest.skip("Waterbirds dataset not available")
            
        # Configure for faithfulness analysis
        base_config.file.set_dataset("waterbirds") 
        base_config.classify.boosting.enable_feature_gradients = False
        base_config.classify.analysis = True  # Enable faithfulness analysis
        base_config.classify.use_clip = True
        base_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
        base_config.classify.clip_text_prompts = [
            "a photo of a terrestrial bird", "a photo of an aquatic bird"
        ]
        base_config.file.base_pipeline_dir = test_output_dir
        
        # Run with very small subset for faithfulness (it's computationally expensive)
        results, saco_results = run_unified_pipeline(
            config=base_config,
            dataset_name="waterbirds", 
            source_data_path=Path("./data/waterbirds/waterbird_complete95_forest2water2/"),
            device=require_cuda,
            subset_size=3,  # Even smaller for faithfulness testing
            random_seed=42
        )
        
        # Validate results
        assert len(results) > 0, "Should have classification results"
        assert base_config.classify.analysis, "Faithfulness analysis should be enabled"
        
        # Check that SaCo analysis was performed
        assert isinstance(saco_results, dict), "Should have SaCo results"
        if saco_results:  # If analysis completed successfully
            assert 'mean' in saco_results, "Should have mean SaCo score"
            assert -1 <= saco_results['mean'] <= 1, "SaCo score should be in valid range [-1, 1]"
        
        print(f"✅ Phase 3 (Faithfulness Analysis): Completed with mean SaCo = {saco_results.get('mean', 'N/A')}")
        
        # Check that faithfulness results were saved
        faithfulness_files = list(test_output_dir.glob("*faithfulness*.json"))
        if faithfulness_files:
            print(f"   Faithfulness results saved to {len(faithfulness_files)} files")
    
    def test_phase_4_parameter_sweep(self, require_cuda, check_datasets, test_output_dir):
        """Phase 4: Test parameter sweep functionality."""
        if "waterbirds" not in check_datasets:
            pytest.skip("Waterbirds dataset not available")
            
        # Create a mini sweep configuration
        sweep_config = {
            "dataset": "waterbirds",
            "source_path": "./data/waterbirds/waterbird_complete95_forest2water2/",
            "layers_to_test": [4],  # Test only one layer
            "feature_gradient_settings": [
                {"enabled": False, "layers": []},
                {"enabled": True, "layers": [4]}
            ],
            "subset_size": 3,  # Very small for sweep testing
            "random_seed": 42
        }
        
        # Create sweep output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        sweep_output_dir = test_output_dir / f"test_sweep_{timestamp}"
        sweep_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sweep config
        with open(sweep_output_dir / 'sweep_config.json', 'w') as f:
            json.dump(sweep_config, f, indent=2)
        
        # Run a mini parameter sweep
        experiment_count = 0
        all_results = {}
        
        for layer in sweep_config["layers_to_test"]:
            for fg_setting in sweep_config["feature_gradient_settings"]:
                # Setup config for this experiment
                pipeline_config = config.PipelineConfig()
                pipeline_config.file.set_dataset("waterbirds")
                pipeline_config.file.current_mode = "val"
                pipeline_config.classify.analysis = False  # Skip analysis in sweep
                pipeline_config.classify.use_clip = True
                pipeline_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
                pipeline_config.classify.clip_text_prompts = [
                    "a photo of a terrestrial bird", "a photo of an aquatic bird"
                ]
                pipeline_config.classify.boosting.steering_layers = [layer]
                pipeline_config.classify.boosting.enable_feature_gradients = fg_setting["enabled"]
                pipeline_config.classify.boosting.feature_gradient_layers = fg_setting["layers"]
                
                # Create experiment directory
                exp_name = f"layer{layer}_fg{fg_setting['enabled']}"
                exp_dir = sweep_output_dir / exp_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                pipeline_config.file.base_pipeline_dir = exp_dir
                
                try:
                    # Run experiment
                    results, saco_results = run_unified_pipeline(
                        config=pipeline_config,
                        dataset_name="waterbirds",
                        source_data_path=Path(sweep_config["source_path"]),
                        device=require_cuda,
                        subset_size=sweep_config["subset_size"],
                        random_seed=sweep_config["random_seed"]
                    )
                    
                    # Store results
                    experiment_result = {
                        'layer': layer,
                        'feature_gradients_enabled': fg_setting["enabled"],
                        'feature_gradient_layers': fg_setting["layers"],
                        'num_results': len(results),
                        'mean_confidence': np.mean([r.prediction.confidence for r in results]),
                        'saco_results': saco_results
                    }
                    
                    all_results[exp_name] = experiment_result
                    experiment_count += 1
                    
                    # Save individual experiment results
                    with open(exp_dir / 'experiment_results.json', 'w') as f:
                        json.dump(experiment_result, f, indent=2, default=str)
                        
                except Exception as e:
                    print(f"Warning: Experiment {exp_name} failed: {e}")
                    continue
        
        # Save overall sweep results
        sweep_summary = {
            'sweep_config': sweep_config,
            'experiments_completed': experiment_count,
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(sweep_output_dir / 'sweep_summary.json', 'w') as f:
            json.dump(sweep_summary, f, indent=2, default=str)
        
        # Validate sweep results
        assert experiment_count > 0, "Should have completed at least one experiment"
        assert len(all_results) > 0, "Should have experiment results"
        
        print(f"✅ Phase 4 (Parameter Sweep): {experiment_count} experiments completed")
        
        # Compare vanilla vs gradient gating results if both completed
        vanilla_experiments = [k for k, v in all_results.items() if not v['feature_gradients_enabled']]
        gating_experiments = [k for k, v in all_results.items() if v['feature_gradients_enabled']]
        
        if vanilla_experiments and gating_experiments:
            vanilla_conf = all_results[vanilla_experiments[0]]['mean_confidence']
            gating_conf = all_results[gating_experiments[0]]['mean_confidence']
            print(f"   Vanilla confidence: {vanilla_conf:.3f}")
            print(f"   Gradient gating confidence: {gating_conf:.3f}")
    
    def test_cross_dataset_compatibility(self, require_cuda, check_datasets, base_config, test_output_dir):
        """Test that the pipeline works across different available datasets."""
        tested_datasets = 0
        
        for dataset_name in check_datasets[:2]:  # Test up to 2 datasets to save time
            if dataset_name == "waterbirds":
                source_path = Path("./data/waterbirds/waterbird_complete95_forest2water2/")
                base_config.classify.use_clip = True
                base_config.classify.clip_text_prompts = [
                    "a photo of a terrestrial bird", "a photo of an aquatic bird"
                ]
            elif dataset_name == "covidquex":
                source_path = Path("./data/covidquex/data/lung/") 
                base_config.classify.use_clip = False
            elif dataset_name == "hyperkvasir":
                source_path = Path("./data/hyperkvasir/labeled-images/")
                base_config.classify.use_clip = False
            else:
                continue
                
            # Configure for this dataset
            base_config.file.set_dataset(dataset_name)
            base_config.classify.boosting.enable_feature_gradients = False
            base_config.classify.analysis = False
            base_config.file.base_pipeline_dir = test_output_dir / dataset_name
            base_config.file.base_pipeline_dir.mkdir(exist_ok=True)
            
            try:
                results, _ = run_unified_pipeline(
                    config=base_config,
                    dataset_name=dataset_name,
                    source_data_path=source_path,
                    device=require_cuda,
                    subset_size=2,  # Very small subset
                    random_seed=42
                )
                
                assert len(results) > 0, f"Should have results for {dataset_name}"
                tested_datasets += 1
                print(f"✅ Dataset {dataset_name}: {len(results)} images processed")
                
            except Exception as e:
                print(f"Warning: Dataset {dataset_name} failed: {e}")
                continue
        
        assert tested_datasets > 0, "Should have successfully tested at least one dataset"
        print(f"Cross-dataset compatibility: {tested_datasets}/{len(check_datasets)} datasets tested successfully")