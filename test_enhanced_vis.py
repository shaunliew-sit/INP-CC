#!/usr/bin/env python3
"""
Test script for INP-CC Enhanced Visualization
Tests the new visualization system with sample data
"""

import torch
import numpy as np
from PIL import Image
import argparse
import os
from pathlib import Path

# Import INP-CC modules
from utils.enhanced_visualizer import INPCCVisualizationEngine


def create_mock_data():
    """Create mock data that resembles INP-CC outputs for testing"""
    
    # Mock image (224x224x3)
    mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Mock INP-CC model outputs
    batch_size = 2
    num_queries = 100
    num_classes = 600  # HICO classes
    
    outputs = {
        "logits_per_hoi": torch.randn(batch_size, num_queries, num_classes).softmax(dim=-1),
        "pred_boxes": torch.rand(batch_size, num_queries, 8),  # human + object boxes (cxcywh format)
        "box_scores": torch.rand(batch_size, num_queries).sigmoid(),
    }
    
    # Mock targets (ground truth)
    targets = []
    for i in range(batch_size):
        target = {
            'image_id': torch.tensor(i + 1),
            'orig_size': torch.tensor([224, 224]),
            'boxes': torch.tensor([
                [0.3, 0.3, 0.7, 0.7],  # Human box (normalized xyxy)
                [0.5, 0.5, 0.8, 0.8],  # Object box
            ]),
            'classes': torch.tensor([1, 15]),  # Person and object class
            'hois': [
                {
                    'hoi_id': 42,
                    'subject_id': 0,
                    'object_id': 1,
                }
            ]
        }
        targets.append(target)
    
    # Mock images tensor
    images_tensor = torch.from_numpy(mock_image).unsqueeze(0).repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2).float() / 255.0
    
    # Mock dataset handler
    class MockDatasetHandler:
        def __init__(self):
            self.dataset_texts = [
                ["ride", "bicycle"], ["hold", "cup"], ["eat", "pizza"], 
                ["play", "guitar"], ["read", "book"]
            ] * 120  # Extend to cover more HOI classes
        
        def interactions(self):
            return [{"action": "test", "object": "object", "interaction_id": i} for i in range(600)]
    
    return images_tensor, targets, outputs, MockDatasetHandler()


def test_visualization_engine():
    """Test the enhanced visualization engine"""
    print("🧪 Testing INP-CC Enhanced Visualization Engine...")
    
    # Create test directory
    test_output_dir = "./test_enhanced_vis_output"
    Path(test_output_dir).mkdir(exist_ok=True)
    
    # Create mock data
    images_tensor, targets, outputs, mock_dataset = create_mock_data()
    
    # Initialize visualization engine
    vis_engine = INPCCVisualizationEngine(
        output_dir=test_output_dir,
        dataset_handler=mock_dataset,
        max_images=5
    )
    
    print(f"✅ Visualization engine initialized")
    print(f"📁 Output directory: {test_output_dir}")
    
    # Test visualization
    try:
        image_ids = [1, 2]
        vis_engine.visualize_predictions(
            images=images_tensor,
            targets=targets,
            outputs=outputs,
            image_ids=image_ids,
            vis_threshold=0.1,
            max_detections=5
        )
        
        print("✅ Predictions visualization completed")
        
        # Generate summary
        summary = vis_engine.generate_summary()
        print("✅ Summary generated")
        
        # Check output files
        prediction_files = list(Path(test_output_dir, "predictions").glob("*.jpg"))
        analysis_files = list(Path(test_output_dir, "analysis").glob("*.json"))
        summary_files = list(Path(test_output_dir, "summary").glob("*.json"))
        
        print(f"\n📊 Test Results:")
        print(f"   📸 Prediction images: {len(prediction_files)}")
        print(f"   📋 Analysis files: {len(analysis_files)}")
        print(f"   📈 Summary files: {len(summary_files)}")
        
        if prediction_files:
            print(f"   🎨 Sample visualization: {prediction_files[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_args():
    """Test that arguments are properly integrated"""
    print("\n🔧 Testing argument integration...")
    
    try:
        from arguments import get_args_parser
        parser = get_args_parser()
        
        # Test enhanced visualization arguments
        test_args = [
            '--enhanced_vis',
            '--vis_max_images', '10',
            '--vis_threshold', '0.2',
            '--vis_max_detections', '5'
        ]
        
        args = parser.parse_args(test_args)
        
        assert hasattr(args, 'enhanced_vis')
        assert hasattr(args, 'vis_max_images')
        assert hasattr(args, 'vis_threshold')
        assert hasattr(args, 'vis_max_detections')
        
        assert args.enhanced_vis == True
        assert args.vis_max_images == 10
        assert args.vis_threshold == 0.2
        assert args.vis_max_detections == 5
        
        print("✅ All visualization arguments properly integrated")
        return True
        
    except Exception as e:
        print(f"❌ Error testing arguments: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 INP-CC Enhanced Visualization Test Suite")
    print("=" * 50)
    
    # Test 1: Argument integration
    arg_test_passed = test_integration_args()
    
    # Test 2: Visualization engine
    vis_test_passed = test_visualization_engine()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   🔧 Argument Integration: {'✅ PASSED' if arg_test_passed else '❌ FAILED'}")
    print(f"   🎨 Visualization Engine: {'✅ PASSED' if vis_test_passed else '❌ FAILED'}")
    
    if arg_test_passed and vis_test_passed:
        print("\n🎉 All tests passed! Enhanced visualization is ready for INP-CC.")
        print("\n📋 Usage Instructions:")
        print("   Add --enhanced_vis to your INP-CC evaluation command:")
        print("   python main.py --eval --enhanced_vis --vis_max_images 20 --pretrained <checkpoint>")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()