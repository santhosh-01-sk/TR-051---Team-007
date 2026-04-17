import torch
import numpy as np
from model import SiameseUNetDamageAssessor
from pipeline import DisasterAssessmentPipeline, calculate_iou

def test_model_shapes():
    print("Testing Architecture Forward Pass...")
    model = SiameseUNetDamageAssessor(in_channels=3, num_classes=4, num_disaster_types=10)
    
    # Batch of 2, 3 channels, 256x256
    pre_img = torch.randn(2, 3, 256, 256)
    post_img = torch.randn(2, 3, 256, 256)
    disaster_tag = torch.tensor([1, 5], dtype=torch.long)
    
    seg_logits, damage_logits = model(pre_img, post_img, disaster_tag)
    
    assert seg_logits.shape == (2, 1, 256, 256), f"Expected seg shape (2, 1, 256, 256), got {seg_logits.shape}"
    assert damage_logits.shape == (2, 4, 256, 256), f"Expected damage shape (2, 4, 256, 256), got {damage_logits.shape}"
    print("Model forward pass passed! Output shapes are correct.")

def test_iou_calculation():
    print("\nTesting IoU Calculation...")
    # Create dummy 5x5 masks
    # Both predict 0 (No damage) for most pixels
    true_mask = np.zeros((5, 5))
    pred_mask = np.zeros((5, 5))
    
    # Class 1 (Minor)
    true_mask[0:2, 0:2] = 1
    pred_mask[0:2, 0:2] = 1
    pred_mask[2, 2] = 1 # false positive for class 1
    
    # Class 3 (Destroyed)
    true_mask[3:5, 3:5] = 3
    pred_mask[3:5, 3:5] = 3
    
    iou_scores = calculate_iou(pred_mask, true_mask, num_classes=4)
    print(f"Calculated IoUs: {iou_scores}")
    
    # Minor damage IoU: Intersection is 4 pixels. Union is 5 pixels. IoU = 4/5 = 0.8
    assert np.isclose(iou_scores['Class_1_IoU'], 0.8), f"Expected 0.8, got {iou_scores['Class_1_IoU']}"
    # Destroyed damage IoU: Intersection is 4. Union is 4. IoU = 1.0
    assert np.isclose(iou_scores['Class_3_IoU'], 1.0), f"Expected 1.0, got {iou_scores['Class_3_IoU']}"
    # Major damage IoU: Neither mask has it, so it naturally handles with NaN logic (we used np.isnan to check it)
    assert np.isnan(iou_scores['Class_2_IoU']), "Class 2 should be NaN"
    print("IoU calculation passed!")

def test_pipeline_wrapper():
    print("\nTesting Inference Pipeline Wrapper...")
    pipeline = DisasterAssessmentPipeline(num_disaster_types=10, device='cpu')
    
    # Create dummy images
    pre_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    post_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    seg_map, damage_map, stats, geojson_str = pipeline.predict(pre_image, post_image, disaster_type_id=2)
    
    assert seg_map.shape == (256, 256), f"Expected seg_map shape (256, 256), got {seg_map.shape}"
    assert damage_map.shape == (256, 256), f"Expected damage_map shape (256, 256), got {damage_map.shape}"
    
    # Validate stats structure
    assert 'objects_counted' in stats, "Stats missing 'objects_counted'"
    assert 'impact_area_ha' in stats, "Stats missing 'impact_area_ha'"
    assert 'impact_area_sq_km' in stats, "Stats missing 'impact_area_sq_km'"
    for key in ['No Damage', 'Minor', 'Major', 'Destroyed']:
        assert key in stats['objects_counted'], f"Stats missing class '{key}'"
    
    # Validate GeoJSON
    import json
    geojson = json.loads(geojson_str)
    assert geojson['type'] == 'FeatureCollection', "GeoJSON type must be FeatureCollection"
    assert 'features' in geojson, "GeoJSON missing 'features'"
    
    print(f"Stats: {stats}")
    print(f"GeoJSON features count: {len(geojson['features'])}")
    print("Inference wrapper passed! Returning correctly shaped numpy masks with stats and GeoJSON.")

if __name__ == "__main__":
    test_model_shapes()
    test_iou_calculation()
    test_pipeline_wrapper()
    print("\nAll sanity checks passed successfully!")
