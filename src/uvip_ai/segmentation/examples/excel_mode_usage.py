"""
Excel-Aligned Segmentation Usage Example

Demonstrates how to use SegFormerB5 with Excel-compliant composite indices.
Features:
- 8 composite visual indices (matching Excel analysis format)
- Moving object filtering (pedestrians, vehicles excluded from static analysis)
- Road/sidewalk separation for accurate ground accessibility
- Heritage dominance placeholder for future enhancement
"""

from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uvip_ai.segmentation.segformer import SegformerB5


def main():
    """Example usage of Excel-aligned segmentation."""
    
    # Initialize model
    print("[Setup] Loading SegFormer-B5 model...")
    model = SegformerB5(
        model_id="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        device="cuda",  # Use "cpu" if no GPU available (slower)
        low_vram_mode=True  # Enable VRAM optimization
    )
    
    # Example image path (replace with your actual image)
    image_path = Path("data/test_image.jpg")
    
    if not image_path.exists():
        print(f"[Warning] Test image not found at {image_path}")
        print("Creating synthetic example output...")
        show_example_output()
        return
    
    # Run inference in Excel mode
    print(f"\n[Inference] Processing image: {image_path}")
    print("[Info] Mode: Excel-aligned with moving object filtering\n")
    
    result = model.infer(
        image=str(image_path),
        max_resolution=512,  # Lower = faster, Higher = more accurate
        excel_mode=True      # Enable Excel-compliant metrics
    )
    
    # Display results
    display_excel_results(result)
    
    # Cleanup
    model.free_memory()
    print("\n[Done] Model memory freed.")


def display_excel_results(result: dict):
    """Display Excel-aligned composite indices."""
    
    metrics = result["metrics"]
    pct_by_class = result["pct_by_class"]
    
    print("=" * 70)
    print("EXCEL-ALIGNED COMPOSITE VISUAL INDICES")
    print("=" * 70)
    
    # 8 Composite Indices
    indices = [
        ("Building Visibility Index", metrics.get("building_visibility_index")),
        ("Vegetation Coverage Index", metrics.get("vegetation_coverage_index")),
        ("Sky Openness Index", metrics.get("sky_openness_index")),
        ("Ground Accessibility Index", metrics.get("ground_accessibility_index")),
        ("Human Activity Index", metrics.get("human_activity_index")),
        ("Vehicle Intensity Index", metrics.get("vehicle_intensity_index")),
        ("Traffic Infrastructure Index", metrics.get("traffic_infrastructure_index")),
        ("Heritage Dominance Index", metrics.get("heritage_dominance_index")),
    ]
    
    print("\nComposite Visual Indices (normalized [0, 1]):")
    print("-" * 70)
    for name, value in indices:
        print(f"  {name:.<50} {value:.4f}")
    
    # Raw percentages (for reference)
    print("\n" + "=" * 70)
    print("RAW CLASS PERCENTAGES (Static Objects Only)")
    print("-" * 70)
    
    static_classes = ["building", "vegetation", "road", "sidewalk", "sky", 
                     "signage", "street_furniture", "heritage"]
    
    for class_name in static_classes:
        raw_pct = pct_by_class.get(class_name, 0)
        normalized = metrics.get(f"_raw_percentages", {}).get(class_name, 0)
        print(f"  {class_name:.<45} {raw_pct:>6.2f}% (norm: {normalized:.4f})")
    
    # Moving objects metadata
    print("\n" + "=" * 70)
    print("MOVING OBJECTS (Filtered from Static Analysis)")
    print("-" * 70)
    print(f"  Pedestrian percentage: {pct_by_class.get('pedestrian', 0):>6.2f}%")
    print(f"  Vehicle percentage: {pct_by_class.get('vehicle', 0):>6.2f}%")
    print(f"  Total moving objects: {metrics.get('_moving_object_percentage', 0):.2f}%")
    print(f"  Normalization factor applied: {metrics.get('_static_normalization_factor', 1):.4f}x")
    
    # Output format compatible with Excel
    print("\n" + "=" * 70)
    print("JSON OUTPUT (Ready for CSV/DB import)")
    print("-" * 70)
    print(json.dumps({
        "indices": {k: v for k, v in indices},
        "raw_percentages": metrics.get("_raw_percentages", {}),
        "metadata": {
            "moving_object_percentage": metrics.get("_moving_object_percentage"),
            "normalization_factor": metrics.get("_static_normalization_factor")
        }
    }, indent=2))
    
    print("=" * 70)


def show_example_output():
    """Show example output structure when no test image available."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE EXCEL-ALIGNED OUTPUT STRUCTURE")
    print("=" * 70)
    
    example_result = {
        "indices": {
            "building_visibility_index": 0.146,
            "vegetation_coverage_index": 0.536,
            "sky_openness_index": 0.235,
            "ground_accessibility_index": 0.343,
            "human_activity_index": 0.013,
            "vehicle_intensity_index": 0.161,
            "traffic_infrastructure_index": 0.013,
            "heritage_dominance_index": 0.532,
        },
        "raw_percentages": {
            "building": 5.04,
            "vegetation": 56.61,
            "road": 15.23,
            "sidewalk": 6.12,
            "sky": 9.68,
            "signage": 1.34,
            "street_furniture": 2.08,
            "heritage": 4.70,
        },
        "moving_objects": {
            "pedestrian": 1.34,
            "vehicle": 16.14,
            "total_moving_percentage": 17.48,
        },
        "metadata": {
            "normalization_factor": 1.2065,
            "excel_compliant": True,
            "moving_objects_filtered": True,
            "road_sidewalk_separated": True,
        }
    }
    
    print(json.dumps(example_result, indent=2))
    
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
    Building Visibility Index (0.146):
      - Low visibility = mostly open space or vegetation
      - > 0.5 = heavily built environment
      
    Vegetation Coverage Index (0.536):
      - High vegetation density
      - Good for green infrastructure assessment
      
    Ground Accessibility Index (0.343):
      - Combined road + sidewalk accessibility
      - Higher = better pedestrian/vehicle movement
      
    Heritage Dominance Index (0.532):
      - Current: Placeholder (requires custom CNN training)
      - Future: Detect historical/architectural heritage buildings
      - > 0.5 = significant heritage architecture presence
    """)
    
    print("=" * 70)


if __name__ == "__main__":
    main()
