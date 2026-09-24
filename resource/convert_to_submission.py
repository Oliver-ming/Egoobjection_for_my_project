#!/usr/bin/env python3
"""
Convert MMDetection test.bbox.json to Kaggle submission format.
Usage:
   # Recommended: use the default data/test.json file to ensure that
   # all test images are included.
   python convert_to_submission.py \
       --pred mm/work_dirs/deformable-detr_r50/test.bbox.json \
       --output submission.csv
   # Explicitly specify the test.json path.
   python convert_to_submission.py \
       --pred mm/work_dirs/deformable-detr_r50/test.bbox.json \
       --test data/test.json \
       --output submission.csv
   # Use a text file containing image IDs.
   python convert_to_submission.py \
       --pred mm/work_dirs/deformable-detr_r50/test.bbox.json \
       --images test_images.txt \
       --output submission.csv
   # Auto-detect image IDs from predictions.
   # NOT recommended: this may omit images with no predictions.
   python convert_to_submission.py \
       --pred mm/work_dirs/deformable-detr_r50/test.bbox.json \
       --no-test \
       --output submission.csv
Input format (test.bbox.json):
   [
       {
           "image_id": 123,
           "category_id": 0,
           "bbox": [x, y, w, h],
           "score": 0.95
       }
   ]
Output format (submission.csv):
   id,predictions
   123,"[{\"image_id\": 123, \"category_id\": 0, \"bbox\": [x, y, w, h], \"score\": 0.95}]"
   456,"[]"
"""
import argparse
import json
import os
from collections import defaultdict
import pandas as pd
def convert_to_submission(
   pred_file,
   output_file,
   image_ids=None,
   test_json=None,
):
   """Convert MMDetection predictions to Kaggle submission format."""
   # Load predictions.
   with open(pred_file, "r") as file:
       predictions = json.load(file)
   print(f"Loaded {len(predictions)} predictions from {pred_file}")
   # Group predictions by image ID.
   pred_by_img = defaultdict(list)
   for prediction in predictions:
       pred_by_img[prediction["image_id"]].append(prediction)
   # Get all image IDs.
   if image_ids is None:
       if test_json:
           # Load IDs from test.json to ensure that all images are included.
           with open(test_json, "r") as file:
               test_data = json.load(file)
           image_ids = [image["id"] for image in test_data["images"]]
           print(f"Loaded {len(image_ids)} image IDs from {test_json}")
       else:
           # Fallback: use only images that have predictions.
           image_ids = sorted(pred_by_img.keys())
           print(
               "WARNING: Using only images with predictions. "
               "This may omit some images!"
           )
   print(f"Processing {len(image_ids)} unique images")
   # Create submission rows.
   rows = []
   images_with_preds = 0
   images_without_preds = 0
   for image_id in image_ids:
       image_predictions = pred_by_img.get(image_id, [])
       if image_predictions:
           images_with_preds += 1
       else:
           images_without_preds += 1
       rows.append(
           {
               "id": image_id,
               "predictions": json.dumps(image_predictions),
           }
       )
   # Save the submission as a CSV file.
   submission_df = pd.DataFrame(rows)
   submission_df.to_csv(output_file, index=False)
   total_predictions = sum(
       len(pred_by_img[image_id]) for image_id in image_ids
   )
   print(f"\n✓ Saved submission to {output_file}")
   print(f"  - {len(rows)} rows (one per image)")
   print(f"  - {images_with_preds} images with predictions")
   print(f"  - {images_without_preds} images with empty predictions")
   print(f"  - {total_predictions} total predictions")
   return submission_df
def load_image_ids(image_list_file):
   """Load image IDs from a text file containing one ID per line."""
   with open(image_list_file, "r") as file:
       return [
           int(line.strip())
           for line in file
           if line.strip()
       ]
def main():
   parser = argparse.ArgumentParser(
       description=(
           "Convert MMDetection predictions to Kaggle submission format"
       ),
       formatter_class=argparse.RawDescriptionHelpFormatter,
       epilog=__doc__,
   )
   parser.add_argument(
       "--pred",
       required=True,
       help="Path to test.bbox.json",
   )
   parser.add_argument(
       "--output",
       default="submission.csv",
       help="Output CSV file",
   )
   parser.add_argument(
       "--test",
       default="data/test.json",
       help="Path to test.json (default: data/test.json)",
   )
   parser.add_argument(
       "--no-test",
       action="store_true",
       help=(
           "Do not use test.json. This may omit images without predictions."
       ),
   )
   parser.add_argument(
       "--images",
       help="Optional text file containing one image ID per line",
   )
   args = parser.parse_args()
   # Determine the test.json path.
   test_json = None
   if not args.no_test:
       if args.images:
           # Use the provided image list instead of test.json.
           test_json = None
       elif os.path.exists(args.test):
           test_json = args.test
       else:
           print(f"WARNING: test.json not found at {args.test}")
           print(
               "Only images with predictions will be used. "
               "Use --no-test to suppress this warning."
           )
   # Load image IDs if an image list is provided.
   image_ids = None
   if args.images:
       image_ids = load_image_ids(args.images)
       print(f"Loaded {len(image_ids)} image IDs from {args.images}")
   convert_to_submission(
       args.pred,
       args.output,
       image_ids,
       test_json,
   )
   print(f"\n✓ Done! You can now submit {args.output} to Kaggle")
if __name__ == "__main__":
   main()
