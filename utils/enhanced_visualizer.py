"""
Enhanced HOI Visualization Engine for INP-CC
Provides comprehensive prediction vs ground truth analysis with side-by-side comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import torchvision.transforms as transforms
import utils.box_ops as box_ops


class INPCCVisualizationEngine:
    """Enhanced visualization engine for INP-CC HOI detection analysis"""
    
    def __init__(self, output_dir: str, dataset_handler=None, max_images: int = 50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_handler = dataset_handler
        self.max_images = max_images
        
        # Create subdirectories
        self.prediction_dir = self.output_dir / "predictions"
        self.analysis_dir = self.output_dir / "analysis"
        self.summary_dir = self.output_dir / "summary"
        
        for dir_path in [self.prediction_dir, self.analysis_dir, self.summary_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Font handling with fallbacks
        self.font_large = self._load_font(size=24)
        self.font_medium = self._load_font(size=18)
        self.font_small = self._load_font(size=14)
        
        # Color schemes for visualization
        self.colors = {
            'human_pred': (0, 150, 255),      # Blue for predicted human
            'object_pred': (255, 100, 0),    # Orange for predicted object  
            'human_gt': (0, 255, 0),         # Green for ground truth human
            'object_gt': (255, 0, 0),        # Red for ground truth object
            'background': (248, 248, 248),   # Light gray background
            'text_success': (0, 128, 0),     # Green for correct predictions
            'text_error': (200, 0, 0),       # Red for incorrect predictions
            'text_info': (50, 50, 50),       # Dark gray for info text
        }
        
        # Analysis statistics
        self.stats = {
            'total_images': 0,
            'total_predictions': 0,
            'total_ground_truths': 0,
            'correct_predictions': 0,
            'hoi_class_distribution': {},
            'confidence_distribution': [],
            'detection_errors': {
                'classification': 0,
                'localization': 0,
                'missing': 0,
                'false_positive': 0
            }
        }
    
    def _load_font(self, size: int = 16) -> ImageFont.ImageFont:
        """Load font with multiple fallback options"""
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, size)
            except Exception:
                continue
        
        try:
            return ImageFont.load_default()
        except Exception:
            return None
    
    def _draw_box(self, draw: ImageDraw.Draw, bbox: List[float], color: Tuple[int, int, int], 
                  width: int = 3, label: str = "", font: ImageFont.ImageFont = None):
        """Draw bounding box with label"""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Safety check: ensure valid coordinates
        if x2 <= x1 or y2 <= y1:
            print(f"   ⚠️  Skipping invalid box: [{x1}, {y1}, {x2}, {y2}]")
            return
        
        # Additional safety: ensure reasonable box size
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            print(f"   ⚠️  Skipping tiny box: [{x1}, {y1}, {x2}, {y2}]")
            return
        
        # Draw rectangle
        try:
            for i in range(width):
                draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color, fill=None)
        except Exception as e:
            print(f"   ⚠️  Error drawing box [{x1}, {y1}, {x2}, {y2}]: {e}")
            return
        
        # Draw label if provided
        if label and font:
            try:
                # Get text size using textbbox (newer PIL)
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                # Fallback for older PIL versions
                text_width, text_height = draw.textsize(label, font=font)
            
            # Draw background rectangle for text
            text_bg = [x1, y1 - text_height - 4, x1 + text_width + 8, y1]
            draw.rectangle(text_bg, fill=color)
            draw.text((x1 + 4, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
    
    def _safe_text_draw(self, draw: ImageDraw.Draw, position: Tuple[int, int], 
                       text: str, fill: Tuple[int, int, int], font: ImageFont.ImageFont = None):
        """Safely draw text with font fallback"""
        try:
            if font:
                draw.text(position, text, fill=fill, font=font)
            else:
                draw.text(position, text, fill=fill)
        except Exception:
            # Fallback without font
            draw.text(position, text, fill=fill)
    
    def visualize_predictions(self, images: torch.Tensor, targets: List[Dict], 
                            outputs: Dict[str, torch.Tensor], image_ids: List[int],
                            vis_threshold: float = 0.1, max_detections: int = 10):
        """
        Create comprehensive prediction visualizations with ground truth comparison
        """
        try:
            if not hasattr(images, 'tensors'):
                # Handle case where images is already a tensor
                image_tensors = images
                image_masks = None
            else:
                image_tensors = images.tensors
                image_masks = images.mask
            
            batch_size = image_tensors.shape[0]
            
            # Debug info
            print(f"🎨 Processing batch: {batch_size} images, tensor shape: {image_tensors.shape}")
            
            # Convert tensors to numpy arrays
            vis_images = image_tensors.permute(0, 2, 3, 1).detach().cpu().numpy()
            
            for batch_idx in range(min(batch_size, self.max_images - self.stats['total_images'])):
                if self.stats['total_images'] >= self.max_images:
                    break
                    
                try:
                    image_id = image_ids[batch_idx] if image_ids else batch_idx
                    print(f"🖼️  Processing image {image_id} (batch {batch_idx})")
                    
                    self._process_single_image(
                        vis_images[batch_idx], 
                        targets[batch_idx], 
                        outputs, 
                        batch_idx,
                        image_id,
                        image_masks[batch_idx] if image_masks is not None else None,
                        vis_threshold,
                        max_detections
                    )
                    
                    self.stats['total_images'] += 1
                    print(f"✅ Successfully processed image {image_id}")
                    
                except Exception as e:
                    print(f"⚠️ Error processing image {image_id}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Critical error in visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_single_image(self, image_array: np.ndarray, target: Dict, outputs: Dict,
                            batch_idx: int, image_id: int, image_mask: Optional[torch.Tensor],
                            vis_threshold: float, max_detections: int):
        """Process and visualize a single image with predictions and ground truth"""
        
        # Properly extract and normalize image for display
        img_rgb = image_array.copy()
        
        print(f"   🖼️  Image {image_id} processing started:")
        print(f"      📊 Raw input: shape {img_rgb.shape}, dtype {img_rgb.dtype}, range [{img_rgb.min():.3f}-{img_rgb.max():.3f}]")
        
        # Handle different input formats
        if len(img_rgb.shape) == 4:  # Remove batch dimension if present
            img_rgb = img_rgb[0]
            print(f"      📦 Removed batch dimension: {img_rgb.shape}")
        
        # Convert from tensor to numpy if needed
        if hasattr(img_rgb, 'detach'):
            img_rgb = img_rgb.detach().cpu().numpy()
        
        # Ensure RGB format (H, W, 3) - INP-CC uses CHW format
        if len(img_rgb.shape) == 3 and img_rgb.shape[0] == 3:  # CHW format (3, H, W)
            img_rgb = img_rgb.transpose(1, 2, 0)  # Convert to HWC (H, W, 3)
            print(f"      🔄 Transposed CHW->HWC: {img_rgb.shape}")
        elif len(img_rgb.shape) == 2:  # Grayscale
            img_rgb = np.stack([img_rgb, img_rgb, img_rgb], axis=-1)
            print(f"      🔳 Converted grayscale to RGB: {img_rgb.shape}")
        
        # Handle INP-CC CLIP normalization properly
        if img_rgb.dtype == np.float32 or img_rgb.dtype == np.float64:
            print(f"      🔢 Processing float image, range: [{img_rgb.min():.3f}-{img_rgb.max():.3f}]")
            
            # CLIP normalization constants (RGB order) - exactly matching CLIP preprocessing
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            
            # Reshape for broadcasting if needed
            if len(img_rgb.shape) == 3:
                mean = mean.reshape(1, 1, 3)
                std = std.reshape(1, 1, 3)
            
            # Check if this looks like CLIP-normalized data 
            # CLIP normalization produces values roughly in [-2.8, 2.6] range
            if img_rgb.min() >= -4.0 and img_rgb.max() <= 4.0:
                print(f"      ✅ Detected CLIP normalization, applying denormalization")
                # Denormalize: (normalized * std) + mean
                # This should bring values back to [0, 1] range
                img_rgb = (img_rgb * std) + mean
                print(f"      🎯 After denormalization: range [{img_rgb.min():.3f}-{img_rgb.max():.3f}]")
                
                # Clamp to [0, 1] and scale to [0, 255]
                img_rgb = np.clip(img_rgb, 0.0, 1.0)
                img_rgb = (img_rgb * 255.0).astype(np.uint8)
                print(f"      🎨 Final scaling to uint8: range [{img_rgb.min()}-{img_rgb.max()}]")
                
            elif img_rgb.min() >= -0.5 and img_rgb.max() <= 1.5:
                print(f"      ✅ Detected [0,1] range, scaling to [0,255]")
                # Values likely already in [0,1] range, just scale
                img_rgb = np.clip(img_rgb, 0.0, 1.0)
                img_rgb = (img_rgb * 255.0).astype(np.uint8)
                
            else:
                print(f"      ⚠️  Unknown normalization, applying min-max scaling")
                # Unknown normalization, normalize to [0,1] then scale
                img_min, img_max = img_rgb.min(), img_rgb.max()
                if img_max > img_min:
                    img_rgb = (img_rgb - img_min) / (img_max - img_min)
                else:
                    img_rgb = np.zeros_like(img_rgb)
                img_rgb = (img_rgb * 255.0).astype(np.uint8)
                print(f"      🔧 Min-max normalized: [{img_min:.3f}-{img_max:.3f}] -> [0-255]")
                
        else:
            print(f"      🔢 Processing integer image: {img_rgb.dtype}")
            # Already integer type, ensure proper range and uint8
            if img_rgb.max() <= 1:
                # Likely in [0,1] range, scale up
                img_rgb = (img_rgb * 255.0).astype(np.uint8)
            else:
                # Likely already in [0,255] range
                img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        
        # Final validation and range clamping
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        
        # Enhanced debug output
        print(f"      ✅ Final image: shape {img_rgb.shape}, dtype {img_rgb.dtype}, range [{img_rgb.min()}-{img_rgb.max()}]")
        if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
            r_stats = f"R[{img_rgb[:,:,0].min()}-{img_rgb[:,:,0].max()}]"
            g_stats = f"G[{img_rgb[:,:,1].min()}-{img_rgb[:,:,1].max()}]"  
            b_stats = f"B[{img_rgb[:,:,2].min()}-{img_rgb[:,:,2].max()}]"
            print(f"      🌈 RGB channels: {r_stats} {g_stats} {b_stats}")
            
            # Check for potential issues
            if img_rgb[:,:,0].max() - img_rgb[:,:,0].min() < 10:
                print(f"      ⚠️  Warning: Low R channel variation")
            if img_rgb[:,:,1].max() - img_rgb[:,:,1].min() < 10:
                print(f"      ⚠️  Warning: Low G channel variation")
            if img_rgb[:,:,2].max() - img_rgb[:,:,2].min() < 10:
                print(f"      ⚠️  Warning: Low B channel variation")
        
        # Ensure minimum size
        if img_rgb.shape[0] < 10 or img_rgb.shape[1] < 10:
            print(f"Warning: Image {image_id} too small ({img_rgb.shape}), skipping visualization")
            return
        
        # Get original dimensions
        if image_mask is not None:
            ori_h = int(torch.sum(~image_mask[:, 0]))
            ori_w = int(torch.sum(~image_mask[0, :]))
        else:
            ori_h, ori_w = img_rgb.shape[:2]
        
        # Extract predictions
        predictions = self._extract_predictions(outputs, batch_idx, ori_w, ori_h, vis_threshold)
        ground_truths = self._extract_ground_truths(target, ori_w, ori_h)
        
        # Create side-by-side visualization
        self._create_comparison_visualization(
            img_rgb, predictions, ground_truths, image_id, ori_w, ori_h
        )
        
        # Update statistics
        self._update_statistics(predictions, ground_truths)
    
    def _extract_predictions(self, outputs: Dict, batch_idx: int, ori_w: int, ori_h: int,
                           vis_threshold: float) -> List[Dict]:
        """Extract and process predictions from model outputs"""
        predictions = []
        
        try:
            # Get prediction scores and boxes (following INP-CC format)
            if "logits_per_hoi" in outputs:
                hoi_logits = outputs["logits_per_hoi"][batch_idx].detach().cpu()  # [num_queries, num_classes]
                hoi_scores = hoi_logits.softmax(dim=-1)
                
                if "box_scores" in outputs:
                    box_scores = outputs["box_scores"][batch_idx].sigmoid().detach().cpu()  # [num_queries, 1] or [num_queries]
                    # Handle extra dimension in box_scores
                    if len(box_scores.shape) > 1:
                        box_scores = box_scores.squeeze(-1)  # [num_queries]
                    # Get the max HOI class score for each query
                    max_hoi_scores, predicted_classes = hoi_scores.max(dim=-1)  # [num_queries]
                    scores = max_hoi_scores * box_scores  # Combine HOI and detection scores
                else:
                    max_hoi_scores, predicted_classes = hoi_scores.max(dim=-1)  # [num_queries]
                    scores = max_hoi_scores
                    
                # Ensure tensors are 1D
                scores = scores.flatten()
                predicted_classes = predicted_classes.flatten()
                
            else:
                # Fallback for different output formats
                scores = outputs.get("scores", outputs.get("pred_logits", torch.tensor([])))[batch_idx]
                if len(scores.shape) > 1:
                    scores, predicted_classes = scores.softmax(dim=-1).max(dim=-1)
                else:
                    predicted_classes = torch.zeros_like(scores, dtype=torch.long)
                scores = scores.detach().cpu().flatten()
                predicted_classes = predicted_classes.detach().cpu().flatten()
            
            boxes = outputs["pred_boxes"][batch_idx].detach().cpu()  # [num_queries, 8]
            
            print(f"   🔍 PREDICTION BOXES DEBUG:")
            print(f"      📦 Raw pred_boxes shape: {boxes.shape}")
            print(f"      📦 Raw pred_boxes (first 3): {boxes[:3]}")
            
            # Convert from cxcywh to xyxy and scale to original image size
            if boxes.shape[-1] == 8:  # human + object boxes
                human_boxes = box_ops.box_cxcywh_to_xyxy(boxes[:, :4])
                object_boxes = box_ops.box_cxcywh_to_xyxy(boxes[:, 4:])
                print(f"      🔄 After CXCYWH->XYXY conversion (first 3):")
                print(f"         👤 Human boxes: {human_boxes[:3]}")
                print(f"         📦 Object boxes: {object_boxes[:3]}")
            else:
                # Handle different box formats
                human_boxes = box_ops.box_cxcywh_to_xyxy(boxes[:, :4])  
                object_boxes = human_boxes  # Fallback
            
            print(f"      📐 Scaling by ori_w={ori_w}, ori_h={ori_h}")
            
            # Scale to original image dimensions
            human_boxes[:, [0, 2]] *= ori_w
            human_boxes[:, [1, 3]] *= ori_h  
            object_boxes[:, [0, 2]] *= ori_w
            object_boxes[:, [1, 3]] *= ori_h
            
            print(f"      ✅ After scaling (first 3):")
            print(f"         👤 Human boxes: {human_boxes[:3]}")
            print(f"         📦 Object boxes: {object_boxes[:3]}")
            
            # Validate and fix bounding boxes
            def fix_boxes(boxes):
                # Convert to numpy for easier manipulation
                boxes_np = boxes.numpy() if isinstance(boxes, torch.Tensor) else boxes
                
                # Ensure coordinates are within image bounds and valid
                for i in range(len(boxes_np)):
                    x1, y1, x2, y2 = boxes_np[i]
                    
                    # Clamp to image bounds
                    x1 = max(0, min(x1, ori_w - 2))
                    y1 = max(0, min(y1, ori_h - 2))
                    x2 = max(x1 + 2, min(x2, ori_w))
                    y2 = max(y1 + 2, min(y2, ori_h))
                    
                    # Ensure minimum box size and x2 > x1, y2 > y1
                    if x2 <= x1:
                        x2 = x1 + 2
                    if y2 <= y1:
                        y2 = y1 + 2
                    
                    boxes_np[i] = [x1, y1, x2, y2]
                
                return torch.from_numpy(boxes_np) if isinstance(boxes, torch.Tensor) else boxes_np
            
            human_boxes = fix_boxes(human_boxes)
            object_boxes = fix_boxes(object_boxes)
            
            # Debug confidence scores
            print(f"   🔍 All confidence scores (first 10): {scores[:10].tolist()}")
            print(f"   📊 Score range: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
            print(f"   🎯 Using threshold: {vis_threshold} (aligned with core inference)")
            
            # Apply threshold filtering to match core model inference pipeline
            valid_mask = scores > vis_threshold
            print(f"   ✅ Valid predictions: {valid_mask.sum()}/{len(scores)} above threshold")
            
            valid_scores = scores[valid_mask]
            valid_classes = predicted_classes[valid_mask]
            valid_human_boxes = human_boxes[valid_mask]
            valid_object_boxes = object_boxes[valid_mask]
            
            # Create prediction objects
            for i, (score, hclass, hbox, obox) in enumerate(
                zip(valid_scores, valid_classes, valid_human_boxes, valid_object_boxes)
            ):
                hoi_name = self._get_hoi_name(int(hclass))
                
                # Convert to lists for JSON serialization
                hbox_list = hbox.tolist() if hasattr(hbox, 'tolist') else list(hbox)
                obox_list = obox.tolist() if hasattr(obox, 'tolist') else list(obox)
                
                # Debug: print bbox coordinates for first few predictions
                if i < 3:
                    print(f"   📦 Pred {i}: human_bbox={hbox_list}, object_bbox={obox_list}")
                
                predictions.append({
                    'hoi_id': int(hclass),
                    'hoi_name': hoi_name,
                    'confidence': float(score),
                    'human_bbox': hbox_list,
                    'object_bbox': obox_list
                })
            
            # Sort by confidence - show ALL predictions exactly as model produces them
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Debug: Check predictions before and after filtering
            print(f"   📊 Before filtering: {len(predictions)} predictions")
            for i, pred in enumerate(predictions[:3]):
                print(f"      Pre-filter {i}: {pred['hoi_name']} confidence={pred['confidence']:.3f}")
            
            # Apply threshold filtering to match core inference pipeline
            if vis_threshold > 0:
                filtered_predictions = [p for p in predictions if p['confidence'] >= vis_threshold]
                print(f"   🎯 After threshold {vis_threshold}: {len(predictions)} -> {len(filtered_predictions)} predictions")
                predictions = filtered_predictions
            else:
                print(f"   ✅ No threshold applied: keeping all {len(predictions)} predictions")
            
            # Only limit by max_detections parameter if specified, otherwise show all
            if hasattr(self, 'max_detections') and self.max_detections > 0:
                predictions = predictions[:self.max_detections]
                print(f"   📊 Limited to max_detections: {len(predictions)} predictions shown")
            else:
                print(f"   📊 Showing ALL predictions: {len(predictions)} predictions")
                
            # Final debug
            print(f"   ✅ Final predictions to draw: {len(predictions)}")
            for i, pred in enumerate(predictions[:3]):
                print(f"      Final {i}: {pred['hoi_name']} confidence={pred['confidence']:.3f}")
            
        except Exception as e:
            print(f"Error extracting predictions: {e}")
            import traceback
            traceback.print_exc()
            
        return predictions
    
    def _extract_ground_truths(self, target: Dict, ori_w: int, ori_h: int) -> List[Dict]:
        """Extract ground truth annotations"""
        ground_truths = []
        
        print(f"   📍 Ground Truth Extraction:")
        print(f"      🔍 Target keys: {list(target.keys())}")
        print(f"      📐 Image size: {ori_w}x{ori_h}")
        
        try:
            # Check different possible keys for ground truth data
            hoi_annotations = None
            boxes = None
            
            # Try different ground truth data formats - Handle SWIG vs HICO differently
            dataset_type = "swig" if 'aux_classes' in target or 'mask_region_hw' in target else "hico"
            print(f"      🔍 Detected dataset type: {dataset_type.upper()}")
            
            if 'hois' in target:
                hoi_annotations = target['hois']
                boxes = target.get('boxes', torch.tensor([]))
                print(f"      ✅ Found 'hois' key with {len(hoi_annotations)} HOI annotations")
                print(f"      📦 Found 'boxes' with shape: {boxes.shape if hasattr(boxes, 'shape') else type(boxes)}")
                
                # SWIG-specific debugging
                if dataset_type == "swig":
                    print(f"      🔧 SWIG format detected - aux_classes: {'aux_classes' in target}")
                    if 'aux_classes' in target:
                        print(f"      🏷️  aux_classes shape: {target['aux_classes'].shape if hasattr(target['aux_classes'], 'shape') else 'N/A'}")
            elif 'annotations' in target:
                hoi_annotations = target['annotations']
                boxes = target.get('boxes', torch.tensor([]))
                print(f"      ✅ Found 'annotations' key with {len(hoi_annotations)} annotations")
            elif 'labels' in target and 'boxes' in target:
                # Fallback: try to construct HOI from labels and boxes
                labels = target['labels']
                boxes = target['boxes']
                print(f"      ✅ Found 'labels' and 'boxes', trying to construct HOIs")
                print(f"      🏷️  Labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}")
                print(f"      📦 Boxes shape: {boxes.shape if hasattr(boxes, 'shape') else type(boxes)}")
                
                # For this fallback, we might need different logic
                # This is a simplified approach - might need adjustment based on actual data format
                if hasattr(boxes, 'cpu'):
                    boxes = boxes.cpu()
                if hasattr(labels, 'cpu'):
                    labels = labels.cpu()
                    
                # Create pseudo-HOI annotations from available data
                hoi_annotations = []
                for i in range(len(boxes) if len(boxes) < len(labels) else len(labels)):
                    if i + 1 < len(boxes):  # Ensure we have at least 2 boxes for human-object pair
                        hoi_annotations.append({
                            'hoi_id': int(labels[i]) if hasattr(labels[i], 'item') else int(labels[i]),
                            'subject_id': i,      # Assume first box is human
                            'object_id': i + 1    # Assume next box is object
                        })
            else:
                print(f"      ❌ No recognized ground truth format found")
                return ground_truths
            
            if hoi_annotations is None or boxes is None or len(hoi_annotations) == 0:
                print(f"      ❌ No valid HOI annotations or boxes found")
                return ground_truths
                
            # Convert boxes to numpy if tensor
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            elif hasattr(boxes, 'numpy'):
                boxes = boxes.numpy()
                
            print(f"      📊 Processing {len(hoi_annotations)} HOI annotations with {len(boxes)} boxes")
            
            for i, hoi in enumerate(hoi_annotations):
                try:
                    # Get HOI information
                    hoi_id = hoi.get('hoi_id', 0) if isinstance(hoi, dict) else getattr(hoi, 'hoi_id', 0)
                    hoi_name = self._get_hoi_name(hoi_id)
                    
                    # Get bounding boxes indices
                    subject_id = hoi.get('subject_id', 0) if isinstance(hoi, dict) else getattr(hoi, 'subject_id', 0)
                    object_id = hoi.get('object_id', 0) if isinstance(hoi, dict) else getattr(hoi, 'object_id', 0)
                    
                    print(f"      🔄 HOI {i}: id={hoi_id}, subject_id={subject_id}, object_id={object_id}")
                    
                    if subject_id < len(boxes) and object_id < len(boxes) and subject_id != object_id:
                        human_box = boxes[subject_id]
                        object_box = boxes[object_id]
                        
                        # Convert to lists if needed
                        if hasattr(human_box, 'tolist'):
                            human_box = human_box.tolist()
                        if hasattr(object_box, 'tolist'):
                            object_box = object_box.tolist()
                        
                        print(f"         👤 Raw human box: {human_box}")
                        print(f"         📦 Raw object box: {object_box}")
                        
                        # USE ORIGINAL RAW COORDINATES - NO SCALING OR CONVERSION
                        if len(human_box) == 4 and len(object_box) == 4:
                            print(f"         🔧 USING ORIGINAL RAW COORDINATES:")
                            print(f"         👤 Original human box: {human_box}")
                            print(f"         📦 Original object box: {object_box}")
                            
                            # Convert to lists if needed, but NO coordinate transformations
                            if hasattr(human_box, 'tolist'):
                                human_box = human_box.tolist()
                            if hasattr(object_box, 'tolist'):
                                object_box = object_box.tolist()
                            
                            # Raw coordinates are normalized [0,1], need to scale to image size for visualization
                            # BUT preserve format detection (CXCYWH vs XYXY)
                            
                            # Check if these are CXCYWH format (dataset-specific handling)
                            if all(0 <= coord <= 1.0 for coord in human_box + object_box):
                                # Dataset-specific CXCYWH detection
                                if dataset_type == "swig":
                                    # SWIG tends to have different coordinate patterns
                                    human_is_cxcywh = (human_box[2] < 0.9 and human_box[3] < 0.9)
                                    object_is_cxcywh = (object_box[2] < 0.9 and object_box[3] < 0.9)
                                    print(f"         🔧 SWIG CXCYWH detection: human={human_is_cxcywh}, object={object_is_cxcywh}")
                                else:  # HICO
                                    # HICO detection logic
                                    human_is_cxcywh = (human_box[2] < 0.8 and human_box[3] < 0.8)
                                    object_is_cxcywh = (object_box[2] < 0.8 and object_box[3] < 0.8)
                                    print(f"         🔧 HICO CXCYWH detection: human={human_is_cxcywh}, object={object_is_cxcywh}")
                                
                                if human_is_cxcywh:
                                    # Convert CXCYWH -> XYXY for human box
                                    cx, cy, w, h = human_box
                                    human_box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                                    print(f"         🔄 Human CXCYWH->XYXY: {human_box}")
                                    
                                if object_is_cxcywh:
                                    # Convert CXCYWH -> XYXY for object box  
                                    cx, cy, w, h = object_box
                                    object_box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                                    print(f"         🔄 Object CXCYWH->XYXY: {object_box}")
                                
                                # Scale normalized coordinates to image size
                                human_box = [human_box[0] * ori_w, human_box[1] * ori_h,
                                           human_box[2] * ori_w, human_box[3] * ori_h]
                                object_box = [object_box[0] * ori_w, object_box[1] * ori_h,
                                            object_box[2] * ori_w, object_box[3] * ori_h]
                                
                                # Fix invalid boxes (ensure x1 <= x2, y1 <= y2)
                                def fix_invalid_box(box):
                                    x1, y1, x2, y2 = box
                                    if x1 > x2:
                                        x1, x2 = x2, x1  # Swap if x1 > x2
                                    if y1 > y2:
                                        y1, y2 = y2, y1  # Swap if y1 > y2
                                    # Ensure minimum box size
                                    if x2 - x1 < 2:
                                        x2 = x1 + 2
                                    if y2 - y1 < 2:
                                        y2 = y1 + 2
                                    return [x1, y1, x2, y2]
                                
                                human_box = fix_invalid_box(human_box)
                                object_box = fix_invalid_box(object_box)
                                            
                                print(f"         ✅ Final coordinates: human={[int(x) for x in human_box]}, object={[int(x) for x in object_box]}")
                            else:
                                print(f"         ✅ Using absolute coordinates: human={[int(x) for x in human_box]}, object={[int(x) for x in object_box]}")
                            
                            ground_truths.append({
                                'hoi_id': hoi_id,
                                'hoi_name': hoi_name,
                                'human_bbox': human_box,
                                'object_bbox': object_box
                            })
                        else:
                            print(f"         ❌ Invalid box format: human={len(human_box)}, object={len(object_box)}")
                    else:
                        print(f"         ❌ Invalid box indices: subject_id={subject_id}, object_id={object_id}, num_boxes={len(boxes)}")
                        
                except Exception as hoi_error:
                    print(f"      ❌ Error processing HOI {i}: {hoi_error}")
                    continue
        
        except Exception as e:
            print(f"❌ Error extracting ground truths: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"   ✅ Extracted {len(ground_truths)} ground truth HOI annotations")
        for i, gt in enumerate(ground_truths[:3]):  # Show first 3 for debugging
            print(f"      GT {i}: {gt['hoi_name']} - Human: {gt['human_bbox']} Object: {gt['object_bbox']}")
        
        return ground_truths
    
    def _boxes_overlap_significantly(self, pred1: Dict, pred2: Dict, threshold: float = 0.5) -> bool:
        """Check if two predictions have significantly overlapping bounding boxes"""
        def calculate_iou(box1, box2):
            """Calculate Intersection over Union (IoU) of two boxes"""
            # Get coordinates
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            # Calculate union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Check IoU for both human and object boxes
        human_iou = calculate_iou(pred1['human_bbox'], pred2['human_bbox'])
        object_iou = calculate_iou(pred1['object_bbox'], pred2['object_bbox'])
        
        # Consider overlapping if either human or object boxes have high IoU
        return human_iou > threshold or object_iou > threshold
    
    def _get_hoi_name(self, hoi_id: int) -> str:
        """Get human-readable HOI name from ID"""
        if hasattr(self.dataset_handler, 'dataset_texts') and hoi_id < len(self.dataset_handler.dataset_texts):
            text_pair = self.dataset_handler.dataset_texts[hoi_id]
            return f"{text_pair[0]} {text_pair[1]}"
        elif hasattr(self.dataset_handler, 'interactions') and hoi_id < len(self.dataset_handler.interactions):
            interaction = self.dataset_handler.interactions[hoi_id]
            return f"{interaction.get('action', 'unknown')} {interaction.get('object', 'unknown')}"
        else:
            return f"HOI_{hoi_id}"
    
    def _create_comparison_visualization(self, image: np.ndarray, predictions: List[Dict],
                                       ground_truths: List[Dict], image_id: int,
                                       ori_w: int, ori_h: int):
        """Create side-by-side comparison visualization"""
        
        # Create base images
        pred_img = Image.fromarray(image)
        gt_img = Image.fromarray(image)
        
        # Draw predictions
        pred_draw = ImageDraw.Draw(pred_img)
        self._draw_predictions(pred_draw, predictions)
        
        # Draw ground truths  
        gt_draw = ImageDraw.Draw(gt_img)
        self._draw_ground_truths(gt_draw, ground_truths)
        
        # Create combined image
        combined_width = ori_w * 2 + 60  # Space for margin
        combined_height = ori_h + 100    # Space for headers and text
        combined_img = Image.new('RGB', (combined_width, combined_height), self.colors['background'])
        
        # Add headers
        header_draw = ImageDraw.Draw(combined_img)
        self._safe_text_draw(header_draw, (20, 10), "Predictions", 
                           self.colors['text_info'], self.font_large)
        self._safe_text_draw(header_draw, (ori_w + 40, 10), "Ground Truth", 
                           self.colors['text_info'], self.font_large)
        
        # Paste images
        combined_img.paste(pred_img, (10, 40))
        combined_img.paste(gt_img, (ori_w + 30, 40))
        
        # Add analysis text
        self._add_analysis_text(header_draw, predictions, ground_truths, 
                              ori_h + 50, combined_width)
        
        # Save visualization
        output_path = self.prediction_dir / f"image_{image_id}_comparison.jpg"
        combined_img.save(output_path)
        
        # Create detailed individual views
        self._save_detailed_views(pred_img, gt_img, predictions, ground_truths, image_id)
    
    def _draw_predictions(self, draw: ImageDraw.Draw, predictions: List[Dict]):
        """Draw prediction boxes and labels - EXACTLY as provided, no filtering"""
        print(f"   🎨 Drawing {len(predictions)} prediction boxes (exact coordinates):")
        for i, pred in enumerate(predictions):  # Draw ALL predictions, no [:10] limit
            human_bbox = pred['human_bbox']
            object_bbox = pred['object_bbox']
            
            print(f"      Pred {i}: {pred['hoi_name']} ({pred['confidence']:.2f})")
            print(f"         👤 Drawing human box: {[int(x) for x in human_bbox]}")
            print(f"         📦 Drawing object box: {[int(x) for x in object_bbox]}")
            
            # Draw human box (blue) - exact coordinates
            self._draw_box(draw, human_bbox, self.colors['human_pred'], 
                         width=3, label=f"Human {pred['confidence']:.2f}", 
                         font=self.font_small)
            
            # Draw object box (orange) - exact coordinates  
            self._draw_box(draw, object_bbox, self.colors['object_pred'],
                         width=2, label=f"Object", font=self.font_small)
            
            # Add HOI label
            x1, y1 = human_bbox[:2]
            hoi_text = f"{pred['hoi_name']} ({pred['confidence']:.2f})"
            self._safe_text_draw(draw, (int(x1), int(y1) - 40), hoi_text,
                               self.colors['text_info'], self.font_medium)
    
    def _draw_ground_truths(self, draw: ImageDraw.Draw, ground_truths: List[Dict]):
        """Draw ground truth boxes and labels - EXACTLY as provided"""
        print(f"   🎯 Drawing {len(ground_truths)} ground truth boxes (exact coordinates):")
        for i, gt in enumerate(ground_truths):
            human_bbox = gt['human_bbox']
            object_bbox = gt['object_bbox']
            
            print(f"      GT {i}: {gt['hoi_name']}")
            print(f"         👤 Drawing GT human box: {[int(x) for x in human_bbox]}")
            print(f"         📦 Drawing GT object box: {[int(x) for x in object_bbox]}")
            
            # Draw human box (green) - exact coordinates
            self._draw_box(draw, human_bbox, self.colors['human_gt'],
                         width=3, label="GT Human", font=self.font_small)
            
            # Draw object box (red) - exact coordinates
            self._draw_box(draw, object_bbox, self.colors['object_gt'],
                         width=2, label="GT Object", font=self.font_small)
            
            # Add HOI label
            x1, y1 = human_bbox[:2] 
            self._safe_text_draw(draw, (int(x1), int(y1) - 40), gt['hoi_name'],
                               self.colors['text_success'], self.font_medium)
    
    def _add_analysis_text(self, draw: ImageDraw.Draw, predictions: List[Dict],
                         ground_truths: List[Dict], y_start: int, width: int):
        """Add analysis summary text below images"""
        analysis_lines = [
            f"Predictions: {len(predictions)} | Ground Truth: {len(ground_truths)}",
            f"Top Prediction: {predictions[0]['hoi_name'] if predictions else 'None'}",
            f"GT Actions: {', '.join(set([gt['hoi_name'] for gt in ground_truths]))}"
        ]
        
        for i, line in enumerate(analysis_lines):
            self._safe_text_draw(draw, (20, y_start + i * 15), line,
                               self.colors['text_info'], self.font_small)
    
    def _save_detailed_views(self, pred_img: Image.Image, gt_img: Image.Image,
                           predictions: List[Dict], ground_truths: List[Dict], image_id: int):
        """Save individual detailed prediction and ground truth views"""
        pred_img.save(self.prediction_dir / f"image_{image_id}_predictions.jpg")
        gt_img.save(self.prediction_dir / f"image_{image_id}_ground_truth.jpg")
        
        # Save analysis JSON
        analysis_data = {
            'image_id': image_id,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'num_predictions': len(predictions),
            'num_ground_truths': len(ground_truths)
        }
        
        with open(self.analysis_dir / f"image_{image_id}_analysis.json", 'w') as f:
            json.dump(analysis_data, f, indent=2)
    
    def _update_statistics(self, predictions: List[Dict], ground_truths: List[Dict]):
        """Update running statistics for final summary"""
        self.stats['total_predictions'] += len(predictions)
        self.stats['total_ground_truths'] += len(ground_truths)
        
        # Track confidence distribution
        for pred in predictions:
            self.stats['confidence_distribution'].append(pred['confidence'])
        
        # Track HOI class distribution
        for gt in ground_truths:
            hoi_name = gt['hoi_name']
            self.stats['hoi_class_distribution'][hoi_name] = \
                self.stats['hoi_class_distribution'].get(hoi_name, 0) + 1
    
    def generate_summary(self):
        """Generate comprehensive visualization summary"""
        summary_data = {
            'total_images_processed': self.stats['total_images'],
            'total_predictions': self.stats['total_predictions'],
            'total_ground_truths': self.stats['total_ground_truths'],
            'average_predictions_per_image': self.stats['total_predictions'] / max(self.stats['total_images'], 1),
            'average_ground_truths_per_image': self.stats['total_ground_truths'] / max(self.stats['total_images'], 1),
            'top_hoi_classes': sorted(self.stats['hoi_class_distribution'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10],
            'confidence_stats': {
                'mean': np.mean(self.stats['confidence_distribution']) if self.stats['confidence_distribution'] else 0,
                'std': np.std(self.stats['confidence_distribution']) if self.stats['confidence_distribution'] else 0,
                'max': np.max(self.stats['confidence_distribution']) if self.stats['confidence_distribution'] else 0,
                'min': np.min(self.stats['confidence_distribution']) if self.stats['confidence_distribution'] else 0
            }
        }
        
        # Save summary
        with open(self.summary_dir / "visualization_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n🎨 Visualization Summary:")
        print(f"   📸 Images Processed: {summary_data['total_images_processed']}")
        print(f"   🎯 Total Predictions: {summary_data['total_predictions']}")
        print(f"   ✅ Total Ground Truths: {summary_data['total_ground_truths']}")
        print(f"   📊 Avg Predictions/Image: {summary_data['average_predictions_per_image']:.2f}")
        print(f"   📈 Confidence Mean: {summary_data['confidence_stats']['mean']:.3f}")
        print(f"   📁 Output Directory: {self.output_dir}")
        
        return summary_data


def draw_rectangle(drawing, bbox_tuple, color="blue", width=3):
    """Helper function for compatibility with existing INP-CC visualizer"""
    (top_left, bottom_right) = bbox_tuple
    color_map = {
        "blue": (0, 0, 255),
        "red": (255, 0, 0), 
        "green": (0, 255, 0),
        "orange": (255, 165, 0)
    }
    rgb_color = color_map.get(color, (0, 0, 255))
    
    for i in range(width):
        drawing.rectangle([top_left[0]-i, top_left[1]-i, 
                         bottom_right[0]+i, bottom_right[1]+i], 
                        outline=rgb_color, fill=None)