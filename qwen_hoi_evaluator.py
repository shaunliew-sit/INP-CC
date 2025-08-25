#!/usr/bin/env python3
"""
Enhanced Qwen2.5VL HOI Detection Evaluation Script
Properly tailored for both HICO-DET and SWIG-HOI dataset formats
Following the exact same pipeline as INP-CC original inference.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from typing import Dict, List, Tuple, Any, Union
from PIL import Image
import time
from datetime import datetime
import logging
from pathlib import Path
import gc
from scipy.optimize import linear_sum_assignment

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import evaluation framework
from datasets.hico_evaluator import HICOEvaluator
from datasets.swig_evaluator import SWiGEvaluator
from datasets.hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS, hico_unseen_index, NON_INTERACTION_IDS
from datasets.swig_v1_categories import SWIG_INTERACTIONS, SWIG_ACTIONS, SWIG_CATEGORIES

# Qwen2.5VL imports
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Please install required packages:")
    print("pip install transformers torch torchvision qwen-vl-utils")
    sys.exit(1)


class DatasetSpecificHandler:
    """Handles dataset-specific formats and mappings"""
    
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type.lower()
        self._load_dataset_specifics()
    
    def _load_dataset_specifics(self):
        """Load dataset-specific configurations"""
        if self.dataset_type == "hico":
            self._setup_hico()
        elif self.dataset_type == "swig":
            self._setup_swig()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_type}")
    
    def _setup_hico(self):
        """Setup HICO-DET specific configurations"""
        # HICO interactions format: action_object pattern
        self.interactions = HICO_INTERACTIONS
        self.actions = HICO_ACTIONS
        self.objects = HICO_OBJECTS
        
        # HICO uses 600 interaction categories (0-599)
        self.num_interactions = 600
        
        # Text format: [action_name, object_name]
        self.dataset_texts = []
        self.text_mapper = {}
        
        for i, hoi in enumerate(HICO_INTERACTIONS):
            action_name = " ".join(hoi["action"].split("_"))  # Convert underscore to space
            object_name = hoi["object"]
            text_pair = [action_name, object_name]
            self.dataset_texts.append(text_pair)
            self.text_mapper[len(self.dataset_texts)-1] = i
        
        # Create reverse mapping for evaluation
        self.hoi_id_to_text_id = {hoi["interaction_id"]: i for i, hoi in enumerate(HICO_INTERACTIONS)}
        
        # Annotation structure: 'hoi_annotation' key
        self.hoi_annotation_key = "hoi_annotation"
        self.box_annotation_key = "annotations"
        
    def _setup_swig(self):
        """Setup SWIG-HOI specific configurations"""
        # SWIG interactions with evaluation flags
        self.interactions = SWIG_INTERACTIONS
        self.actions = SWIG_ACTIONS
        self.categories = SWIG_CATEGORIES
        
        # Only consider interactions marked for evaluation
        self.eval_interactions = [x for x in SWIG_INTERACTIONS if x.get("evaluation", 0) == 1]
        self.num_interactions = len(self.eval_interactions)
        
        # Text format: [action_name, object_name]
        self.dataset_texts = []
        self.text_mapper = {}
        
        for i, hoi in enumerate(SWIG_INTERACTIONS):
            if hoi.get("evaluation", 0) == 0:
                continue  # Skip non-evaluation interactions
            
            action_id = hoi["action_id"]
            object_id = hoi["object_id"]
            
            action_name = SWIG_ACTIONS[action_id]["name"]
            object_name = SWIG_CATEGORIES[object_id]["name"]
            text_pair = [action_name, object_name]
            
            self.dataset_texts.append(text_pair)
            self.text_mapper[len(self.dataset_texts)-1] = i
        
        # Create mapping from (action_id, object_id) to HOI ID
        self.action_object_to_hoi = {(x["action_id"], x["object_id"]): x["id"] for x in SWIG_INTERACTIONS}
        
        # Annotation structure: 'hoi_annotations' key (plural)
        self.hoi_annotation_key = "hoi_annotations" 
        self.box_annotation_key = "box_annotations"
    
    def get_interaction_text(self, hoi_index: int) -> str:
        """Get text description for HOI index"""
        if hoi_index < len(self.dataset_texts):
            action, obj = self.dataset_texts[hoi_index]
            return f"{action} {obj}"
        return "unknown interaction"
    
    def get_all_interaction_texts(self) -> List[str]:
        """Get all interaction texts for prompt engineering"""
        return [f"{pair[0]} {pair[1]}" for pair in self.dataset_texts]
    
    def map_hoi_name_to_id(self, interaction_name: str) -> Union[int, None]:
        """Map interaction name to dataset-specific HOI ID"""
        interaction_name = interaction_name.strip().lower()
        
        # Try exact matching first
        for i, (action, obj) in enumerate(self.dataset_texts):
            full_name = f"{action} {obj}".lower()
            if full_name == interaction_name:
                return self.text_mapper.get(i, i)
        
        # Try fuzzy matching
        from difflib import SequenceMatcher
        best_score = 0.0
        best_id = None
        
        for i, (action, obj) in enumerate(self.dataset_texts):
            full_name = f"{action} {obj}".lower()
            score = SequenceMatcher(None, interaction_name, full_name).ratio()
            if score > best_score and score > 0.7:
                best_score = score
                best_id = self.text_mapper.get(i, i)
        
        return best_id


class EnhancedQwenHOIPromptEngine:
    """Enhanced prompt engineering tailored for both datasets"""
    
    def __init__(self, dataset_handler: DatasetSpecificHandler):
        self.dataset_handler = dataset_handler
        self.dataset_type = dataset_handler.dataset_type
        self._prepare_dataset_specific_prompts()
    
    def _prepare_dataset_specific_prompts(self):
        """Prepare dataset-specific prompt templates"""
        
        # Base instruction optimized for HOI detection
        self.base_instruction = """You are an expert computer vision system specialized in Human-Object Interaction (HOI) detection. Your task is to analyze images and identify all interactions between humans and objects with precise bounding boxes and confidence scores.

CRITICAL REQUIREMENTS:
1. Output ONLY valid JSON format - no additional text, explanations, or markdown formatting
2. Detect ALL visible human-object interactions in the image  
3. Provide precise bounding boxes in [x1, y1, x2, y2] format (absolute pixel coordinates)
4. Include confidence scores between 0.0 and 1.0 based on detection certainty
5. Use exact interaction names from the provided categories"""

        # Dataset-specific examples
        if self.dataset_type == "hico":
            self.few_shot_examples = [
                {
                    "description": "Person holding a cup",
                    "output": {
                        "interactions": [
                            {
                                "interaction": "hold cup",
                                "confidence": 0.95,
                                "human_bbox": [100, 50, 300, 400],
                                "object_bbox": [250, 180, 320, 250]
                            }
                        ]
                    }
                },
                {
                    "description": "Person riding a bicycle", 
                    "output": {
                        "interactions": [
                            {
                                "interaction": "ride bicycle",
                                "confidence": 0.92,
                                "human_bbox": [150, 80, 250, 350],
                                "object_bbox": [120, 200, 280, 380]
                            }
                        ]
                    }
                },
                {
                    "description": "Person eating a sandwich",
                    "output": {
                        "interactions": [
                            {
                                "interaction": "eat sandwich",
                                "confidence": 0.88,
                                "human_bbox": [80, 40, 220, 350],
                                "object_bbox": [160, 120, 210, 160]
                            }
                        ]
                    }
                }
            ]
        else:  # swig
            self.few_shot_examples = [
                {
                    "description": "Person cutting paper with scissors",
                    "output": {
                        "interactions": [
                            {
                                "interaction": "cut paper",
                                "confidence": 0.88,
                                "human_bbox": [80, 60, 320, 450],
                                "object_bbox": [200, 150, 280, 200]
                            }
                        ]
                    }
                },
                {
                    "description": "Person writing on paper",
                    "output": {
                        "interactions": [
                            {
                                "interaction": "write paper",
                                "confidence": 0.90,
                                "human_bbox": [100, 80, 280, 400],
                                "object_bbox": [150, 200, 250, 280]
                            }
                        ]
                    }
                }
            ]
    
    def create_detection_prompt(self, use_few_shot: bool = True, max_categories: int = 150) -> str:
        """Create comprehensive HOI detection prompt"""
        
        # Get interaction categories (limited to avoid token limits)
        all_interactions = self.dataset_handler.get_all_interaction_texts()
        
        # Prioritize common interactions for better performance
        if len(all_interactions) > max_categories:
            # Take first max_categories (usually more common ones)
            interaction_categories = all_interactions[:max_categories]
            remaining_count = len(all_interactions) - max_categories
            category_text = f"{', '.join(interaction_categories[:50])}\n... and {remaining_count} more standard human-object interactions"
        else:
            interaction_categories = all_interactions
            category_text = ', '.join(interaction_categories[:100])
        
        # Construct prompt
        prompt = f"""{self.base_instruction}

TARGET INTERACTION CATEGORIES ({self.dataset_type.upper()} dataset):
{category_text}

REQUIRED OUTPUT FORMAT (JSON only):
{{
    "interactions": [
        {{
            "interaction": "action object",
            "confidence": 0.0-1.0,
            "human_bbox": [x1, y1, x2, y2],
            "object_bbox": [x1, y1, x2, y2]
        }}
    ]
}}

IMPORTANT NOTES:
- Use absolute pixel coordinates for bounding boxes
- Human bbox should contain the entire person performing the action
- Object bbox should tightly contain the target object
- Confidence should reflect detection certainty (higher for clear, obvious interactions)
- Only detect interactions that are clearly visible and unambiguous"""

        if use_few_shot and self.few_shot_examples:
            prompt += "\n\nEXAMPLES:\n"
            for example in self.few_shot_examples:
                prompt += f"Example: {example['description']}\n"
                prompt += f"Output: {json.dumps(example['output'])}\n\n"
        
        prompt += "Now analyze the given image and provide the JSON output:"
        return prompt


class EnhancedQwenHOIDetector:
    """Enhanced Qwen2.5VL HOI Detection System with dataset-specific handling"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
                 device: str = "auto", dtype: str = "fp16"):
        # Initialize logging first
        self._setup_logging()
        
        self.device = self._setup_device(device)
        self.dtype = torch.float16 if dtype == "fp16" else torch.float32
        self.model_name = model_name
        
        # Load model and tokenizer with CUDA optimizations
        self._load_model()
        
        # Will be set by setup_dataset
        self.dataset_handler = None
        self.prompt_engine = None
        
        # Performance tracking
        self.inference_times = []
        self.total_detections = 0
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup optimal device configuration"""
        if device == "auto":
            if torch.cuda.is_available():
                # Use GPU with highest memory
                device_count = torch.cuda.device_count()
                max_memory = 0
                best_device = 0
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory = props.total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_device = i
                
                device = f"cuda:{best_device}"
                self.logger.info(f"Auto-selected GPU {best_device} with {max_memory/1024**3:.1f}GB memory")
            else:
                device = "cpu"
        
        return torch.device(device)
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"qwen_hoi_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """Load Qwen2.5VL model with CUDA optimizations"""
        self.logger.info(f"Loading {self.model_name} on {self.device}")
        
        # Load with optimal settings following official documentation
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        
        # Enable CUDA optimizations
        if torch.cuda.is_available() and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                torch.backends.cudnn.benchmark = True
                self.logger.info("Model compilation enabled")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
                
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        self.logger.info(f"Model loaded successfully on {self.device}")
    
    def setup_dataset(self, dataset_type: str):
        """Setup dataset-specific configurations"""
        self.dataset_handler = DatasetSpecificHandler(dataset_type)
        self.prompt_engine = EnhancedQwenHOIPromptEngine(self.dataset_handler)
        
        self.logger.info(f"Dataset setup completed for {dataset_type.upper()}")
        self.logger.info(f"Number of interaction categories: {self.dataset_handler.num_interactions}")
        
    def create_dataset_loader(self, data_root: str, annotation_file: str, max_images: int = None):
        """Create optimized dataset loader"""
        return EnhancedDatasetLoader(
            dataset_type=self.dataset_handler.dataset_type,
            data_root=data_root,
            annotation_file=annotation_file,
            dataset_handler=self.dataset_handler,
            max_images=max_images
        )
    
    @torch.no_grad()
    def detect_hoi_batch(self, images: List[Image.Image], 
                        batch_size: int = 4) -> List[Dict]:
        """Batch processing for HOI detection with memory optimization"""
        all_results = []
        
        # Process in batches to manage memory
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = []
            
            for image in batch_images:
                result = self.detect_hoi_single(image)
                batch_results.append(result)
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            all_results.extend(batch_results)
            
            # Progress tracking
            self.logger.debug(f"Processed batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1}")
        
        return all_results
    
    @torch.no_grad()
    def detect_hoi_single(self, image: Image.Image) -> Dict:
        """Single image HOI detection with optimized inference"""
        start_time = time.time()
        
        try:
            # Create detection prompt
            prompt = self.prompt_engine.create_detection_prompt()
            
            # Prepare inputs
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process with optimal settings
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate with optimal parameters  
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # Increased for complex scenes
                temperature=0.1,      # Low temperature for consistent outputs
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Parse JSON output
            result = self._parse_hoi_output(output_text, image.size)
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_detections += len(result.get('interactions', []))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in HOI detection: {str(e)}")
            return {'interactions': []}
    
    def _parse_hoi_output(self, output_text: str, image_size: Tuple[int, int]) -> Dict:
        """Parse and validate Qwen2.5VL output with dataset-specific mapping"""
        try:
            # Extract JSON from output
            json_start = output_text.find('{')
            json_end = output_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return {'interactions': []}
            
            json_str = output_text[json_start:json_end]
            parsed_output = json.loads(json_str)
            
            # Validate and normalize format
            if 'interactions' not in parsed_output:
                return {'interactions': []}
            
            validated_interactions = []
            img_width, img_height = image_size
            
            for interaction in parsed_output['interactions']:
                if self._validate_interaction(interaction, img_width, img_height):
                    # Map interaction name to dataset-specific ID
                    interaction_name = interaction.get('interaction', '').strip().lower()
                    hoi_id = self.dataset_handler.map_hoi_name_to_id(interaction_name)
                    
                    if hoi_id is not None:
                        validated_interactions.append({
                            'hoi_id': hoi_id,
                            'confidence': float(interaction.get('confidence', 0.5)),
                            'human_bbox': interaction['human_bbox'],
                            'object_bbox': interaction['object_bbox'],
                            'interaction_name': interaction_name  # Keep for debugging
                        })
            
            return {'interactions': validated_interactions}
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse HOI output: {str(e)}")
            return {'interactions': []}
    
    def _validate_interaction(self, interaction: Dict, img_width: int, img_height: int) -> bool:
        """Validate interaction detection format and bounds"""
        required_keys = ['interaction', 'confidence', 'human_bbox', 'object_bbox']
        
        if not all(key in interaction for key in required_keys):
            return False
        
        # Validate confidence
        conf = interaction.get('confidence')
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            return False
        
        # Validate bounding boxes
        for bbox_key in ['human_bbox', 'object_bbox']:
            bbox = interaction.get(bbox_key)
            if not isinstance(bbox, list) or len(bbox) != 4:
                return False
            
            x1, y1, x2, y2 = bbox
            if not all(isinstance(coord, (int, float)) for coord in bbox):
                return False
            
            # Check bounds and validity
            if not (0 <= x1 < x2 <= img_width and 0 <= y1 < y2 <= img_height):
                return False
        
        return True
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'total_images': len(self.inference_times),
            'total_detections': self.total_detections,
            'avg_detections_per_image': self.total_detections / len(self.inference_times),
            'images_per_second': 1.0 / np.mean(self.inference_times)
        }


class EnhancedDatasetLoader:
    """Enhanced data loading for both HICO-DET and SWIG-HOI with proper format handling"""
    
    def __init__(self, dataset_type: str, data_root: str, annotation_file: str, 
                 dataset_handler: DatasetSpecificHandler, max_images: int = None):
        self.dataset_type = dataset_type.lower()
        self.data_root = Path(data_root)
        self.annotation_file = annotation_file
        self.dataset_handler = dataset_handler
        self.max_images = max_images
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        if max_images:
            self.annotations = self.annotations[:max_images]
        
        # Setup paths based on dataset
        if self.dataset_type == "hico":
            self.image_dir = self.data_root / "hico_20160224_det" / "images" / "test2015"
        else:  # swig
            self.image_dir = self.data_root / "swig_hoi" / "images_512"
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded {len(self.annotations)} annotations from {annotation_file}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict]:
        """Get image and annotation with proper format handling"""
        annotation = self.annotations[idx]
        
        # Handle different file name formats
        if self.dataset_type == "hico":
            # HICO format: direct file_name usage  
            image_path = self.image_dir / annotation['file_name']
            image_id = annotation.get('img_id', idx)
        else:  # swig
            # SWIG format: file_name is relative path
            image_path = self.image_dir / annotation['file_name']
            image_id = annotation.get('img_id', idx)
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image, {
                'image_id': image_id,
                'annotation': annotation,
                'orig_size': (annotation.get('height', image.height), annotation.get('width', image.width))
            }
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {str(e)}")
            # Return dummy image
            return Image.new('RGB', (224, 224), color='black'), {
                'image_id': image_id, 
                'annotation': annotation,
                'orig_size': (224, 224)
            }
    
    def get_batch(self, indices: List[int]) -> Tuple[List[Image.Image], List[Dict]]:
        """Get batch of images and annotations"""
        images, annotations = [], []
        
        for idx in indices:
            img, ann = self[idx]
            images.append(img)
            annotations.append(ann)
        
        return images, annotations


class QwenHOILossComputer:
    """
    Loss computation framework for Qwen2.5VL HOI detection to enable direct comparison with INP-CC.
    Computes post-hoc losses between predictions and ground truth using the same methodology as INP-CC criterion.
    """
    
    def __init__(self, dataset_handler: DatasetSpecificHandler, device: torch.device = None):
        self.dataset_handler = dataset_handler
        self.device = device if device else torch.device('cpu')
        
        # Loss tracking (initialize empty to allow dynamic keys)
        self.loss_stats = {}
        
        # Matching parameters (same as INP-CC)
        self.cost_class = 1.0
        self.cost_bbox = 5.0
        self.cost_giou = 2.0
        self.cost_conf = 1.0
        
        # Loss weights (same as INP-CC default)
        self.weight_dict = {
            'loss_ce': 5.0,      # INP-CC uses 5.0 scaling for classification
            'loss_bbox': 5.0,    # INP-CC uses 5.0 scaling for bbox
            'loss_giou': 2.0,    # INP-CC uses 2.0 scaling for giou  
            'loss_conf': 10.0    # INP-CC uses 10.0 scaling for confidence
        }
        
        self.logger = logging.getLogger(__name__)
    
    def compute_losses(self, predictions: List[Dict], ground_truth: List[Dict], 
                      image_sizes: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        Compute losses between Qwen2.5VL predictions and ground truth annotations.
        
        Args:
            predictions: List of prediction dicts with 'interactions' key
            ground_truth: List of ground truth annotation dicts
            image_sizes: List of (width, height) tuples for each image
            
        Returns:
            Dict with computed loss values
        """
        if not predictions or not ground_truth:
            return self._zero_losses()
        
        batch_losses = []
        
        for pred, gt, img_size in zip(predictions, ground_truth, image_sizes):
            try:
                # Convert to tensor format for loss computation
                pred_data = self._prepare_predictions(pred, img_size)
                target_data = self._prepare_targets(gt, img_size)
                
                if not pred_data['interactions'] or not target_data['hois']:
                    # No predictions or targets - add zero loss
                    batch_losses.append(self._zero_losses())
                    continue
                
                # Perform Hungarian matching (same as INP-CC)
                indices = self._hungarian_matching(pred_data, target_data)
                
                # Compute individual loss components  
                losses = {}
                losses.update(self._compute_classification_loss(pred_data, target_data, indices))
                losses.update(self._compute_bbox_loss(pred_data, target_data, indices))
                losses.update(self._compute_confidence_loss(pred_data, target_data, indices, len(target_data['hois'])))
                
                # Add unscaled versions for INP-CC compatibility
                unscaled_losses = {
                    'loss_ce_unscaled': losses.get('loss_ce', 0.0),
                    'loss_bbox_unscaled': losses.get('loss_bbox', 0.0), 
                    'loss_giou_unscaled': losses.get('loss_giou', 0.0),
                    'loss_conf_unscaled': losses.get('loss_conf', 0.0)
                }
                
                # Apply INP-CC scaling to match their output
                scaled_losses = {
                    'loss_ce': losses.get('loss_ce', 0.0) * self.weight_dict['loss_ce'],
                    'loss_bbox': losses.get('loss_bbox', 0.0) * self.weight_dict['loss_bbox'],
                    'loss_giou': losses.get('loss_giou', 0.0) * self.weight_dict['loss_giou'],
                    'loss_conf': losses.get('loss_conf', 0.0) * self.weight_dict['loss_conf']
                }
                
                # Total loss (weighted sum)
                total_loss = sum(scaled_losses.values())
                
                # Combine all losses 
                final_losses = {**scaled_losses, **unscaled_losses, 'total_loss': total_loss}
                
                batch_losses.append(final_losses)
                
            except Exception as e:
                self.logger.warning(f"Error computing losses for sample: {e}")
                batch_losses.append(self._zero_losses())
        
        # Average losses across batch
        avg_losses = self._average_batch_losses(batch_losses)
        
        # Update running statistics
        for key, value in avg_losses.items():
            if key not in self.loss_stats:
                self.loss_stats[key] = []
            self.loss_stats[key].append(value)
        
        return avg_losses
    
    def _prepare_predictions(self, pred: Dict, img_size: Tuple[int, int]) -> Dict:
        """Convert Qwen2.5VL predictions to tensor format for loss computation"""
        interactions = pred.get('interactions', [])
        
        if not interactions:
            return {'interactions': [], 'logits': None, 'boxes': None, 'conf_scores': None}
        
        num_preds = len(interactions)
        img_w, img_h = img_size
        
        # Prepare logits (one-hot encoded from predicted HOI IDs)
        logits = torch.zeros(num_preds, self.dataset_handler.num_interactions, device=self.device)
        pred_boxes = []
        conf_scores = []
        
        for i, interaction in enumerate(interactions):
            hoi_id = interaction.get('hoi_id', 0)
            if 0 <= hoi_id < self.dataset_handler.num_interactions:
                logits[i, hoi_id] = 10.0  # High logit for predicted class
            
            # Convert bboxes to normalized cxcywh format
            human_bbox = interaction.get('human_bbox', [0, 0, 1, 1])
            object_bbox = interaction.get('object_bbox', [0, 0, 1, 1])
            
            # Normalize and convert to cxcywh
            h_x1, h_y1, h_x2, h_y2 = human_bbox
            o_x1, o_y1, o_x2, o_y2 = object_bbox
            
            h_cx = (h_x1 + h_x2) / (2 * img_w)
            h_cy = (h_y1 + h_y2) / (2 * img_h)
            h_w = (h_x2 - h_x1) / img_w
            h_h = (h_y2 - h_y1) / img_h
            
            o_cx = (o_x1 + o_x2) / (2 * img_w)
            o_cy = (o_y1 + o_y2) / (2 * img_h)
            o_w = (o_x2 - o_x1) / img_w
            o_h = (o_y2 - o_y1) / img_h
            
            pred_boxes.append([h_cx, h_cy, h_w, h_h, o_cx, o_cy, o_w, o_h])
            conf_scores.append(interaction.get('confidence', 0.5))
        
        pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32, device=self.device)
        conf_scores = torch.tensor(conf_scores, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        return {
            'interactions': interactions,
            'logits': logits,
            'boxes': pred_boxes,
            'conf_scores': conf_scores
        }
    
    def _prepare_targets(self, gt: Dict, img_size: Tuple[int, int]) -> Dict:
        """Convert ground truth annotations to tensor format"""
        annotation = gt.get('annotation', {})
        
        # Get HOI annotations based on dataset type
        if self.dataset_handler.dataset_type == "hico":
            hoi_annotations = annotation.get('hoi_annotation', [])
            box_annotations = annotation.get('annotations', [])
        else:  # swig
            hoi_annotations = annotation.get('hoi_annotations', [])
            box_annotations = annotation.get('box_annotations', [])
        
        if not hoi_annotations:
            return {'hois': [], 'boxes': None}
        
        img_w, img_h = img_size
        
        # Process bounding boxes
        boxes = []
        for box_ann in box_annotations:
            bbox = box_ann.get('bbox', [0, 0, 1, 1])
            x, y, w, h = bbox
            
            # Convert to normalized cxcywh
            cx = (x + w/2) / img_w
            cy = (y + h/2) / img_h
            nw = w / img_w
            nh = h / img_h
            
            boxes.append([cx, cy, nw, nh])
        
        boxes = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        
        return {
            'hois': hoi_annotations,
            'boxes': boxes
        }
    
    def _hungarian_matching(self, pred_data: Dict, target_data: Dict) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Perform Hungarian matching between predictions and targets (same as INP-CC)"""
        if not pred_data['interactions'] or not target_data['hois']:
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))]
        
        pred_logits = pred_data['logits']  # [num_preds, num_classes]
        pred_boxes = pred_data['boxes']    # [num_preds, 8]
        pred_conf = pred_data['conf_scores'] # [num_preds, 1]
        
        target_hois = target_data['hois']
        target_boxes = target_data['boxes']
        
        num_preds = len(pred_data['interactions'])
        num_targets = len(target_hois)
        
        # Build target data for matching
        tgt_ids = []
        tgt_bbox = []
        
        for hoi in target_hois:
            hoi_id = hoi.get('hoi_id', 0)
            subject_id = hoi.get('subject_id', 0)
            object_id = hoi.get('object_id', 0)
            
            tgt_ids.append(hoi_id)
            
            if subject_id < len(target_boxes) and object_id < len(target_boxes):
                subject_box = target_boxes[subject_id]
                object_box = target_boxes[object_id]
                combined_box = torch.cat([subject_box, object_box])
            else:
                # Default fallback
                combined_box = torch.zeros(8, device=self.device)
            
            tgt_bbox.append(combined_box)
        
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.int64, device=self.device)
        tgt_bbox = torch.stack(tgt_bbox)
        
        # Compute cost matrices (same as INP-CC HungarianMatcher)
        out_prob = pred_logits.sigmoid()
        cost_class = -out_prob[:, tgt_ids]  # [num_preds, num_targets]
        
        cost_conf = -pred_conf  # [num_preds, 1] -> broadcast to [num_preds, num_targets]
        
        # L1 cost for bounding boxes
        cost_pbbox = torch.cdist(pred_boxes[:, :4], tgt_bbox[:, :4], p=1)
        cost_obbox = torch.cdist(pred_boxes[:, 4:], tgt_bbox[:, 4:], p=1)
        
        # GIoU cost
        pred_h_boxes_xyxy = self._cxcywh_to_xyxy(pred_boxes[:, :4])
        pred_o_boxes_xyxy = self._cxcywh_to_xyxy(pred_boxes[:, 4:])
        tgt_h_boxes_xyxy = self._cxcywh_to_xyxy(tgt_bbox[:, :4])
        tgt_o_boxes_xyxy = self._cxcywh_to_xyxy(tgt_bbox[:, 4:])
        
        cost_pgiou = -self._generalized_box_iou(pred_h_boxes_xyxy, tgt_h_boxes_xyxy)
        cost_ogiou = -self._generalized_box_iou(pred_o_boxes_xyxy, tgt_o_boxes_xyxy)
        
        # Final cost matrix
        C = (self.cost_bbox * cost_pbbox + self.cost_bbox * cost_obbox +
             self.cost_giou * cost_pgiou + self.cost_giou * cost_ogiou +
             self.cost_class * cost_class + self.cost_conf * cost_conf)
        
        # Hungarian assignment
        C_cpu = C.cpu().numpy()
        indices = linear_sum_assignment(C_cpu)
        
        pred_indices = torch.tensor(indices[0], dtype=torch.int64, device=self.device)
        tgt_indices = torch.tensor(indices[1], dtype=torch.int64, device=self.device)
        
        return [(pred_indices, tgt_indices)]
    
    def _compute_classification_loss(self, pred_data: Dict, target_data: Dict, 
                                   indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Compute cross-entropy classification loss (same as INP-CC)"""
        pred_logits = pred_data['logits']
        target_hois = target_data['hois']
        
        if not indices or len(indices[0][0]) == 0:
            return {'loss_ce': 0.0}
        
        pred_idx, tgt_idx = indices[0]
        
        # Get target classes for matched predictions
        target_classes = []
        for i in tgt_idx:
            hoi_id = target_hois[i].get('hoi_id', 0)
            target_classes.append(hoi_id)
        
        target_classes = torch.tensor(target_classes, dtype=torch.int64, device=self.device)
        matched_logits = pred_logits[pred_idx]
        
        # INP-CC uses image-to-text alignment loss approach
        # Apply softmax then take negative log likelihood manually to match INP-CC
        probs = F.softmax(matched_logits, dim=-1)
        
        # Gather target probabilities and compute negative log likelihood
        target_probs = probs.gather(1, target_classes.unsqueeze(1)).squeeze(1)
        loss_ce = -torch.log(target_probs + 1e-8).mean()
        
        return {'loss_ce': loss_ce.item()}
    
    def _compute_bbox_loss(self, pred_data: Dict, target_data: Dict, 
                         indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Compute L1 and GIoU losses for bounding boxes (same as INP-CC)"""
        if not indices or len(indices[0][0]) == 0:
            return {'loss_bbox': 0.0, 'loss_giou': 0.0}
        
        pred_boxes = pred_data['boxes']
        target_hois = target_data['hois']
        target_boxes = target_data['boxes']
        
        pred_idx, tgt_idx = indices[0]
        
        # Get matched predictions and targets
        matched_pred_boxes = pred_boxes[pred_idx]
        
        matched_target_boxes = []
        for i in tgt_idx:
            hoi = target_hois[i]
            subject_id = hoi.get('subject_id', 0)
            object_id = hoi.get('object_id', 0)
            
            if subject_id < len(target_boxes) and object_id < len(target_boxes):
                subject_box = target_boxes[subject_id]
                object_box = target_boxes[object_id]
                combined_box = torch.cat([subject_box, object_box])
            else:
                combined_box = torch.zeros(8, device=self.device)
            
            matched_target_boxes.append(combined_box)
        
        matched_target_boxes = torch.stack(matched_target_boxes)
        
        # L1 loss for person and object boxes (normalized by number of boxes like INP-CC)
        num_boxes = len(matched_pred_boxes)
        loss_pbbox = F.l1_loss(matched_pred_boxes[:, :4], matched_target_boxes[:, :4], reduction='sum') / num_boxes
        loss_obbox = F.l1_loss(matched_pred_boxes[:, 4:], matched_target_boxes[:, 4:], reduction='sum') / num_boxes
        loss_bbox = loss_pbbox + loss_obbox
        
        # GIoU loss (normalized by number of boxes like INP-CC)
        pred_h_xyxy = self._cxcywh_to_xyxy(matched_pred_boxes[:, :4])
        pred_o_xyxy = self._cxcywh_to_xyxy(matched_pred_boxes[:, 4:])
        tgt_h_xyxy = self._cxcywh_to_xyxy(matched_target_boxes[:, :4])
        tgt_o_xyxy = self._cxcywh_to_xyxy(matched_target_boxes[:, 4:])
        
        loss_pgiou = 1 - torch.diag(self._generalized_box_iou(pred_h_xyxy, tgt_h_xyxy))
        loss_ogiou = 1 - torch.diag(self._generalized_box_iou(pred_o_xyxy, tgt_o_xyxy))
        loss_giou = (loss_pgiou.sum() + loss_ogiou.sum()) / num_boxes
        
        return {
            'loss_bbox': loss_bbox.item(),
            'loss_giou': loss_giou.item()
        }
    
    def _compute_confidence_loss(self, pred_data: Dict, target_data: Dict, 
                               indices: List[Tuple[torch.Tensor, torch.Tensor]], num_targets: int) -> Dict[str, float]:
        """Compute confidence loss (same as INP-CC)"""
        pred_conf = pred_data['conf_scores'].sigmoid()
        
        if not indices:
            return {'loss_conf': 0.0}
        
        pred_idx, tgt_idx = indices[0]
        
        # Create target confidence labels (1 for matched, 0 for unmatched)
        target_conf = torch.zeros_like(pred_conf.flatten())
        if len(pred_idx) > 0:
            target_conf[pred_idx] = 1.0
        
        # Binary cross-entropy loss
        loss_conf = F.binary_cross_entropy(pred_conf.flatten(), target_conf, reduction='mean')
        
        return {'loss_conf': loss_conf.item()}
    
    def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert from (cx, cy, w, h) to (x1, y1, x2, y2) format"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute generalized IoU (same as INP-CC)"""
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        
        iou, union = self._box_iou(boxes1, boxes2)
        
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        area = wh[:, :, 0] * wh[:, :, 1]
        
        return iou - (area - union) / area
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        return iou, union
    
    def _zero_losses(self) -> Dict[str, float]:
        """Return zero losses for edge cases"""
        return {
            'loss_ce': 0.0,
            'loss_bbox': 0.0,
            'loss_giou': 0.0,
            'loss_conf': 0.0,
            'loss_ce_unscaled': 0.0,
            'loss_bbox_unscaled': 0.0,
            'loss_giou_unscaled': 0.0,
            'loss_conf_unscaled': 0.0,
            'total_loss': 0.0
        }
    
    def _average_batch_losses(self, batch_losses: List[Dict[str, float]]) -> Dict[str, float]:
        """Average losses across a batch"""
        if not batch_losses:
            return self._zero_losses()
        
        avg_losses = {}
        loss_keys = ['loss_ce', 'loss_bbox', 'loss_giou', 'loss_conf', 
                     'loss_ce_unscaled', 'loss_bbox_unscaled', 'loss_giou_unscaled', 'loss_conf_unscaled',
                     'total_loss']
        
        for key in loss_keys:
            values = [loss_dict.get(key, 0.0) for loss_dict in batch_losses]
            avg_losses[key] = np.mean(values)
        
        return avg_losses
    
    def get_final_loss_statistics(self) -> Dict[str, float]:
        """Get final averaged loss statistics across all processed samples"""
        final_stats = {}
        
        for key, values in self.loss_stats.items():
            if values:
                final_stats[key] = np.mean(values)
            else:
                final_stats[key] = 0.0
        
        return final_stats


class EnhancedPostProcessor:
    """Enhanced post-processor with dataset-specific handling"""
    
    def __init__(self, dataset_handler: DatasetSpecificHandler, 
                 score_threshold: float = 0.1, nms_threshold: float = 0.5):
        self.dataset_handler = dataset_handler
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
    
    def __call__(self, qwen_output: Dict, image_id: int) -> List[List]:
        """Convert Qwen output to evaluation format following INP-CC postprocessors"""
        results = []
        
        for interaction in qwen_output.get('interactions', []):
            confidence = interaction.get('confidence', 0.0)
            
            # Filter by threshold
            if confidence < self.score_threshold:
                continue
            
            hoi_id = interaction.get('hoi_id')
            human_bbox = interaction.get('human_bbox', [])
            object_bbox = interaction.get('object_bbox', [])
            
            if hoi_id is not None and len(human_bbox) == 4 and len(object_bbox) == 4:
                # Format following INP-CC PostProcess: [hoi_id, score, h_x1, h_y1, h_x2, h_y2, o_x1, o_y1, o_x2, o_y2]
                result = [hoi_id, confidence] + human_bbox + object_bbox
                results.append(result)
        
        return results
    
    def apply_nms(self, detections: List[List]) -> List[List]:
        """Apply Non-Maximum Suppression to reduce duplicate detections"""
        if not detections or self.nms_threshold >= 1.0:
            return detections
        
        # Group by HOI ID (following INP-CC approach)
        hoi_groups = {}
        for det in detections:
            hoi_id = det[0]
            if hoi_id not in hoi_groups:
                hoi_groups[hoi_id] = []
            hoi_groups[hoi_id].append(det)
        
        final_results = []
        
        for hoi_id, group in hoi_groups.items():
            if len(group) <= 1:
                final_results.extend(group)
                continue
            
            # Sort by confidence (descending)
            group.sort(key=lambda x: x[1], reverse=True)
            
            keep = []
            for i, det_i in enumerate(group):
                should_keep = True
                
                for det_j in keep:
                    # Calculate IoU between human boxes and object boxes
                    human_iou = self._calculate_iou(det_i[2:6], det_j[2:6])
                    object_iou = self._calculate_iou(det_i[6:10], det_j[6:10])
                    
                    # If both IoUs are high, consider it a duplicate (following INP-CC logic)
                    if human_iou > self.nms_threshold and object_iou > self.nms_threshold:
                        should_keep = False
                        break
                
                if should_keep:
                    keep.append(det_i)
            
            final_results.extend(keep)
        
        return final_results
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate IoU between two boxes (same as INP-CC implementation)"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Enhanced Qwen2.5VL HOI Detection Evaluation")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Qwen2.5VL model name")
    parser.add_argument('--device', type=str, default="auto", 
                       help="Device to use (auto, cuda, cuda:0, cpu)")
    parser.add_argument('--dtype', type=str, default="fp16", choices=["fp16", "fp32"],
                       help="Model precision")
    
    # Dataset arguments
    parser.add_argument('--dataset_file', type=str, required=True, choices=["hico", "swig"],
                       help="Dataset type")
    parser.add_argument('--data_root', type=str, required=True,
                       help="Root directory of dataset")
    parser.add_argument('--annotation_file', type=str, required=True,
                       help="Path to test annotation file")
    
    # Evaluation arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help="Output directory for results")
    parser.add_argument('--batch_size', type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument('--max_images', type=int, default=None,
                       help="Maximum number of images to evaluate (for testing)")
    parser.add_argument('--score_threshold', type=float, default=0.1,
                       help="Confidence threshold for detections")
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                       help="NMS threshold")
    
    # HICO-specific arguments
    parser.add_argument('--zero_shot_type', type=str, default="rare_first",
                       choices=["rare_first", "non_rare", "unseen_verb", "unseen_object"],
                       help="Zero-shot evaluation type for HICO")
    parser.add_argument('--ignore_non_interaction', action='store_true', default=True,
                       help="Ignore non-interaction categories in HICO")
    
    # Performance arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help="Number of workers for data loading")
    parser.add_argument('--save_predictions', action='store_true',
                       help="Save raw predictions for analysis")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Enhanced Qwen2.5VL HOI Detection Evaluation")
    logger.info(f"Dataset: {args.dataset_file}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    
    # Initialize components
    detector = EnhancedQwenHOIDetector(args.model_name, args.device, args.dtype)
    detector.setup_dataset(args.dataset_file)
    
    dataloader = detector.create_dataset_loader(args.data_root, args.annotation_file, args.max_images)
    postprocessor = EnhancedPostProcessor(detector.dataset_handler, args.score_threshold, args.nms_threshold)
    
    # Initialize loss computer for INP-CC-compatible loss computation
    loss_computer = QwenHOILossComputer(detector.dataset_handler, detector.device)
    logger.info("Loss computation framework initialized")
    
    # Initialize evaluator (following INP-CC approach)
    if args.dataset_file == "hico":
        evaluator = HICOEvaluator(
            args.annotation_file, 
            args.output_dir,
            args.zero_shot_type,
            args.ignore_non_interaction
        )
    else:  # swig
        evaluator = SWiGEvaluator(args.annotation_file, args.output_dir)
    
    logger.info(f"Loaded dataset with {len(dataloader)} images")
    
    # Evaluation loop
    total_images = len(dataloader)
    all_predictions = {}
    
    # Process in batches
    batch_indices = []
    for i in range(total_images):
        batch_indices.append(i)
        
        if len(batch_indices) == args.batch_size or i == total_images - 1:
            # Load batch
            images, annotations = dataloader.get_batch(batch_indices)
            
            # Run detection
            logger.info(f"Processing batch {i//args.batch_size + 1}/{(total_images-1)//args.batch_size + 1}")
            batch_results = detector.detect_hoi_batch(images, args.batch_size)
            
            # Compute losses for this batch
            image_sizes = [(img.width, img.height) for img in images]
            batch_losses = loss_computer.compute_losses(batch_results, annotations, image_sizes)
            logger.debug(f"Batch losses: {batch_losses}")
            
            # Process results (following INP-CC evaluation format)
            for img, ann, result in zip(images, annotations, batch_results):
                image_id = ann['image_id']
                
                # Post-process (following INP-CC PostProcess)
                predictions = postprocessor(result, image_id)
                predictions = postprocessor.apply_nms(predictions)
                
                all_predictions[image_id] = predictions
            
            # Update evaluator (following INP-CC evaluation pipeline)
            evaluator.update(all_predictions)
            
            # Clear batch
            batch_indices = []
            all_predictions = {}
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    logger.info("Running evaluation metrics...")
    
    # Compute final metrics (following INP-CC evaluation)
    evaluator.accumulate()
    
    # Capture mAP metrics manually since summarize() only prints
    if args.dataset_file == "hico":
        # Extract mAP values for HICO-DET
        from datasets.hico_categories import NON_INTERACTION_IDS
        
        if args.ignore_non_interaction:
            valid_hois = np.setdiff1d(np.arange(600), NON_INTERACTION_IDS)
            seen_hois = np.setdiff1d(valid_hois, evaluator.zero_shot_interaction_ids)
            zero_shot_hois = np.setdiff1d(evaluator.zero_shot_interaction_ids, NON_INTERACTION_IDS)
        else:
            valid_hois = np.setdiff1d(np.arange(600), [])
            seen_hois = np.setdiff1d(valid_hois, evaluator.zero_shot_interaction_ids)
            zero_shot_hois = np.setdiff1d(evaluator.zero_shot_interaction_ids, [])
            
        zero_shot_mAP = np.mean(evaluator.hico_ap[zero_shot_hois])
        seen_mAP = np.mean(evaluator.hico_ap[seen_hois])
        full_mAP = np.mean(evaluator.hico_ap[valid_hois])
        
        # HICO-specific metrics (uses zero_shot and seen terminology)
        metrics = {
            "zero_shot_mAP": float(zero_shot_mAP),
            "seen_mAP": float(seen_mAP), 
            "full_mAP": float(full_mAP)
        }
    else:  # SWIG
        # Extract mAP values for SWIG-HOI dataset
        from datasets.swig_v1_categories import SWIG_INTERACTIONS
        
        eval_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1])
        zero_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 0 and x["evaluation"] == 1])
        rare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 1 and x["evaluation"] == 1])
        nonrare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 2 and x["evaluation"] == 1])
        
        full_mAP = np.mean(evaluator.swig_ap[eval_hois])
        zero_shot_mAP = np.mean(evaluator.swig_ap[zero_hois])
        rare_mAP = np.mean(evaluator.swig_ap[rare_hois])
        nonrare_mAP = np.mean(evaluator.swig_ap[nonrare_hois])
        
        # SWIG-specific metrics (uses zero_shot, rare, nonrare terminology)  
        metrics = {
            "zero_shot_mAP": float(zero_shot_mAP),
            "rare_mAP": float(rare_mAP),
            "nonrare_mAP": float(nonrare_mAP),
            "full_mAP": float(full_mAP)
        }
    
    # Also call the original summarize for console output
    evaluator.summarize()
    
    # Get final loss statistics
    loss_stats = loss_computer.get_final_loss_statistics()
    logger.info("Loss Statistics (INP-CC compatible):")
    
    # Log scaled losses first
    logger.info("Scaled losses:")
    for key in ['loss_ce', 'loss_bbox', 'loss_giou', 'loss_conf', 'total_loss']:
        if key in loss_stats:
            logger.info(f"  {key}: {loss_stats[key]:.4f}")
    
    # Log unscaled losses
    logger.info("Unscaled losses:")
    for key in ['loss_ce_unscaled', 'loss_bbox_unscaled', 'loss_giou_unscaled', 'loss_conf_unscaled']:
        if key in loss_stats:
            logger.info(f"  {key}: {loss_stats[key]:.4f}")
        else:
            logger.info(f"  {key}: 0.0000 (missing)")
    
    # Log scaling factors used
    logger.info("Scaling factors applied:")
    logger.info(f"  loss_ce_scale: {loss_computer.weight_dict['loss_ce']}")
    logger.info(f"  loss_bbox_scale: {loss_computer.weight_dict['loss_bbox']}")
    logger.info(f"  loss_giou_scale: {loss_computer.weight_dict['loss_giou']}")
    logger.info(f"  loss_conf_scale: {loss_computer.weight_dict['loss_conf']}")
    
    # Prepare enhanced loss metrics with both scaled and unscaled values + scaling info
    enhanced_loss_metrics = {
        # Scaled losses (main losses)
        "loss_ce": loss_stats.get("loss_ce", 0.0),
        "loss_bbox": loss_stats.get("loss_bbox", 0.0), 
        "loss_giou": loss_stats.get("loss_giou", 0.0),
        "loss_conf": loss_stats.get("loss_conf", 0.0),
        "total_loss": loss_stats.get("total_loss", 0.0),
        
        # Unscaled losses (INP-CC compatible)
        "loss_ce_unscaled": loss_stats.get("loss_ce_unscaled", 0.0),
        "loss_bbox_unscaled": loss_stats.get("loss_bbox_unscaled", 0.0),
        "loss_giou_unscaled": loss_stats.get("loss_giou_unscaled", 0.0), 
        "loss_conf_unscaled": loss_stats.get("loss_conf_unscaled", 0.0),
        
        # Scaling factors for transparency
        "scaling_factors": {
            "loss_ce_scale": loss_computer.weight_dict['loss_ce'],
            "loss_bbox_scale": loss_computer.weight_dict['loss_bbox'],
            "loss_giou_scale": loss_computer.weight_dict['loss_giou'],
            "loss_conf_scale": loss_computer.weight_dict['loss_conf']
        }
    }
    
    # Combine evaluation and loss metrics in required format
    combined_results = {
        "evaluation_metrics": metrics,
        "loss_metrics": enhanced_loss_metrics
    }
    
    # Save combined results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    # Performance statistics
    perf_stats = detector.get_performance_stats()
    logger.info("Performance Statistics:")
    for key, value in perf_stats.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save performance stats
    perf_file = output_dir / "performance_stats.json"
    with open(perf_file, 'w') as f:
        json.dump(perf_stats, f, indent=2)
    
    # Save loss statistics separately for analysis
    loss_file = output_dir / "loss_statistics.json"
    with open(loss_file, 'w') as f:
        json.dump(loss_stats, f, indent=2)
    
    if args.save_predictions:
        evaluator.save_preds()
        logger.info(f"Predictions saved to {args.output_dir}")
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Loss statistics saved to: {loss_file}")
    
    return combined_results


if __name__ == "__main__":
    main()