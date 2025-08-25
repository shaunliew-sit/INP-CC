# HOI Detection Evaluation Framework

This document provides a comprehensive guide to implement the evaluation metrics and pipeline used in INP-CC for Human-Object Interaction (HOI) detection, which can be adapted for Qwen2.5VL or other HOI detection models.

## Core Evaluation Metrics

### 1. HICO-DET Evaluation Metrics
- **Primary Metric**: mAP (mean Average Precision) using **11-point interpolation**
- **Evaluation Categories**:
  - **Unseen/Zero-shot mAP**: Performance on novel interaction classes
  - **Seen mAP**: Performance on training interaction classes  
  - **Full mAP**: Overall performance across all 600 HOI categories
- **IoU Threshold**: 0.5 for both human and object boxes
- **Hit Criteria**: `min(human_IoU, object_IoU) >= 0.5`

### 2. SWIG-HOI Evaluation Metrics
- **Primary Metric**: mAP using **11-point interpolation**
- **Evaluation Categories**:
  - **Zero-shot mAP**: Unseen interactions (frequency=0)
  - **Rare mAP**: Rare interactions (frequency=1) 
  - **Non-rare mAP**: Common interactions (frequency=2)
  - **Full mAP**: Overall performance
- **IoU Threshold**: 0.5 for both human and object boxes
- **Hit Criteria**: Same as HICO-DET

## Evaluation Pipeline Implementation

### Step 1: Prediction Format

Your model predictions must follow this exact format:

```python
# For each image
predictions = {
    image_id: [
        [hoi_id, confidence_score, h_x1, h_y1, h_x2, h_y2, o_x1, o_y1, o_x2, o_y2],
        # ... more predictions
    ]
}
```

**Format Details:**
- `image_id`: Integer image identifier
- `hoi_id`: HOI category ID (0-599 for HICO-DET)
- `confidence_score`: Float confidence score [0, 1]
- `h_x1, h_y1, h_x2, h_y2`: Human bounding box coordinates (absolute)
- `o_x1, o_y1, o_x2, o_y2`: Object bounding box coordinates (absolute)

### Step 2: Postprocessing Implementation

```python
class PostProcess:
    def __init__(self, score_threshold=0.1, bbox_lambda=1.0, enable_softmax=False):
        self.score_threshold = score_threshold
        self.bbox_lambda = bbox_lambda
        self.enable_softmax = enable_softmax
    
    def __call__(self, outputs, target_sizes, hoi_mapper):
        """
        Convert model outputs to evaluation format
        
        Args:
            outputs: Dict containing 'pred_logits', 'pred_boxes', 'box_scores'
            target_sizes: Original image sizes for coordinate conversion
            hoi_mapper: Mapping from class indices to HOI IDs
        
        Returns:
            List of predictions in evaluation format
        """
        # Extract logits and boxes
        logits = outputs['pred_logits']  # [num_queries, num_hoi_classes]
        pred_boxes = outputs['pred_boxes']  # [num_queries, 8]
        box_scores = outputs['box_scores']  # [num_queries, 1]
        
        # Apply softmax or sigmoid
        if self.enable_softmax:
            scores = torch.softmax(logits, dim=-1)
        else:
            scores = torch.sigmoid(logits)
        
        # Combine with box confidence
        scores = scores * (box_scores ** self.bbox_lambda)
        
        # Convert to absolute coordinates
        h, w = target_sizes
        pred_boxes[:, [0, 2, 4, 6]] *= w  # x coordinates
        pred_boxes[:, [1, 3, 5, 7]] *= h  # y coordinates
        
        # Filter by threshold and format results
        results = []
        for i in range(len(scores)):
            for j, score in enumerate(scores[i]):
                if score > self.score_threshold:
                    hoi_id = hoi_mapper[j]
                    boxes = pred_boxes[i].tolist()
                    results.append([hoi_id, score.item()] + boxes)
        
        return results
```

### Step 3: Evaluator Classes

#### HICO-DET Evaluator

```python
import collections
import numpy as np

class HICOEvaluator:
    def __init__(self, anno_file, output_dir, zero_shot_type="rare", ignore_non_interaction=True):
        self.size = 600  # 600 HOI categories
        self.gts = self.load_anno(anno_file)
        self.scores = {i: [] for i in range(600)}
        self.boxes = {i: [] for i in range(600)}
        self.keys = {i: [] for i in range(600)}
        self.hico_ap = np.zeros(600)
        self.hico_rec = np.zeros(600)
        self.output_dir = output_dir
        self.zero_shot_type = zero_shot_type
        self.zero_shot_interaction_ids = self._get_zero_shot_ids(zero_shot_type)
        self.ignore_non_interaction = ignore_non_interaction
        
    def update(self, predictions):
        """Store predictions by HOI category"""
        for img_id, preds in predictions.items():
            for pred in preds:
                hoi_id = pred[0]
                score = pred[1]
                boxes = pred[2:]  # [h_x1, h_y1, h_x2, h_y2, o_x1, o_y1, o_x2, o_y2]
                self.scores[hoi_id].append(score)
                self.boxes[hoi_id].append(boxes)
                self.keys[hoi_id].append(img_id)
    
    def accumulate(self):
        """Compute AP for each HOI category"""
        for hoi_id in range(600):
            gts_per_hoi = self.gts[hoi_id]
            ap, rec = calc_ap(self.scores[hoi_id], self.boxes[hoi_id], 
                             self.keys[hoi_id], gts_per_hoi)
            self.hico_ap[hoi_id] = ap
            self.hico_rec[hoi_id] = rec
            
    def summarize(self):
        """Report metrics by category splits"""
        if self.ignore_non_interaction:
            valid_hois = np.setdiff1d(np.arange(600), NON_INTERACTION_IDS)
            seen_hois = np.setdiff1d(valid_hois, self.zero_shot_interaction_ids)
            zero_shot_hois = np.setdiff1d(self.zero_shot_interaction_ids, NON_INTERACTION_IDS)
        else:
            valid_hois = np.arange(600)
            seen_hois = np.setdiff1d(valid_hois, self.zero_shot_interaction_ids)
            zero_shot_hois = self.zero_shot_interaction_ids
            
        zero_shot_mAP = np.mean(self.hico_ap[zero_shot_hois])
        seen_mAP = np.mean(self.hico_ap[seen_hois])
        full_mAP = np.mean(self.hico_ap[valid_hois])
        
        print(f"zero-shot mAP: {zero_shot_mAP * 100:.2f}")
        print(f"seen mAP: {seen_mAP * 100:.2f}")
        print(f"full mAP: {full_mAP * 100:.2f}")
        
        return {
            'zero_shot_mAP': zero_shot_mAP,
            'seen_mAP': seen_mAP,
            'full_mAP': full_mAP
        }
    
    def load_anno(self, anno_file):
        """Load and process ground truth annotations"""
        with open(anno_file, "r") as f:
            dataset_dicts = json.load(f)
        
        # Initialize ground truth storage
        gts = {i: collections.defaultdict(list) for i in range(self.size)}
        
        # Process annotations
        for anno_dict in dataset_dicts:
            image_id = anno_dict["img_id"]
            box_annos = anno_dict.get("annotations", [])
            hoi_annos = anno_dict.get("hoi_annotation", [])
            
            for hoi in hoi_annos:
                person_box = box_annos[hoi["subject_id"]]["bbox"]
                object_box = box_annos[hoi["object_id"]]["bbox"]
                action_id = hoi["category_id"] - 1
                object_id = box_annos[hoi["object_id"]]["category_id"]
                hoi_id = self._get_hoi_id(action_id, object_id)
                gts[hoi_id][image_id].append(person_box + object_box)
        
        # Convert to numpy arrays
        for hoi_id in gts:
            for img_id in gts[hoi_id]:
                gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])
        
        return gts
```

#### SWIG-HOI Evaluator

```python
class SWiGEvaluator:
    def __init__(self, anno_file, output_dir):
        # Get evaluatable HOIs
        self.eval_hois = [x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1]
        size = max(self.eval_hois) + 1
        
        self.gts = self.load_anno(anno_file)
        self.scores = {i: [] for i in range(size)}
        self.boxes = {i: [] for i in range(size)}
        self.keys = {i: [] for i in range(size)}
        self.swig_ap = np.zeros(size)
        self.swig_rec = np.zeros(size)
        self.output_dir = output_dir
    
    def update(self, predictions):
        """Store predictions by HOI category"""
        for img_id, preds in predictions.items():
            for pred in preds:
                hoi_id = pred[0]
                score = pred[1]
                boxes = pred[2:]
                self.scores[hoi_id].append(score)
                self.boxes[hoi_id].append(boxes)
                self.keys[hoi_id].append(img_id)
    
    def accumulate(self):
        """Compute AP for each evaluatable HOI category"""
        for hoi_id in self.eval_hois:
            gts_per_hoi = self.gts[hoi_id]
            ap, rec = calc_ap(self.scores[hoi_id], self.boxes[hoi_id], 
                             self.keys[hoi_id], gts_per_hoi)
            self.swig_ap[hoi_id] = ap
            self.swig_rec[hoi_id] = rec
    
    def summarize(self):
        """Report metrics by frequency splits"""
        eval_hois = np.array([x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1])
        zero_hois = np.array([x["id"] for x in SWIG_INTERACTIONS 
                             if x["frequency"] == 0 and x["evaluation"] == 1])
        rare_hois = np.array([x["id"] for x in SWIG_INTERACTIONS 
                             if x["frequency"] == 1 and x["evaluation"] == 1])
        nonrare_hois = np.array([x["id"] for x in SWIG_INTERACTIONS 
                                if x["frequency"] == 2 and x["evaluation"] == 1])
        
        full_mAP = np.mean(self.swig_ap[eval_hois])
        zero_mAP = np.mean(self.swig_ap[zero_hois])
        rare_mAP = np.mean(self.swig_ap[rare_hois])
        nonrare_mAP = np.mean(self.swig_ap[nonrare_hois])
        
        print(f"zero-shot mAP: {zero_mAP * 100:.2f}")
        print(f"rare mAP: {rare_mAP * 100:.2f}")
        print(f"nonrare mAP: {nonrare_mAP * 100:.2f}")
        print(f"full mAP: {full_mAP * 100:.2f}")
        
        return {
            'zero_shot_mAP': zero_mAP,
            'rare_mAP': rare_mAP,
            'nonrare_mAP': nonrare_mAP,
            'full_mAP': full_mAP
        }
```

### Step 4: AP Calculation Core Algorithm

```python
def calc_ap(scores, boxes, keys, gt_boxes):
    """
    Calculate Average Precision using 11-point interpolation
    
    Args:
        scores: List of confidence scores
        boxes: List of predicted boxes [h_x1, h_y1, h_x2, h_y2, o_x1, o_y1, o_x2, o_y2]
        keys: List of image IDs
        gt_boxes: Dict of ground truth boxes {img_id: np.array of GT boxes}
    
    Returns:
        ap: Average Precision
        max_recall: Maximum recall achieved
    """
    if len(keys) == 0:
        return 0, 0
    
    if isinstance(boxes, list):
        scores, boxes, keys = np.array(scores), np.array(boxes), np.array(keys)
    
    # Sort by confidence scores (descending)
    idx = np.argsort(scores)[::-1]
    
    hit = []
    npos = sum(gt_boxes[key].shape[0] for key in gt_boxes.keys())
    used = {key: set() for key in gt_boxes.keys()}
    
    # Evaluate each prediction (limit to top 19999)
    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        box = boxes[pair_id]
        key = keys[pair_id]
        
        if key in gt_boxes:
            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx in range(gt_boxes[key].shape[0]):
                iou_score = calc_hit(box, gt_boxes[key][gt_idx])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = gt_idx
            
            # Check if match is valid (IoU >= 0.5 and not already matched)
            if best_gt_idx in used[key] or best_iou < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(best_gt_idx)
        else:
            hit.append(0)
    
    # Calculate precision and recall
    bottom = np.array(range(len(hit))) + 1
    hit = np.cumsum(hit)
    rec = hit / npos if npos > 0 else hit / (npos + 1e-8)
    prec = hit / bottom
    
    # 11-point interpolation for AP
    ap = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0
    
    return ap, np.max(rec) if len(rec) else 0


def calc_hit(det_box, gt_box):
    """
    Calculate hit score as minimum of human and object IoU
    
    Args:
        det_box: Predicted box [h_x1, h_y1, h_x2, h_y2, o_x1, o_y1, o_x2, o_y2]
        gt_box: Ground truth box [h_x1, h_y1, h_x2, h_y2, o_x1, o_y1, o_x2, o_y2]
    
    Returns:
        Hit score (minimum of human and object IoU)
    """
    gt_box = gt_box.astype(np.float64)
    human_iou = iou(det_box[:4], gt_box[:4])    # Human boxes
    object_iou = iou(det_box[4:], gt_box[4:])   # Object boxes
    return min(human_iou, object_iou)


def iou(bb1, bb2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        bb1, bb2: Bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    # Calculate box areas
    x1, y1 = bb1[2] - bb1[0], bb1[3] - bb1[1]
    x2, y2 = bb2[2] - bb2[0], bb2[3] - bb2[1]
    
    # Handle degenerate boxes
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(0, x2), max(0, y2)
    
    # Calculate intersection
    xiou = max(0, min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]))
    yiou = max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))
    
    intersection = xiou * yiou
    if intersection <= 0:
        return 0.0
    
    # Calculate union and IoU
    union = x1 * y1 + x2 * y2 - intersection
    return intersection / union if union > 0 else 0.0
```

## Key Implementation Details

### 1. Dataset Configuration

```python
# HICO-DET: 600 HOI categories
HICO_ZERO_SHOT_TYPES = {
    "rare": [list of rare HOI IDs],
    "non_rare": [list of non-rare HOI IDs], 
    "unseen_verb": [list of unseen verb HOI IDs],
    "unseen_object": [list of unseen object HOI IDs]
}

# SWIG-HOI: Variable number based on evaluation flag
SWIG_FREQUENCY_SPLITS = {
    0: "zero-shot (unseen)",
    1: "rare", 
    2: "non-rare (common)"
}
```

### 2. Ground Truth Format

```python
# Ground truth structure for both datasets
gts = {
    hoi_id: {
        image_id: np.array([
            [h_x1, h_y1, h_x2, h_y2, o_x1, o_y1, o_x2, o_y2],
            # ... more GT instances for this image and HOI
        ])
    }
}
```

### 3. Complete Evaluation Loop

```python
def evaluate_hoi_model(model, dataloader, evaluator, postprocessor):
    """Complete evaluation loop for HOI detection"""
    model.eval()
    
    with torch.no_grad():
        for images, targets in dataloader:
            # Forward pass
            outputs = model(images)
            
            # Post-process predictions
            results = {}
            for i, target in enumerate(targets):
                img_id = int(target['image_id'])
                pred_dict = {
                    'pred_logits': outputs['logits_per_hoi'][i],
                    'pred_boxes': outputs['pred_boxes'][i], 
                    'box_scores': outputs['box_scores'][i]
                }
                results[img_id] = postprocessor(
                    pred_dict, 
                    target['orig_size'],
                    dataloader.dataset.text_mapper
                )
            
            # Update evaluator
            evaluator.update(results)
    
    # Compute final metrics
    evaluator.accumulate()
    metrics = evaluator.summarize()
    return metrics
```

### 4. Critical Implementation Notes

- **Box Format**: Always use `[x1, y1, x2, y2]` format with absolute coordinates
- **Score Combination**: Multiply HOI classification scores with box confidence scores
- **Threshold**: Use `score_threshold=0.1` for filtering predictions
- **IoU Calculation**: Standard IoU with proper handling of degenerate boxes
- **Max Predictions**: Limit to top 19,999 predictions per HOI category
- **Coordinate Conversion**: Convert normalized coordinates to absolute coordinates
- **Memory Efficiency**: Process predictions in batches to avoid memory issues

## Usage Example

```python
# Initialize evaluator
evaluator = HICOEvaluator(
    anno_file="path/to/test_annotations.json",
    output_dir="path/to/output",
    zero_shot_type="rare",
    ignore_non_interaction=True
)

# Initialize postprocessor
postprocessor = PostProcess(
    score_threshold=0.1,
    bbox_lambda=2.0,
    enable_softmax=False
)

# Run evaluation
metrics = evaluate_hoi_model(model, test_dataloader, evaluator, postprocessor)
print(f"Full mAP: {metrics['full_mAP']:.4f}")
```

This framework provides the exact same evaluation methodology as the original INP-CC implementation, ensuring fair and consistent comparison when implementing HOI detection with Qwen2.5VL or other models.

---

## Dataset Format Differences and Handling

This section outlines the key differences between HICO-DET and SWIG-HOI dataset formats that are properly handled in the enhanced Qwen2.5VL evaluation pipeline.

### Dataset Format Comparison

#### 1. **Annotation Structure**

**HICO-DET Format**
```json
{
    "file_name": "HICO_test2015_00000001.jpg",
    "img_id": 1,
    "height": 480,
    "width": 640,
    "annotations": [  // Object annotations
        {
            "bbox": [x, y, w, h],
            "category_id": 1  // COCO object category
        }
    ],
    "hoi_annotation": [  // Singular, HOI annotations  
        {
            "subject_id": 0,   // Person box index
            "object_id": 1,    // Object box index
            "category_id": 76  // Action category (1-indexed)
        }
    ]
}
```

**SWIG-HOI Format**
```json
{
    "file_name": "flickr_123.jpg", 
    "img_id": 123,
    "height": 512,
    "width": 512,
    "box_annotations": [  // Different key name
        {
            "bbox": [x, y, w, h],
            "category_id": 0,          // SWIG object category  
            "aux_category_id": [1, 2]  // Additional object categories
        }
    ],
    "hoi_annotations": [  // Plural, HOI annotations
        {
            "subject_id": 0,    // Person box index
            "object_id": 1,     // Object box index  
            "action_id": 15     // SWIG action ID (0-indexed)
        }
    ]
}
```

#### 2. **Text Representation**

**HICO-DET Text Format**
```python
# From datasets/hico.py:prepare_dataset_text()
texts = []
for hoi in HICO_INTERACTIONS:
    action_name = " ".join(hoi["action"].split("_"))  # Convert underscore to space
    object_name = hoi["object"]
    text_pair = [action_name, object_name]  # List format
    texts.append(text_pair)

# Example: ["hold", "cup"], ["ride", "bicycle"]
```

**SWIG-HOI Text Format**  
```python
# From datasets/swig.py:generate_text()
def generate_text(action_id, object_id):
    action_name = SWIG_ACTIONS[action_id]["name"]
    object_name = SWIG_CATEGORIES[object_id]["name"] 
    return [action_name, object_name]  # Same list format

# Example: ["cut", "paper"], ["write", "paper"]
```

#### 3. **HOI ID Mapping**

**HICO-DET Mapping**
```python
# Direct interaction_id from HICO_INTERACTIONS
hoi_mapper = {(action_text, object_text): interaction_id}
text_mapper = {text_index: interaction_index} 

# 600 total interactions (0-599)
# Uses interaction_id directly for evaluation
```

**SWIG-HOI Mapping**
```python
# Complex mapping with evaluation flags
HOI_MAPPER = {(action_id, object_id): hoi["id"] for hoi in SWIG_INTERACTIONS}

# Only interactions with evaluation=1 are used
eval_interactions = [x for x in SWIG_INTERACTIONS if x["evaluation"] == 1]

# text_mapper maps to SWIG_INTERACTIONS index, not hoi["id"]
text_mapper = {text_index: swig_interaction_index}
```

#### 4. **Evaluation Filtering**

**HICO-DET**
- Uses all 600 interactions by default
- Can ignore non-interaction categories (`action_id == 57`)
- Zero-shot splits based on `zero_shot_type` parameter
- Direct mapping from text_mapper to interaction_id

**SWIG-HOI**  
- Only uses interactions where `evaluation == 1`
- Frequency-based splits: 0=zero-shot, 1=rare, 2=non-rare
- Complex mapping: text_mapper → SWIG index → actual HOI ID via HOI_MAPPER

#### 5. **Image Paths**

**HICO-DET**
```python
image_dir = data_root / "hico_20160224_det/images/test2015"
image_path = image_dir / annotation["file_name"]  # Direct usage
```

**SWIG-HOI**
```python  
image_dir = data_root / "swig_hoi/images_512"
image_path = image_dir / annotation["file_name"]  # Direct usage
```

### Enhanced Qwen2.5VL Dataset Handling

#### 1. **DatasetSpecificHandler Class**
```python
class DatasetSpecificHandler:
    def _setup_hico(self):
        # HICO-specific configuration
        self.hoi_annotation_key = "hoi_annotation"  # Singular
        self.box_annotation_key = "annotations"
        
    def _setup_swig(self):
        # SWIG-specific configuration  
        self.hoi_annotation_key = "hoi_annotations"  # Plural
        self.box_annotation_key = "box_annotations"
```

#### 2. **Text Mapping Consistency**
```python
# Both datasets now use same format: [action, object]
self.dataset_texts = []
self.text_mapper = {}

# HICO: Direct mapping
for i, hoi in enumerate(HICO_INTERACTIONS):
    action_name = " ".join(hoi["action"].split("_"))
    object_name = hoi["object"]
    text_pair = [action_name, object_name]
    self.text_mapper[len(self.dataset_texts)] = i

# SWIG: Evaluation filtering + mapping
for i, hoi in enumerate(SWIG_INTERACTIONS):
    if hoi.get("evaluation", 0) == 0:
        continue
    # Same text pair format as HICO
    text_pair = [action_name, object_name]
    self.text_mapper[len(self.dataset_texts)] = i
```

#### 3. **Enhanced Validation**
```python
# Dataset-specific format validation
if dataset == 'hico':
    required_keys = ['file_name', 'img_id', 'annotations', 'hoi_annotation']
elif dataset == 'swig':
    required_keys = ['file_name', 'img_id', 'box_annotations', 'hoi_annotations']

# Validate annotation structure matches expected format
if not all(key in sample_ann for key in required_keys):
    raise ValueError(f"Missing required keys for {dataset}: {required_keys}")
```

### Critical Implementation Details

#### 1. **HOI ID Resolution**
The enhanced system properly handles the different ID mapping schemes:

- **HICO**: `text_mapper[text_index]` → `HICO_INTERACTIONS[index]["interaction_id"]`
- **SWIG**: `text_mapper[text_index]` → `SWIG_INTERACTIONS[index]` → `HOI_MAPPER[(action_id, object_id)]`

#### 2. **Evaluation Compatibility**
Both datasets now feed into the same evaluator interface:
```python
# HICO evaluator expects: hoi_id (0-599)
# SWIG evaluator expects: hoi_id (SWIG-specific IDs with evaluation=1)
evaluator.update({image_id: [[hoi_id, score, h_x1, h_y1, h_x2, h_y2, o_x1, o_y1, o_x2, o_y2], ...]})
```

#### 3. **Prompt Engineering Adaptation**
```python
# Dataset-specific interaction lists
hico_interactions = ["hold cup", "ride bicycle", "eat sandwich", ...]
swig_interactions = ["cut paper", "write paper", "hold rope", ...]

# Adaptive prompting based on dataset
if dataset_type == "hico":
    # Use HICO-specific examples and categories
else:
    # Use SWIG-specific examples and categories
```

This enhanced handling ensures that Qwen2.5VL can properly work with both datasets while maintaining exact compatibility with the original INP-CC evaluation pipeline.