# HOI Detection: mAP and Loss Calculation Analysis

## Case Study Analysis

**Specific Example (SWIG-HOI Dataset):**
- **Image 5**: 
  - **Prediction**: "kneading dough" (confidence: 0.900)
  - **Ground Truth**: "twisting dough"

**Question**: Is this prediction considered correct or wrong for mAP and loss calculations?

## Answer: **WRONG** - This is considered an incorrect prediction

**Dataset Context**: This example is from the **SWIG-HOI dataset**, where both "kneading dough" and "twisting dough" are valid interaction classes within the 5,539-class vocabulary. However, they represent semantically different actions and are treated as completely separate classes in evaluation.

**Important Clarification**: All categories (zero_shot, rare, nonrare) have ground truth labels in both datasets. Only exact HOI class ID matches count as correct (binary 0/1, no partial credit). "Zero_shot" means excluded from training, not unlabeled.

## Detailed Explanation

### 1. HOI Detection Evaluation Methodology

HOI (Human-Object Interaction) detection requires **exact matching** of:
1. **Action**: The specific action being performed
2. **Object**: The object being interacted with  
3. **Spatial Localization**: Bounding boxes for both human and object

### 2. Why "kneading dough" vs "twisting dough" is WRONG

#### **Action Mismatch**
- **Predicted Action**: "kneading" 
- **Ground Truth Action**: "twisting"
- **Result**: ❌ **Different actions = Wrong prediction**

Even though both interactions involve the same object ("dough"), they are **semantically different actions**:
- **Kneading**: Pressing and folding dough
- **Twisting**: Rotating or turning dough

In HOI detection, each unique `(action, object)` pair represents a **distinct interaction class**.

---

## Dataset-Specific mAP Calculation Methodology

### HICO-DET vs SWIG-HOI: Different Evaluation Approaches

**HICO-DET Dataset (600 classes):**
- **Seen vs Zero-shot Split**: Uses predefined rare interaction split
- **seen_mAP**: Average AP over frequent/common interactions (seen during training)
- **zero_shot_mAP**: Average AP over 120 rare interactions (unseen during training) 
- **full_mAP**: Overall average AP across all 600 valid interactions

**SWIG-HOI Dataset (5,539 classes):**
- **Frequency-based Split**: Uses training frequency to categorize interactions
- **nonrare_mAP**: Average AP over frequent interactions (`frequency == 2`)
- **rare_mAP**: Average AP over infrequent interactions (`frequency == 1`) 
- **zero_shot_mAP**: Average AP over completely unseen interactions (`frequency == 0`)
- **full_mAP**: Overall average AP across all evaluation interactions

### Step 1: Class-Specific Evaluation

Both datasets treat each `(action, object)` combination as a **separate class**:

**For our SWIG-HOI example:**
- Class A: "kneading dough" (HOI_ID = X, frequency = 1 or 2)
- Class B: "twisting dough" (HOI_ID = Y, frequency = 1 or 2) 
- These are **completely different classes** despite semantic similarity

### Step 2: Prediction Matching Process

For each image, the system attempts to match predictions to ground truth using:

1. **IoU Threshold**: Bounding box overlap (typically 0.5)
   - Human bounding box IoU > 0.5
   - Object bounding box IoU > 0.5

2. **Class Matching**: Exact HOI class match required
   - Predicted HOI_ID must equal Ground Truth HOI_ID

### Step 3: Classification Result for Image 5

```
Ground Truth: "twisting dough" (HOI_ID = Y)
Prediction:   "kneading dough" (HOI_ID = X, conf = 0.9)

Match Check:
- HOI_ID X ≠ HOI_ID Y  ❌
- Result: False Positive for "kneading dough" class
         False Negative for "twisting dough" class
```

### Step 4: mAP Computation

**Per-Class AP Calculation:**
```
Class "kneading dough" (HOI_ID = X):
- True Positives: 0 (no matching GT)
- False Positives: 1 (our incorrect prediction)
- AP = 0.0

Class "twisting dough" (HOI_ID = Y): 
- True Positives: 0 (no matching prediction)
- False Negatives: 1 (missed GT)
- AP = 0.0
```

**Dataset-Specific mAP Impact:**

**For SWIG-HOI:**
```python
# From swig_evaluator.py:44-50
rare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS 
                      if x["frequency"] == 1 and x["evaluation"] == 1])
nonrare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS 
                         if x["frequency"] == 2 and x["evaluation"] == 1])

rare_mAP = np.mean(self.swig_ap[rare_hois])      # Average AP for rare interactions
nonrare_mAP = np.mean(self.swig_ap[nonrare_hois]) # Average AP for frequent interactions
full_mAP = np.mean(self.swig_ap[eval_hois])      # Overall average
```

Both "kneading dough" and "twisting dough" likely fall into `rare_hois` category, so this error contributes 0.0 to the rare_mAP calculation, significantly hurting performance in the fine-grained evaluation.

---

## Loss Calculation Analysis

### Hungarian Matching Process

The loss calculation uses **bipartite matching** (Hungarian algorithm) to find optimal prediction-GT pairs:

```python
# Cost Matrix Calculation
cost_class = -predicted_prob[predicted_class]  # Classification cost
cost_bbox = L1_distance(pred_bbox, gt_bbox)    # Bounding box cost  
cost_giou = 1 - GIoU(pred_bbox, gt_bbox)      # Geometric IoU cost

total_cost = w1*cost_class + w2*cost_bbox + w3*cost_giou
```

### INP-CC Loss Scaling Framework

Before analyzing the specific loss components, it's crucial to understand **why loss scaling is necessary** in HOI detection:

#### Why Loss Scaling is Required

1. **Multi-Component Loss Balance**: HOI detection involves multiple loss types with vastly different scales:
   - Classification loss: Typically 0.1-10 range
   - Bounding box loss: Usually 0.01-1 range  
   - GIoU loss: Normalized to 0-2 range
   - Confidence loss: Similar to classification (0.1-10)

2. **Training Stability**: Without proper scaling, one loss component can dominate:
   ```python
   # Without scaling - classification dominates
   total_loss = 8.5 + 0.2 + 0.15 + 2.3 = 11.15 (85% from classification)
   
   # With INP-CC scaling weights
   total_loss = 5.0*8.5 + 5.0*0.2 + 2.0*0.15 + 10.0*2.3 = 66.3
   ```

3. **Gradient Balance**: Scaled losses ensure each component contributes meaningfully to gradients during backpropagation.

#### INP-CC Loss Weights (from training configuration)
```python
weight_dict = {
    'loss_ce': 5.0,      # Classification importance
    'loss_bbox': 5.0,    # Bounding box regression importance  
    'loss_giou': 2.0,    # Geometric IoU importance
    'loss_conf': 10.0    # Confidence calibration importance (highest)
}
```

**Key Insight**: `loss_conf` has the highest weight (10.0) because confidence calibration is critical for HOI detection - the model must learn when it's uncertain about complex interactions.

### Loss Components for Image 5

#### 1. Classification Loss (`loss_ce`)
```python
# Our example:
predicted_class = "kneading dough" (HOI_ID = X)
ground_truth_class = "twisting dough" (HOI_ID = Y)

# Cross-entropy loss:
loss_ce = -log(P(class_Y | prediction))
# Since model predicted class_X, P(class_Y) is very low
# Result: HIGH classification loss (~2-10)
```

#### 2. Bounding Box Loss (`loss_bbox`)
```python
# L1 loss between predicted and GT bounding boxes
pred_human_bbox = [x1, y1, x2, y2]  # From "kneading dough" prediction
pred_object_bbox = [x1, y1, x2, y2] # From "kneading dough" prediction

gt_human_bbox = [x1, y1, x2, y2]    # From "twisting dough" GT  
gt_object_bbox = [x1, y1, x2, y2]   # From "twisting dough" GT

# If bounding boxes are similar (same dough, same person):
loss_bbox = L1(pred_boxes, gt_boxes)  # Might be low (0.1-0.5)
```

#### 3. GIoU Loss (`loss_giou`) 
```python
# Geometric IoU loss
if bounding boxes overlap well:
    loss_giou = 1 - GIoU(pred_bbox, gt_bbox)  # Might be low (0.1-0.3)
else:
    loss_giou = high_value  # (0.8-1.0)
```

#### 4. Confidence Loss (`loss_conf`)
```python
# Binary cross-entropy for matched vs unmatched predictions
# Since this is a "wrong" match:
target_confidence = 0.0  # Should have low confidence
predicted_confidence = 0.9  # Model is very confident
loss_conf = BCE(0.9, 0.0)  # HIGH loss (~2.3)
```

### Final Loss Values for Image 5

#### Unscaled Losses (Raw Loss Values)
```
Estimated unscaled losses for this wrong prediction:
- loss_ce_unscaled: ~1.70    (Cross-entropy for wrong class)
- loss_bbox_unscaled: ~0.04  (L1 distance between similar boxes)  
- loss_giou_unscaled: ~0.075 (1 - GIoU for overlapping boxes)
- loss_conf_unscaled: ~0.23  (BCE for overconfident prediction)
```

#### Scaled Losses (INP-CC Training Losses)
```python
# Applied with INP-CC weight_dict scaling
scaled_losses = {
    'loss_ce': 5.0 * 1.70 = 8.50,      # HIGH - wrong class heavily penalized
    'loss_bbox': 5.0 * 0.04 = 0.20,    # LOW - boxes spatially similar
    'loss_giou': 2.0 * 0.075 = 0.15,   # LOW - good geometric overlap
    'loss_conf': 10.0 * 0.23 = 2.30,   # HIGH - overconfidence penalty
    'total_loss': 8.50 + 0.20 + 0.15 + 2.30 = 11.15
}
```

#### Loss Distribution Analysis
```
Total Loss Breakdown:
- Classification (loss_ce): 76.2% of total loss
- Confidence (loss_conf): 20.6% of total loss  
- Bounding box (loss_bbox): 1.8% of total loss
- GIoU (loss_giou): 1.3% of total loss
```

This distribution shows the model's main failures are in **semantic classification** and **confidence calibration**, not spatial localization.

---

## Key Insights

### 1. Semantic Precision Requirements
HOI detection requires **exact semantic matching**. Similar actions like:
- "holding cup" vs "drinking cup"
- "riding bicycle" vs "pushing bicycle"  
- "kneading dough" vs "twisting dough"

Are treated as **completely different classes**.

### 2. Loss vs mAP Behavior
- **mAP**: Binary (0 or 1) - either exact match or complete miss
- **Loss**: Continuous - reflects "how wrong" the prediction is
  - Wrong class but good boxes = moderate loss
  - Wrong class and bad boxes = high loss

### 3. Model Implications
This example shows the model has:
- **Good spatial understanding**: Likely detected correct human and dough regions (low bbox/giou losses)
- **Poor action recognition**: Cannot distinguish between similar dough manipulation actions (high classification loss)
- **Overconfidence issue**: Very confident (0.9) in wrong prediction (high confidence loss)
- **Loss scaling working correctly**: Classification and confidence dominate total loss, indicating proper penalty for semantic errors

### 4. Why Performance is Low
The Qwen2.5VL model struggles with:
- **Fine-grained action distinction**: Confuses similar actions
- **Limited HOI training**: Not specifically trained for HOI detection
- **Prompt engineering challenges**: Hard to describe subtle action differences

---

## Recommendations for Improvement

1. **Better Prompt Engineering**: Include more specific action descriptions
2. **Few-shot Examples**: Show examples of similar but different actions
3. **Confidence Calibration**: Reduce overconfidence in wrong predictions  
4. **Action Hierarchy**: Consider semantic similarity in evaluation metrics

---

## Technical Details

### Dataset Information
- **HICO-DET**: 600 HOI classes, 47,776 images
- **SWIG-HOI**: 5,539 HOI classes (evaluation on subset), 76,374 images
- **Evaluation Metric**: mAP @ IoU 0.5
- **Loss Weights**: Typically ce:5.0, bbox:5.0, giou:2.0, conf:10.0

### Code References
- **HICO mAP Calculation**: `datasets/hico_evaluator.py:47-66`
- **SWIG mAP Calculation**: `datasets/swig_evaluator.py:35-54`
- **Loss Computation**: `models/criterion.py:81-183` 
- **Hungarian Matching**: `models/matcher.py:53-144`
- **Loss Scaling Configuration**: Search for `weight_dict` in training scripts

---

## Open-Vocabulary HOI Detection Evaluation

### What is Open-Vocabulary in HOI Context?

**Traditional HOI Detection (Closed Vocabulary):**
- Model trained and evaluated only on predefined interaction classes
- HICO-DET: 600 classes, SWIG-HOI: 5,539 classes (subset for evaluation)
- Predictions must exactly match one of the training vocabulary classes

**Open-Vocabulary HOI Detection:**
- Model can predict interactions beyond the training vocabulary
- Uses natural language understanding to generalize to unseen action-object combinations
- Example: Model trained on "cutting apple" can potentially detect "slicing pear"

### How Would Our "Dough" Case Be Evaluated in Open-Vocabulary Setting?

#### Scenario Analysis

**If "kneading dough" vs "twisting dough" were open-vocabulary:**

1. **Semantic Similarity Evaluation**:
   ```python
   # Instead of exact class matching, use semantic similarity
   from sentence_transformers import SentenceTransformer
   
   predicted = "kneading dough"
   ground_truth = "twisting dough"
   
   # Compute semantic similarity score
   similarity = compute_semantic_similarity(predicted, ground_truth)
   # Result: ~0.6-0.7 (moderately similar due to same object, different actions)
   ```

2. **Flexible Matching Criteria**:
   - **Object Match**: ✅ Both involve "dough" 
   - **Action Similarity**: ⚠️ "kneading" vs "twisting" - related but distinct
   - **Overall Assessment**: Partial credit (0.3-0.5) instead of complete failure (0.0)

3. **Open-Vocabulary mAP Calculation**:
   ```python
   # Traditional: Binary matching (0 or 1)
   if predicted_hoi_id == ground_truth_hoi_id:
       match_score = 1.0
   else:
       match_score = 0.0
   
   # Open-vocabulary: Continuous similarity matching
   semantic_similarity = compute_similarity(predicted_text, gt_text)
   if semantic_similarity > threshold:  # e.g., 0.5
       match_score = semantic_similarity
   else:
       match_score = 0.0
   ```

#### Current Dataset Limitations

**HICO-DET and SWIG-HOI are NOT Open-Vocabulary:**
- Fixed vocabulary with exact class matching requirements
- No semantic similarity evaluation implemented
- "kneading dough" prediction would be:
  - **HICO**: Invalid (out-of-vocabulary) → Automatic 0.0
  - **SWIG**: Valid but wrong class → Automatic 0.0

### Open-Vocabulary Evaluation Methodology

#### 1. Text Encoder Approach
```python
# Encode both predicted and ground truth as text embeddings
pred_embedding = text_encoder("kneading dough")
gt_embedding = text_encoder("twisting dough")

# Compute cosine similarity
similarity = cosine_similarity(pred_embedding, gt_embedding)
```

#### 2. Compositional Understanding
```python
# Break down into action + object components
pred_action = "kneading"
pred_object = "dough"
gt_action = "twisting" 
gt_object = "dough"

# Compute component similarities
action_sim = compute_similarity(pred_action, gt_action)  # ~0.3
object_sim = compute_similarity(pred_object, gt_object)   # ~1.0

# Weighted combination
total_sim = 0.6 * action_sim + 0.4 * object_sim  # ~0.58
```

#### 3. Performance Implications

**Advantages of Open-Vocabulary Evaluation:**
- More realistic assessment of model capabilities
- Gives partial credit for semantically reasonable predictions
- Better reflects real-world deployment scenarios

**Challenges:**
- No standardized evaluation metrics yet
- Requires careful threshold tuning
- Computational overhead for similarity computation

### Research Directions

Current research in open-vocabulary HOI detection includes:
1. **Vision-Language Models**: Using CLIP-like models for flexible text-image matching
2. **Compositional Learning**: Learning action and object representations separately
3. **Few-shot Adaptation**: Quick adaptation to new interaction classes with minimal examples
4. **Semantic Evaluation Metrics**: Developing better metrics beyond exact matching

---

## Summary: Current vs Future Evaluation

**Current State (Our Case):**
- "kneading dough" vs "twisting dough" = **Complete failure (0.0 mAP)**
- Binary evaluation with no semantic understanding
- High loss penalty for semantically reasonable but technically wrong prediction

**Open-Vocabulary Future:**
- Same prediction could receive **partial credit (0.3-0.6 mAP)**
- Continuous evaluation with semantic similarity
- More nuanced loss functions that consider semantic relatedness

This analysis explains why similar-but-different predictions hurt both mAP scores and loss values significantly in current HOI detection tasks, and how future open-vocabulary approaches might provide more nuanced evaluation.