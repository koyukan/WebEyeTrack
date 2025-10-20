# Buffer Management Investigation Report
## WebEyeTrack Calibration Point Persistence Analysis

**Date:** 2025-10-20
**Status:** ⚠️ CRITICAL BUG CONFIRMED
**Priority:** HIGH - Affects calibration accuracy

---

## 1. Executive Summary

### Current Behavior
Calibration points and clickstream points are stored in a **unified buffer** with a **FIFO (First-In-First-Out) eviction policy** based on `maxPoints` limit. This causes **initial calibration points to be evicted** when clickstream points are added, degrading gaze accuracy over time.

### Intended Behavior
Initial calibration points should **persist for the entire application session** and always be included in model adaptation, while clickstream points should have a 60-second TTL and FIFO eviction among themselves only.

### Impact
- **Severity**: HIGH - Core functionality compromised
- **User Experience**: Calibration accuracy degrades after just 2-3 clicks
- **Scope**: JavaScript implementation confirmed; Python has different bug

---

## 2. Research Paper Context

**Paper**: WEBEYETRACK: Scalable Eye-Tracking for the Browser via On-Device Few-Shot Personalization
**arXiv**: https://arxiv.org/abs/2508.19544
**Section**: 3.3 - Few-Shot Personalization

### Key Insights from Implementation

Based on code analysis and documentation (CALIBRATION.md:272-273):

1. **Initial Few-Shot Calibration**: Uses 4 carefully collected calibration points with statistical filtering (mean-based outlier removal)
2. **Continuous Adaptation**: Clickstream points provide ongoing refinement
3. **MAML Training**: All points in buffer are used for Meta-Learning adaptation (WebEyeTrack.ts:389-392)

### Inferred Intent

The implementation suggests calibration points should be:
- **Permanent**: Collected with precision (crosshair focus, 25 samples, statistical filtering)
- **High Quality**: Represent ground truth for key screen locations
- **Always Present**: Used in every adaptation step to anchor the model

Click points should be:
- **Ephemeral**: Collected opportunistically from user interactions
- **Time-Limited**: 60-second TTL to reflect recent behavior
- **Supplementary**: Enhance calibration without replacing it

---

## 3. Python Implementation Analysis

**File**: `/python/webeyetrack/webeyetrack.py`

### Buffer Structure (Lines 213-218)

```python
self.calib_data = {
    'support_x': [],        # List of feature dictionaries
    'support_y': [],        # List of target gaze points
    'timestamps': [],       # Point creation timestamps
    'pt_type': []          # 'calib' or 'click'
}
```

### Configuration (Lines 154-156)

```python
@dataclass
class CalibConfig:
    max_points: int = 100        # Much larger buffer
    click_ttl: float = 60        # 60-second TTL for clicks
```

### Pruning Logic (Lines 220-238)

```python
def prune_calib_data(self):
    # Step 1: Apply maxPoints limit (FIFO)
    max_points = self.config.calib_config.max_points
    if len(self.calib_data['support_x']) > max_points:
        self.calib_data['support_x'] = self.calib_data['support_x'][-max_points:]
        self.calib_data['support_y'] = self.calib_data['support_y'][-max_points:]
        self.calib_data['timestamps'] = self.calib_data['timestamps'][-max_points:]
        self.calib_data['pt_type'] = self.calib_data['pt_type'][-max_points:]

    # Step 2: Apply TTL pruning for 'click' points
    current_time = time.time()
    ttl = self.config.calib_config.click_ttl
    for i in self.calib_data['timestamps']:  # ⚠️ BUG: iterates over values, not indices!
        if current_time - i > ttl and self.calib_data['pt_type'][i] == 'click':
            index = self.calib_data['timestamps'].index(i)
            self.calib_data['support_x'].pop(index)
            # ... (more pops)
```

### Python Issues

1. **TTL Bug**: Iterates over timestamp values instead of indices (line 232)
   - `for i in self.calib_data['timestamps']` gives timestamp values like `1234567890.123`
   - Then tries to use `i` as index: `self.calib_data['pt_type'][i]` → IndexError or wrong behavior

2. **Same FIFO Issue**: Even if TTL worked, maxPoints=100 still applies FIFO to ALL points
   - With enough clicks, calibration points can still be evicted
   - The large buffer (100) masks the problem in practice

### How Python Mitigates the Issue

- **Large Buffer (100)**: Takes 96 clicks to evict first calibration point
- **Real-world Usage**: Users rarely generate 96+ clicks rapidly
- **Result**: Bug exists but rarely manifests in practice

---

## 4. JavaScript Buffer Audit

**File**: `/js/src/WebEyeTrack.ts`

### Buffer Structure (Lines 72-82)

```typescript
public calibData: {
  supportX: SupportX[],           // Array of {eyePatches, headVectors, faceOrigins3D}
  supportY: tf.Tensor[],          // Target gaze points (labels)
  timestamps: number[],           // Point creation times
  ptType: ('calib' | 'click')[]   // Point type discriminator
} = {
  supportX: [],
  supportY: [],
  timestamps: [],
  ptType: ['calib']
};
```

### Configuration (Lines 85-86)

```typescript
public maxPoints: number = 5;       // ⚠️ VERY SMALL BUFFER
public clickTTL: number = 60;       // 60-second TTL for clicks
```

### Pruning Logic (Lines 179-229)

```typescript
pruneCalibData() {
  // ===== STEP 1: Apply maxPoints limit (FIFO) =====
  if (this.calibData.supportX.length > this.maxPoints) {
    // Dispose tensors that will be removed
    const itemsToRemove = this.calibData.supportX.slice(0, -this.maxPoints);
    itemsToRemove.forEach(item => {
      tf.dispose([item.eyePatches, item.headVectors, item.faceOrigins3D]);
    });

    const tensorsToRemove = this.calibData.supportY.slice(0, -this.maxPoints);
    tensorsToRemove.forEach(tensor => tf.dispose(tensor));

    // Slice arrays to keep only last maxPoints
    this.calibData.supportX = this.calibData.supportX.slice(-this.maxPoints);
    this.calibData.supportY = this.calibData.supportY.slice(-this.maxPoints);
    this.calibData.timestamps = this.calibData.timestamps.slice(-this.maxPoints);
    this.calibData.ptType = this.calibData.ptType.slice(-this.maxPoints);
  }

  // ===== STEP 2: Apply TTL pruning for 'click' points =====
  const currentTime = Date.now();
  const ttl = this.clickTTL * 1000;

  const indicesToKeep: number[] = [];
  const indicesToRemove: number[] = [];

  this.calibData.timestamps.forEach((timestamp, index) => {
    // Keep if: (1) not expired OR (2) not a click point
    if (currentTime - timestamp <= ttl || this.calibData.ptType[index] !== 'click') {
      indicesToKeep.push(index);
    } else {
      indicesToRemove.push(index);
    }
  });

  // Dispose and filter expired click points
  indicesToRemove.forEach(index => {
    const item = this.calibData.supportX[index];
    tf.dispose([item.eyePatches, item.headVectors, item.faceOrigins3D]);
    tf.dispose(this.calibData.supportY[index]);
  });

  this.calibData.supportX = indicesToKeep.map(index => this.calibData.supportX[index]);
  this.calibData.supportY = indicesToKeep.map(index => this.calibData.supportY[index]);
  this.calibData.timestamps = indicesToKeep.map(index => this.calibData.timestamps[index]);
  this.calibData.ptType = indicesToKeep.map(index => this.calibData.ptType[index]);
}
```

### How Points Are Added (Lines 362-381)

```typescript
adapt(..., ptType: 'calib' | 'click' = 'calib') {
  // Prune old data BEFORE adding new point
  this.pruneCalibData();  // ⚠️ This is where calibration points can be lost

  const opt = tf.train.adam(innerLR, 0.85, 0.9, 1e-8);

  try {
    let { supportX, supportY } = generateSupport(eyePatches, headVectors, faceOrigins3D, normPogs);

    // Append new point to buffer
    this.calibData.supportX.push(supportX);
    this.calibData.supportY.push(supportY);
    this.calibData.timestamps.push(Date.now());
    this.calibData.ptType.push(ptType);  // ⚠️ Only metadata tag, no special handling

    // ... training uses ALL points in buffer (lines 389-392)
  }
}
```

### How Points Are Used in Training (Lines 388-392)

```typescript
// Concatenate ALL points in buffer for training
if (this.calibData.supportX.length > 1) {
  tfEyePatches = tf.concat(this.calibData.supportX.map(s => s.eyePatches), 0);
  tfHeadVectors = tf.concat(this.calibData.supportX.map(s => s.headVectors), 0);
  tfFaceOrigins3D = tf.concat(this.calibData.supportX.map(s => s.faceOrigins3D), 0);
  tfSupportY = tf.concat(this.calibData.supportY, 0);
}
// ALL buffer points are used for MAML adaptation
```

### Critical Issues in JavaScript

1. **Two-Stage Pruning with Wrong Order**:
   - Stage 1: FIFO on ALL points (ignores `ptType`)
   - Stage 2: TTL on 'click' points only
   - **Problem**: Stage 1 can remove calibration points before Stage 2 runs

2. **Small Buffer Size**: maxPoints=5 vs Python's 100
   - Only 1 click buffer slot available after 4 calibration points
   - Problem manifests immediately in real usage

3. **Point Type is Metadata Only**: `ptType` field exists but doesn't affect eviction in Stage 1

---

## 5. Bug Confirmation

### Scenario: Rapid Clicking with maxPoints=5

**Initial State**: User completes 4-point calibration

```
Buffer: [C1, C2, C3, C4]
Length: 4
ptType: ['calib', 'calib', 'calib', 'calib']
```

**Click 1**: User clicks at normalized position (0.2, 0.3)

```
Before adapt(): [C1, C2, C3, C4]
pruneCalibData() called:
  - Length (4) <= maxPoints (5) → No FIFO pruning
  - No expired clicks → No TTL pruning
Add K1: [C1, C2, C3, C4, K1]
After adapt(): [C1, C2, C3, C4, K1]  ✅ All calibration points safe
```

**Click 2**: User clicks at normalized position (-0.1, 0.4) within 1 second

```
Before adapt(): [C1, C2, C3, C4, K1]
pruneCalibData() called:
  - Length (5) == maxPoints (5) → No FIFO pruning yet
  - K1 age < 60s → No TTL pruning
Add K2: [C1, C2, C3, C4, K1, K2]
Buffer grows to 6 elements temporarily
```

**Click 3**: User clicks at normalized position (0.3, -0.2) within 2 seconds

```
Before adapt(): [C1, C2, C3, C4, K1, K2]
pruneCalibData() called:
  ⚠️ STEP 1 (FIFO):
    - Length (6) > maxPoints (5)
    - slice(-5) → KEEPS: [C2, C3, C4, K1, K2]
    - REMOVES: [C1] ❌ CALIBRATION POINT LOST!
  - K1, K2 age < 60s → No TTL pruning
Add K3: [C2, C3, C4, K1, K2, K3]
After adapt(): [C2, C3, C4, K1, K2, K3]
Result: C1 calibration point permanently lost
```

**Click 4**: User clicks at normalized position (-0.3, -0.1) within 3 seconds

```
Before adapt(): [C2, C3, C4, K1, K2, K3]
pruneCalibData() called:
  ⚠️ STEP 1 (FIFO):
    - Length (6) > maxPoints (5)
    - slice(-5) → KEEPS: [C3, C4, K1, K2, K3]
    - REMOVES: [C2] ❌ SECOND CALIBRATION POINT LOST!
Add K4: [C3, C4, K1, K2, K3, K4]
After adapt(): [C3, C4, K1, K2, K3, K4]
Result: C1, C2 calibration points permanently lost
```

### Visual Timeline

```
Time      Event           Buffer State                        Calib Points
------------------------------------------------------------------------
T=0s      Initial Calib   [C1, C2, C3, C4]                   4 ✅
T=1s      Click 1         [C1, C2, C3, C4, K1]               4 ✅
T=2s      Click 2         [C1, C2, C3, C4, K1, K2]           4 ✅
T=3s      Click 3         [C2, C3, C4, K1, K2, K3]           3 ⚠️ (C1 lost)
T=4s      Click 4         [C3, C4, K1, K2, K3, K4]           2 ❌ (C1, C2 lost)
T=5s      Click 5         [C4, K1, K2, K3, K4, K5]           1 ❌ (C1, C2, C3 lost)
T=6s      Click 6         [K1, K2, K3, K4, K5, K6]           0 ❌ ALL LOST!
```

**Critical Finding**: After just **6 rapid clicks**, all calibration points are lost!

### Why TTL Pruning Doesn't Help

The TTL mechanism (Step 2) only helps after 60 seconds:

```
T=0s      Initial Calib   [C1, C2, C3, C4]
T=1s      Click 1         [C1, C2, C3, C4, K1]
T=62s     Click 2         Before prune: [C1, C2, C3, C4, K1]
                          After TTL:     [C1, C2, C3, C4] (K1 expired and removed)
                          Add K2:        [C1, C2, C3, C4, K2]
```

But if clicks happen rapidly (< 60s apart), TTL doesn't trigger and FIFO evicts calibration points.

---

## 6. Impact Assessment

### Calibration Quality Degradation

**Initial Calibration** (demo-app/CALIBRATION.md:108-115):
- 4 strategic screen positions (corners)
- 25 samples collected per point
- Statistical filtering (mean-based outlier removal)
- User focuses on crosshair with precision
- Training: stepsInner=10, innerLR=1e-4

**Click Calibration** (WebEyeTrack.ts:231-265):
- Opportunistic (wherever user clicks)
- Single sample (current gaze estimate)
- No statistical filtering
- User focus unknown (may not be looking at click location)
- Training: stepsInner=10, innerLR=1e-4

**Quality Comparison**:

| Aspect               | Initial Calibration | Click Calibration |
|---------------------|---------------------|-------------------|
| Ground truth quality| HIGH (user focused) | LOW (opportunistic) |
| Spatial coverage    | 4 corners (optimal) | Random (uncontrolled) |
| Statistical robustness | 25 samples filtered | 1 sample, no filter |
| User intention      | Explicit calibration | Incidental interaction |

### Accuracy Impact

Losing calibration points causes:

1. **Spatial Coverage Loss**: Corners no longer represented in training set
2. **Drift Amplification**: Click points may have systematic gaze errors that compound
3. **Model Instability**: Training on lower-quality data degrades learned parameters
4. **User Frustration**: Gaze accuracy inexplicably worsens over time

### Quantitative Estimate

Assuming gaze error is proportional to training data quality:

```
Scenario                    | Est. Gaze Error (cm) | Relative Accuracy
---------------------------|---------------------|-------------------
4 calib points              | 2.0                 | 100% (baseline)
3 calib + 2 clicks          | 2.4                 | 83%
2 calib + 3 clicks          | 3.1                 | 65%
1 calib + 4 clicks          | 4.2                 | 48%
0 calib + 5 clicks          | 6.0+                | 33% or worse
```

*These are estimates based on data quality principles; actual error would require empirical testing.*

### Real-World Usage Pattern

**Typical user session** (15 minutes):
- 1 initial calibration (4 points)
- ~30-50 clicks (browsing, interactions)
- With maxPoints=5, calibration lost after 2-6 clicks (< 1 minute)
- Remaining 14+ minutes: degraded accuracy

---

## 7. Recommended Solution

### Architecture: Separate Buffer Management

Implement **logically separate buffers** for calibration vs clickstream points:

```typescript
public calibData: {
  // === PERSISTENT CALIBRATION BUFFER (never evicted) ===
  calibSupportX: SupportX[],
  calibSupportY: tf.Tensor[],
  calibTimestamps: number[],

  // === TEMPORAL CLICKSTREAM BUFFER (TTL + FIFO) ===
  clickSupportX: SupportX[],
  clickSupportY: tf.Tensor[],
  clickTimestamps: number[],
}

// Separate configuration
public maxCalibPoints: number = 4;      // Fixed, never exceeded
public maxClickPoints: number = 5;      // FIFO within clicks only
public clickTTL: number = 60;           // TTL applies only to clicks
```

### Pruning Logic (Revised)

```typescript
pruneCalibData() {
  // === CALIBRATION BUFFER: No pruning, just enforce max ===
  if (this.calibData.calibSupportX.length > this.maxCalibPoints) {
    // Only enforce during add, never prune existing
    console.warn(`Calibration buffer full (${this.maxCalibPoints} points)`);
  }

  // === CLICKSTREAM BUFFER: TTL + FIFO ===
  const currentTime = Date.now();
  const ttl = this.clickTTL * 1000;

  // Step 1: Remove expired clicks
  const validIndices: number[] = [];
  this.calibData.clickTimestamps.forEach((timestamp, index) => {
    if (currentTime - timestamp <= ttl) {
      validIndices.push(index);
    } else {
      // Dispose expired tensors
      const item = this.calibData.clickSupportX[index];
      tf.dispose([item.eyePatches, item.headVectors, item.faceOrigins3D]);
      tf.dispose(this.calibData.clickSupportY[index]);
    }
  });

  // Filter to keep only valid (non-expired) clicks
  this.calibData.clickSupportX = validIndices.map(i => this.calibData.clickSupportX[i]);
  this.calibData.clickSupportY = validIndices.map(i => this.calibData.clickSupportY[i]);
  this.calibData.clickTimestamps = validIndices.map(i => this.calibData.clickTimestamps[i]);

  // Step 2: Apply FIFO if still over limit
  if (this.calibData.clickSupportX.length > this.maxClickPoints) {
    const itemsToRemove = this.calibData.clickSupportX.slice(0, -this.maxClickPoints);
    itemsToRemove.forEach(item => {
      tf.dispose([item.eyePatches, item.headVectors, item.faceOrigins3D]);
    });

    const tensorsToRemove = this.calibData.clickSupportY.slice(0, -this.maxClickPoints);
    tensorsToRemove.forEach(tensor => tf.dispose(tensor));

    this.calibData.clickSupportX = this.calibData.clickSupportX.slice(-this.maxClickPoints);
    this.calibData.clickSupportY = this.calibData.clickSupportY.slice(-this.maxClickPoints);
    this.calibData.clickTimestamps = this.calibData.clickTimestamps.slice(-this.maxClickPoints);
  }
}
```

### Adapt Method (Revised)

```typescript
adapt(
  eyePatches: ImageData[],
  headVectors: number[][],
  faceOrigins3D: number[][],
  normPogs: number[][],
  stepsInner: number = 5,
  innerLR: number = 1e-5,
  ptType: 'calib' | 'click' = 'calib'
) {
  // Prune clicks (calibration buffer never pruned)
  this.pruneCalibData();

  const opt = tf.train.adam(innerLR, 0.85, 0.9, 1e-8);

  try {
    let { supportX, supportY } = generateSupport(
      eyePatches, headVectors, faceOrigins3D, normPogs
    );

    // === ADD TO APPROPRIATE BUFFER ===
    if (ptType === 'calib') {
      if (this.calibData.calibSupportX.length < this.maxCalibPoints) {
        this.calibData.calibSupportX.push(supportX);
        this.calibData.calibSupportY.push(supportY);
        this.calibData.calibTimestamps.push(Date.now());
      } else {
        console.warn(`Calibration buffer full, ignoring new calibration point`);
        return; // Don't train if we couldn't add the calibration point
      }
    } else {
      this.calibData.clickSupportX.push(supportX);
      this.calibData.clickSupportY.push(supportY);
      this.calibData.clickTimestamps.push(Date.now());
    }

    // === CONCATENATE FROM BOTH BUFFERS FOR TRAINING ===
    let tfEyePatches: tf.Tensor;
    let tfHeadVectors: tf.Tensor;
    let tfFaceOrigins3D: tf.Tensor;
    let tfSupportY: tf.Tensor;

    const allSupportX = [...this.calibData.calibSupportX, ...this.calibData.clickSupportX];
    const allSupportY = [...this.calibData.calibSupportY, ...this.calibData.clickSupportY];

    if (allSupportX.length > 1) {
      tfEyePatches = tf.concat(allSupportX.map(s => s.eyePatches), 0);
      tfHeadVectors = tf.concat(allSupportX.map(s => s.headVectors), 0);
      tfFaceOrigins3D = tf.concat(allSupportX.map(s => s.faceOrigins3D), 0);
      tfSupportY = tf.concat(allSupportY, 0);
    } else {
      // Use current support only
      tfEyePatches = supportX.eyePatches;
      tfHeadVectors = supportX.headVectors;
      tfFaceOrigins3D = supportX.faceOrigins3D;
      tfSupportY = supportY;
    }

    // ... rest of training logic unchanged
  } finally {
    opt.dispose();
  }
}
```

### Configuration Parameters

```typescript
// Recommended values for browser memory constraints
public maxCalibPoints: number = 4;      // 4-point or 9-point calibration
public maxClickPoints: number = 5;      // Recent clickstream context
public clickTTL: number = 60;           // 60-second rolling window

// Total maximum points in memory: 4 + 5 = 9
// Python equivalent: 100 points (much more memory)
```

### Memory Impact Analysis

**Current Implementation** (maxPoints=5):
```
Per point memory:
  - eyePatches: 128 × 512 × 3 × 4 bytes (float32) = 786 KB
  - headVectors: 3 × 4 bytes = 12 bytes
  - faceOrigins3D: 3 × 4 bytes = 12 bytes
  - supportY: 2 × 4 bytes = 8 bytes
Total per point: ~786 KB

Current max: 5 points × 786 KB = 3.93 MB
```

**Proposed Implementation** (maxCalibPoints=4 + maxClickPoints=5):
```
Calibration: 4 points × 786 KB = 3.14 MB
Clicks:      5 points × 786 KB = 3.93 MB
Total max:   9 points × 786 KB = 7.07 MB

Memory increase: 7.07 - 3.93 = 3.14 MB (+80%)
```

**Is this acceptable?**
- Modern browsers: 2-4 GB RAM typical
- TensorFlow.js WebGL: ~50-200 MB for models
- Additional 3.14 MB: **Negligible (< 0.2% of typical RAM)**
- Trade-off: Small memory cost for **dramatically better accuracy**

### Migration Strategy

**Phase 1: Non-Breaking Addition**
1. Add new separate buffer fields
2. Implement new pruning logic
3. Feature flag: `useNewBufferManagement: boolean`
4. Default: `false` (old behavior)

**Phase 2: Testing**
1. Enable for beta testers
2. Measure accuracy improvement (calibration error tests)
3. Validate memory usage in real browsers

**Phase 3: Migration**
1. Set default to `true` (new behavior)
2. Add migration helper for existing calibration data
3. Update documentation

**Phase 4: Cleanup**
1. Remove old buffer fields
2. Remove feature flag
3. Simplify code

### API Changes

**WebEyeTrack Constructor** (backward compatible):
```typescript
constructor(
  maxPoints: number = 5,              // Deprecated, but still accepted
  clickTTL: number = 60,
  maxCalibPoints?: number,            // New: defaults to 4
  maxClickPoints?: number             // New: defaults to maxPoints
) {
  // If new params not provided, use old behavior for compatibility
  this.maxCalibPoints = maxCalibPoints ?? 4;
  this.maxClickPoints = maxClickPoints ?? maxPoints;
  this.clickTTL = clickTTL;
}
```

---

## 8. Implementation Checklist

### Code Changes Required

- [ ] **WebEyeTrack.ts: Buffer structure**
  - [ ] Add separate `calibSupportX`, `calibSupportY`, `calibTimestamps`
  - [ ] Add separate `clickSupportX`, `clickSupportY`, `clickTimestamps`
  - [ ] Add `maxCalibPoints` configuration
  - [ ] Update constructor to accept new parameters

- [ ] **WebEyeTrack.ts: pruneCalibData()**
  - [ ] Remove pruning logic for calibration buffer
  - [ ] Implement TTL pruning for clicks only
  - [ ] Implement FIFO pruning for clicks only (after TTL)

- [ ] **WebEyeTrack.ts: adapt()**
  - [ ] Route points to correct buffer based on `ptType`
  - [ ] Validate calibration buffer capacity before adding
  - [ ] Concatenate from both buffers for training

- [ ] **WebEyeTrack.ts: dispose()**
  - [ ] Update to dispose both buffer types
  - [ ] Ensure all tensors are cleaned up

### Testing Required

- [ ] **Unit Tests**
  - [ ] Test calibration buffer never evicts points
  - [ ] Test click buffer applies TTL correctly
  - [ ] Test click buffer applies FIFO after TTL
  - [ ] Test edge case: 0 calibration points
  - [ ] Test edge case: 0 click points
  - [ ] Test edge case: maxCalibPoints exceeded

- [ ] **Integration Tests**
  - [ ] Test 4-point calibration + rapid clicking
  - [ ] Test 9-point calibration + rapid clicking
  - [ ] Test click calibration with TTL expiration
  - [ ] Test memory cleanup (no leaks)

- [ ] **Accuracy Tests**
  - [ ] Measure calibration error before/after fix
  - [ ] Compare accuracy: old buffer vs new buffer
  - [ ] Test accuracy degradation over time (should be stable now)

### Documentation Updates

- [ ] **CALIBRATION.md**
  - [ ] Document new buffer architecture
  - [ ] Update configuration options table
  - [ ] Add migration guide

- [ ] **README.md**
  - [ ] Update API documentation
  - [ ] Add buffer management section
  - [ ] Document memory usage

- [ ] **Code Comments**
  - [ ] Add detailed comments to pruning logic
  - [ ] Document why calibration points are separate
  - [ ] Add warnings about memory usage

### Python Implementation Fix

**File**: `/python/webeyetrack/webeyetrack.py`

The Python implementation also needs fixing (lines 220-238):

```python
# Current (BROKEN):
for i in self.calib_data['timestamps']:  # BUG: i is timestamp value, not index!
    if current_time - i > ttl and self.calib_data['pt_type'][i] == 'click':
        # ...

# Fixed:
indices_to_remove = []
for idx, timestamp in enumerate(self.calib_data['timestamps']):
    if current_time - timestamp > ttl and self.calib_data['pt_type'][idx] == 'click':
        indices_to_remove.append(idx)

# Remove in reverse order to avoid index shifting
for idx in reversed(indices_to_remove):
    self.calib_data['support_x'].pop(idx)
    self.calib_data['support_y'].pop(idx)
    self.calib_data['timestamps'].pop(idx)
    self.calib_data['pt_type'].pop(idx)
```

**Python should also adopt separate buffers** for consistency and clarity.

---

## 9. Alternative Solutions Considered

### Alternative 1: Priority-Based Eviction

Keep single buffer but evict 'click' points before 'calib' points.

```typescript
pruneCalibData() {
  if (this.calibData.supportX.length > this.maxPoints) {
    // First, try to remove oldest click points
    let clickIndices = this.calibData.ptType
      .map((type, idx) => type === 'click' ? idx : -1)
      .filter(idx => idx !== -1);

    if (clickIndices.length > 0) {
      // Remove oldest click
      const removeIdx = clickIndices[0];
      // ... remove and shift arrays
    } else {
      // No clicks to remove, evict oldest calibration point
      // ... existing FIFO logic
    }
  }
}
```

**Pros**:
- Minimal code changes
- Preserves calibration points longer

**Cons**:
- Complex array manipulation
- Still allows calibration eviction if no clicks exist
- Doesn't solve TTL expiration handling
- Less clear intent than separate buffers

**Decision**: Rejected in favor of separate buffers for clarity and robustness.

### Alternative 2: Circular Buffer with Metadata

Use circular buffer with metadata tags and custom eviction.

**Pros**:
- Memory-efficient (fixed allocation)
- Fast eviction (O(1))

**Cons**:
- More complex implementation
- Harder to debug
- Doesn't align with paper's conceptual model
- Over-engineering for problem size

**Decision**: Rejected. Simplicity and clarity are more valuable.

### Alternative 3: Increase maxPoints to Match Python

Simply increase JavaScript maxPoints from 5 to 100.

**Pros**:
- Trivial change (one line)
- Matches Python behavior

**Cons**:
- Doesn't fix the fundamental issue
- Memory usage increases 20x (78.6 MB)
- Still allows eventual eviction
- Masks problem rather than solving it

**Decision**: Rejected. This is a band-aid, not a fix.

---

## 10. Conclusion

### Summary of Findings

1. **Bug Confirmed**: JavaScript implementation loses calibration points after 2-6 rapid clicks
2. **Root Cause**: FIFO eviction applies to all points regardless of type
3. **Impact**: High - degrades calibration accuracy significantly
4. **Python Status**: Has different bug (TTL loop), but larger buffer masks issue
5. **Solution**: Separate buffers for calibration (persistent) vs clicks (ephemeral)

### Recommended Actions

**Immediate** (Priority: HIGH):
1. Implement separate buffer architecture in JavaScript
2. Fix Python TTL iteration bug
3. Add comprehensive tests

**Short-term** (Priority: MEDIUM):
1. Conduct accuracy testing to quantify improvement
2. Update documentation
3. Add migration path for existing users

**Long-term** (Priority: LOW):
1. Consider adaptive buffer sizing based on available memory
2. Add buffer state visualization in demo app
3. Publish findings as technical note

### Success Metrics

- **Functional**: Calibration points never evicted during normal usage
- **Performance**: Memory usage remains < 10 MB for buffer
- **Accuracy**: Gaze error stable over 15+ minute sessions
- **User Experience**: No degradation complaints after update

---

## References

1. **Research Paper**: WEBEYETRACK: Scalable Eye-Tracking for the Browser via On-Device Few-Shot Personalization
   - URL: https://arxiv.org/abs/2508.19544
   - Section 3.3: Few-Shot Personalization

2. **Implementation Files**:
   - JavaScript: `/js/src/WebEyeTrack.ts` (lines 179-229, 352-463)
   - Python: `/python/webeyetrack/webeyetrack.py` (lines 220-238, 319-427)
   - Demo: `/python/demo/main.py` (lines 158-277)

3. **Documentation**:
   - `/js/examples/demo-app/CALIBRATION.md`
   - `/js/README.md`
   - `/python/README.md`

4. **Related Concepts**:
   - MAML (Model-Agnostic Meta-Learning): https://arxiv.org/pdf/1703.03400.pdf
   - Few-Shot Learning for Personalization
   - WebGL Memory Management in TensorFlow.js

---

**Report Status**: ✅ Complete
**Next Steps**: Present to team, prioritize implementation
**Contact**: See GitHub issues for discussion
