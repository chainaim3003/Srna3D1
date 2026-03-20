# Approach 2 BASIC — Complete Beginner's Guide

## Written for someone with introductory ML knowledge. No jargon unexplained.

---

# PART 1: CLEARING UP THE CONFUSION — What You Need Before Running

---

## The 3 Things You Need (and WHY each is different)

Think of our project like baking a cake. You need:
1. **The recipe** (the model code — RibonanzaNet's architecture)
2. **Pre-made frosting** (the pretrained weights — what RibonanzaNet already learned)
3. **Cake batter to practice with** (training data — RNA structures with known answers)

These are THREE separate downloads from THREE separate places. Let me explain each.

---

### THING 1: The Recipe — Clone the RibonanzaNet Repo

**What is it?**
RibonanzaNet is a neural network (a type of AI model) created by Shujun He at Texas A&M.
He published the CODE for this model on GitHub — the programming instructions that
define the model's architecture (how many layers, how they connect, etc.)

**Where to get it:**
```
https://github.com/Shujun-He/RibonanzaNet
```

**What "cloning" means:**
"Clone" just means "download a copy of the code folder." You run:
```bash
git clone https://github.com/Shujun-He/RibonanzaNet.git
```
This creates a folder on your computer called `RibonanzaNet/` with files like `Network.py`
(the actual neural network code), `config.yaml` (the settings), etc.

**What to set in config.yaml:**
In OUR config.yaml (the one in the BASIC/ folder), you update the path:
```yaml
backbone:
  repo_path: "C:/path/to/where/you/cloned/RibonanzaNet"
```

---

### THING 2: The Pre-Made Frosting — Download Pretrained Weights

**YOUR QUESTION: "What do you mean by Kaggle dataset? You said data comes from Ribonanza."**

Great question. Here's the key confusion: **Kaggle hosts BOTH competition data AND model weights,
but they are completely different things uploaded by different people.**

Think of Kaggle like Google Drive — anyone can upload files there. There are:
- **Competition data** = uploaded by the competition organizers (Das Lab at Stanford)
  This is the RNA sequences + their known 3D structures = the TRAINING DATA
  URL: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/data

- **Model weights** = uploaded by Shujun He (the RibonanzaNet creator)
  This is what the model LEARNED after being trained for weeks on powerful GPUs
  URL: https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights

**What are "weights"?**
A neural network is basically millions of numbers (called "weights" or "parameters").
When the model is trained, these numbers get adjusted until the model makes good predictions.

Imagine teaching a child to recognize dogs:
- BEFORE training: the child guesses randomly (weights are random numbers)
- AFTER training: the child recognizes dogs reliably (weights are tuned numbers)

The "weights file" is just a file containing those millions of tuned numbers.
Shujun He trained RibonanzaNet on 2 million RNA sequences for weeks on expensive GPUs.
He then SAVED those learned numbers into a file and uploaded it to Kaggle as a "dataset"
(Kaggle calls any uploaded file collection a "dataset," even if it's model weights).

**Why we need someone else's weights:**
Training RibonanzaNet from scratch would take weeks on expensive hardware.
By downloading Shujun He's weights, we get all that learning for free.
We "freeze" these weights (never change them) and just use RibonanzaNet as a
feature extractor — it reads RNA and tells us interesting things about it.

**Where to get it:**
```
https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights
```
Download the .pt file (a PyTorch weights file).

**What to set in config.yaml:**
```yaml
backbone:
  weights_path: "C:/path/to/downloaded/ribonanzanet_weights/best_model.pt"
```

---

### THING 3: The Cake Batter — Training Data

**YOUR QUESTION: "Forget the 130 CIF files. For the actual submission, what should be done?"**

Correct — the 130 CIF files you manually downloaded are a TINY subset (130 out of ~15,000+
structures). They're only useful for quick local testing. For a real submission:

**What you need: The competition's training data.**

The competition provides RNA structures where they KNOW the true 3D coordinates.
We use these to train our distance prediction head:
- Input: RNA sequence ("AUGCUUAGCG...")
- Known answer: the true (x, y, z) coordinates for every nucleotide

**RECOMMENDED PATH for actual submission:**

Use the **pre-processed pickle file** prepared by Shujun He:
```
https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle
```

**What is a "pickle file"?**
A pickle (.pkl) is Python's way of saving data to a file. Instead of a CSV or text file,
it saves Python objects (lists, dictionaries, arrays) directly. It's faster to load
than parsing thousands of individual structure files.

Shujun He took the raw competition data (thousands of CIF files), extracted all the
RNA sequences and their C1' coordinates, and saved everything into one convenient pickle.
This saves you HOURS of data processing.

**What to set in config.yaml:**
```yaml
data:
  train_pickle_path: "C:/path/to/downloaded/stanford3d_pickle/data.pkl"
```

**Alternative (harder but also works):**
Download the full 310 GB competition dataset from:
```
https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/data
```
This contains ~15,000 CIF files in the PDB_RNA/ folder. Our code can parse these
directly using BioPython (that's what `utils/pdb_parser.py` does), but it's MUCH
slower than loading the pre-processed pickle.

---

### Summary Table

| What | Where | Why | Size |
|------|-------|-----|------|
| RibonanzaNet code | github.com/Shujun-He/RibonanzaNet | The neural network architecture (the recipe) | ~10 MB |
| Pretrained weights | kaggle.com/datasets/shujun717/ribonanzanet-weights | What the model already learned (the frosting) | ~400 MB |
| Training data (pickle) | kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle | RNA structures with known answers (the batter) | ~1-2 GB |
| Test sequences | Competition page → Data tab | The RNAs we need to predict (the final exam) | Small |

---
---

# PART 2: THE COMPLETE PIPELINE — Step by Step, Simply

---

## THE BIG PICTURE: What Are We Actually Doing?

**The competition asks:** Given an RNA sequence (like "AUGCUUAGCG"), predict where
each nucleotide sits in 3D space — its (x, y, z) coordinates.

**Our approach (in plain English):**
1. We have a smart pre-trained AI (RibonanzaNet) that can read RNA sequences
   and understand relationships between nucleotides.
2. We attach a small "distance predictor" on top that estimates: "how far apart
   is nucleotide #3 from nucleotide #7? How about #3 from #12?" — for ALL pairs.
3. From those distances, we use math (not AI) to figure out the 3D coordinates.

**Analogy — The City Map Problem:**
Imagine someone tells you the driving distances between every pair of cities:
"New York to Boston = 215 miles, New York to DC = 225 miles, Boston to DC = 440 miles..."
From JUST the distances, could you draw a map showing where each city is?

YES! That's exactly what MDS (Multidimensional Scaling) does.
Our pipeline predicts the "distances" and then draws the "map."

---

## STEP-BY-STEP WALKTHROUGH

---

### Step 1: Tokenize the RNA Sequence

**File: `models/backbone.py` (the `tokenize_sequence` function)**

**What happens:**
The computer can't read letters. It needs numbers. So we convert:
```
A → 0
C → 1
G → 2
U → 3
```

The sequence "AUGC" becomes [0, 3, 2, 1].

**Why A=0, C=1, G=2, U=3?**
That's just alphabetical order (A, C, G, U). The official RibonanzaNet code uses
this mapping. We MUST use the same one, otherwise the pretrained weights won't work
(they were trained expecting A=0, C=1, etc.)

**Real-world example:**
Input: "AUGCUUAGCG" (10 nucleotides)
Output: [0, 3, 2, 1, 3, 3, 0, 2, 1, 2] (10 numbers)

---

### Step 2: Run Through Frozen RibonanzaNet Backbone

**File: `models/backbone.py` (the `OfficialBackboneWrapper`)**

**What happens:**
We feed the number sequence into RibonanzaNet. It processes the sequence through
9 transformer layers (like 9 stages of increasingly deep analysis) and produces
TWO outputs:

**Output 1 — Single Representation: (B, N, 256)**
For EACH nucleotide, the model produces 256 numbers that describe it.
These numbers encode things like:
- What base is it? (A, U, G, C)
- What's around it in the sequence?
- Is it in a region that likely forms a structure?
- How conserved is this position across evolution?

Think of it as a "profile" for each nucleotide — 256 numbers that capture
everything the model knows about that position.

For our 10-nucleotide example: we get a table of 10 rows × 256 columns.

**Output 2 — Pairwise Representation: (B, N, N, 64)**
For EACH PAIR of nucleotides, the model produces 64 numbers describing
their RELATIONSHIP.
These encode things like:
- Are nucleotides #3 and #7 likely to form a base pair?
- How far apart are they in the sequence?
- Do they co-evolve (change together across species)?

For our 10-nucleotide example: we get a 10×10 grid, and each cell has 64 numbers.
That's 10 × 10 × 64 = 6,400 numbers describing all pairwise relationships.

**What "frozen" means:**
We do NOT change any of RibonanzaNet's internal numbers during our training.
It's like using a calculator — we use it to get answers, but we don't open it
up and rewire it. The pretrained weights stay exactly as Shujun He trained them.

**What does (B, N, 256) mean?**
- B = Batch size = how many sequences we process at once (e.g., 4)
- N = Sequence length = how many nucleotides (e.g., 10)
- 256 = Feature dimension = how many numbers describe each nucleotide

So (4, 10, 256) means: 4 sequences, each 10 nucleotides long, each nucleotide
described by 256 numbers.

---

### Step 3: Predict Distances with the Distance Head

**File: `models/distance_head.py`**

**What happens:**
This is the ONLY part we actually train (teach from scratch).
It's a small neural network (an MLP — Multi-Layer Perceptron) that takes
the pairwise representation and predicts: "how far apart (in Ångströms)
are these two nucleotides in 3D space?"

**How it works internally:**
The pairwise representation gives us 64 numbers for each pair (i, j).
Our MLP processes these 64 numbers through 3 layers:

```
64 numbers in → Layer 1 (64→128) → Layer 2 (128→128) → Layer 3 (128→1) → 1 distance out
```

Each "layer" is just: multiply by a matrix of numbers, add a bias, then apply
a non-linear function (ReLU = "if negative, make it zero").

**What's Softplus?**
At the very end, we apply "Softplus" which is a smooth function that ensures
the output is always positive. Distances can't be negative!

**What's symmetrization?**
The distance from nucleotide #3 to #7 must equal the distance from #7 to #3.
So we average: dist[3,7] = (dist[3,7] + dist[7,3]) / 2

**Output:**
A symmetric N×N matrix where entry [i,j] = predicted distance in Ångströms
between nucleotide i and nucleotide j.

For our 10-nucleotide example: a 10×10 grid of distances.

**What's an Ångström?**
A unit of measurement used in molecular biology. 1 Ångström = 0.1 nanometers.
Atoms in a molecule are typically 1-10 Ångströms apart. Consecutive C1' atoms
in RNA are about 5.9 Ångströms apart.

---

### Step 4: Training — Teaching the Distance Head

**File: `train.py`**

**What happens:**
We show the model many examples of RNA structures where we KNOW the true
distances, and adjust the distance head's weights to make better predictions.

**The training loop (simplified):**

```
Repeat 100 times (100 "epochs"):
    For each batch of training examples:
        1. Get the RNA sequence and its KNOWN true coordinates
        2. Compute the TRUE distance matrix from the known coordinates
           (just measure the actual distance between every pair)
        3. Run the sequence through frozen RibonanzaNet → get features
        4. Run features through distance head → get PREDICTED distances
        5. Measure how WRONG the predictions are (the "loss")
        6. Adjust the distance head's weights slightly to reduce the error
```

**What is "loss"?**
Loss = a single number that measures how wrong the model is.
Lower loss = better predictions.

Our total loss has 3 parts:

**Part A — Distance MSE (Main Loss):**
For every pair of nucleotides, compute: (predicted_distance - true_distance)²
Then average all those squared errors.
This is called "Mean Squared Error" (MSE).

If the true distance is 10.0 Å and we predicted 12.0 Å:
Error = (12.0 - 10.0)² = 4.0

MSE penalizes big errors more than small ones (because of the squaring).

**Part B — Bond Constraint Loss:**
In real RNA, consecutive nucleotides (like position 3 and position 4) have
their C1' atoms about 5.9 Ångströms apart. This is a physical fact — the
chemical bonds constrain the distance.

So we add a penalty: if the model predicts a consecutive distance far from
5.9 Å, it gets punished. This teaches the model basic RNA physics.

**Part C — Clash Penalty Loss:**
In real molecules, two non-bonded atoms can't overlap — they physically
push each other apart. So no two C1' atoms should be closer than ~3.0 Å
(unless they're directly bonded neighbors).

If the model predicts two distant nucleotides at 1.5 Å apart, that's
physically impossible, so we penalize it.

**How does "adjusting weights" work?**
This is the "backpropagation" + "gradient descent" process:

1. The loss tells us HOW WRONG we are (a single number)
2. Backpropagation computes: "for each weight in the distance head,
   if I increased that weight by a tiny amount, would the loss go
   up or down, and by how much?" These are called "gradients."
3. Gradient descent then nudges each weight in the direction that
   makes the loss go DOWN.
4. The "learning rate" (0.0001) controls how big each nudge is.
   Too big = overshooting, too small = too slow.

After thousands of these nudges across all the training data, the
distance head learns to predict accurate pairwise distances.

**What is mixed precision (FP16)?**
Normally computers use 32 bits to store each number (FP32 = "full precision").
FP16 uses only 16 bits (half the memory). It's slightly less accurate but:
- Uses half the GPU memory → can process larger batches
- Runs roughly 2x faster
- The slight accuracy loss is negligible for training

**What is gradient clipping?**
Sometimes the gradients get absurdly large (the model "panics"). If a gradient
is huge, the weight update is huge, and the model can blow up (loss → infinity).
Gradient clipping says: "if any gradient is larger than 1.0, shrink it to 1.0."
It's like a speed limiter on a car.

---

### Step 5: Save the Best Model

**File: `train.py` (the checkpointing part)**

**What happens:**
After each epoch, we check the validation loss (loss on data the model
hasn't trained on). If it's the best we've seen, we save the model's
weights to a file called `best_model.pt`.

**Why validation loss, not training loss?**
Training loss measures how well the model fits the data it's seen.
Validation loss measures how well it generalizes to NEW data.

If training loss keeps going down but validation loss starts going UP,
the model is "overfitting" — memorizing the training data instead of
learning general patterns. Like a student who memorizes exam answers
but can't solve new problems.

---

### Step 6: Predict — Generate the Distance Matrix for Test Sequences

**File: `predict.py`**

**What happens:**
Now we have a trained distance head. For each test RNA sequence:
1. Tokenize it
2. Run through frozen RibonanzaNet → get features
3. Run through trained distance head → get predicted distance matrix

This gives us an N×N grid of predicted distances for each test RNA.

---

### Step 7: Reconstruct 3D Coordinates from Distances

**File: `models/reconstructor.py`**

**This is the "City Map Problem" — given only distances, find the coordinates.**

**Stage A — MDS (Multidimensional Scaling):**

The math is beautiful and simple (conceptually):

1. Take the N×N distance matrix D
2. Square every entry: D²
3. Apply "double centering": B = -½ × H × D² × H
   (where H = I - 1/N × ones_matrix)
   This converts distances into something related to positions.
4. Find the eigenvalues and eigenvectors of B
   (eigendecomposition — a standard linear algebra operation)
5. The top 3 eigenvectors (times square roots of their eigenvalues)
   give you the (x, y, z) coordinates!

**What are eigenvalues/eigenvectors?**
Think of a square matrix as a transformation that stretches and rotates space.
Eigenvectors are the "special directions" that only get stretched (not rotated).
Eigenvalues tell you HOW MUCH stretching happens in each direction.
The top 3 eigenvectors = the 3 most important directions = x, y, z axes.

**Why does this work?**
If you know all pairwise distances perfectly, MDS gives you the EXACT
coordinates (up to rotation/translation — you can't tell which way is
"north" from distances alone, but the shape is exactly right).

With noisy predicted distances, MDS gives you an APPROXIMATE shape.

**Stage B — Gradient Refinement:**

MDS gives a rough answer. We then polish it with gradient descent:

```
Start with MDS coordinates
Repeat 100 times:
    1. Compute current distances from current coordinates
    2. Measure error: how different are current distances from predicted distances?
    3. Also check: are consecutive nucleotides ~5.9 Å apart?
    4. Nudge each coordinate slightly to reduce these errors
```

This is like MDS giving you a rough sketch, then refinement sharpening
it into a detailed drawing.

---

### Step 8: Generate 5 Diverse Predictions

**File: `predict.py` (the diversity strategy)**

**Why 5 predictions?**
The competition lets you submit 5 predictions per target RNA.
Your score is the BEST of those 5 (measured by TM-score).

So we want 5 DIFFERENT predictions, hoping at least one is close to
the truth. It's like taking 5 shots at a target — more shots = better
chance of hitting near the bullseye.

**How we create diversity:**
We start from the SAME predicted distance matrix but reconstruct
differently each time:

- **Prediction 1:** Clean distances, 50 refinement steps (quick polish)
- **Prediction 2:** Clean distances, 100 refinement steps (thorough polish)
- **Prediction 3:** Add small random noise to distances, 100 steps
- **Prediction 4:** Add medium random noise to distances, 100 steps
- **Prediction 5:** Add larger random noise to distances, 150 steps

Adding noise to distances is like saying "maybe the true distance is
10.0 ± 0.5 Å" — each noisy version gives a slightly different 3D shape.
The more noise, the more different the shape.

---

### Step 9: Format and Save submission.csv

**File: `utils/submission.py`**

**What happens:**
The competition expects a CSV file with specific columns:
```
ID, resname, resid, x_1, y_1, z_1, x_2, y_2, z_2, ..., x_5, y_5, z_5
```

For each nucleotide in each test RNA:
- ID = target name + residue number (e.g., "R1107_1")
- resname = which base (A, U, G, C)
- resid = position number (1, 2, 3, ...)
- x_1, y_1, z_1 = coordinates from prediction #1
- x_2, y_2, z_2 = coordinates from prediction #2
- ... and so on for all 5 predictions

All coordinates are clipped to the range [-999.999, 9999.999] as required
by the competition rules.

---

### Step 10: Submit on Kaggle

For the actual competition submission, you need to:
1. Convert this code into a Kaggle Notebook
2. Upload the pretrained weights as a Kaggle Dataset
3. Upload the trained distance head weights as a Kaggle Dataset
4. The notebook must run on Kaggle's T4 GPU with no internet in under 8 hours
5. The notebook reads test_sequences.csv, predicts, and writes submission.csv

---
---

# PART 3: HOW THE FILES CONNECT — The Complete Flow

---

```
YOU DO THIS ONCE (SETUP):
    Download RibonanzaNet code  ──→  backbone.py reads this
    Download pretrained weights ──→  backbone.py loads these
    Download training pickle    ──→  dataset.py reads this

TRAINING (run once, takes hours):
    train.py
      ├─ Loads frozen backbone          (models/backbone.py)
      ├─ Creates distance head           (models/distance_head.py)
      ├─ Loads training data             (data/dataset.py + collate.py)
      ├─ For each epoch:
      │   ├─ For each batch:
      │   │   ├─ backbone(tokens) → features        [frozen, no training]
      │   │   ├─ distance_head(features) → distances [trainable]
      │   │   ├─ loss(predicted, true)               (losses/*.py)
      │   │   └─ update distance_head weights
      │   └─ Check validation loss
      └─ Saves best_model.pt to checkpoints/

PREDICTION (run for submission):
    predict.py
      ├─ Loads frozen backbone           (models/backbone.py)
      ├─ Loads trained distance head     (from best_model.pt)
      ├─ Loads test sequences            (utils/submission.py)
      ├─ For each test RNA:
      │   ├─ backbone(tokens) → features
      │   ├─ distance_head(features) → distance matrix
      │   ├─ MDS + refine → 3D coords (×5 diverse)  (models/reconstructor.py)
      │   └─ Collect 5 predictions
      └─ Writes submission.csv           (utils/submission.py)
```

---
---

# PART 4: KEY CONCEPTS GLOSSARY

---

| Term | Simple Explanation |
|------|-------------------|
| **Backbone** | The main AI model (RibonanzaNet) that reads RNA and produces features. Like a translator that converts RNA language into numbers. |
| **Head** | A small add-on neural network attached to the backbone. Our "distance head" takes the backbone's output and predicts distances. |
| **Frozen** | We lock the backbone's weights — they don't change during training. Only the head learns. |
| **Epoch** | One complete pass through ALL training data. 100 epochs = seeing every example 100 times. |
| **Batch** | A small group of examples processed together (e.g., 4 RNA sequences at a time). |
| **Loss** | A number measuring how wrong the model is. Training tries to make this smaller. |
| **MSE** | Mean Squared Error — average of (predicted - true)² across all predictions. |
| **Gradient** | The direction to nudge a weight to reduce the loss. |
| **Learning Rate** | How big each nudge is. 0.0001 = very small, careful steps. |
| **Overfitting** | When the model memorizes training data but fails on new data. Like studying only past exams. |
| **Checkpoint** | A saved snapshot of the model's weights at a particular point in training. |
| **Tensor** | A multi-dimensional array of numbers. A matrix is a 2D tensor. |
| **MDS** | Math technique to convert distances into coordinates. |
| **TM-score** | The competition's scoring metric (0-1). Higher = your predicted shape is more similar to the true shape. |
| **C1' atom** | A specific carbon atom in each nucleotide's sugar ring. Its position defines where the nucleotide "is" in 3D space. |
| **Ångström (Å)** | A tiny unit of length (0.1 nanometers). Atoms are ~1-10 Å apart. |
| **CIF file** | A file containing 3D coordinates of atoms in a molecule — like a blueprint. |
| **Pickle (.pkl)** | Python's way of saving data objects to a file for quick loading later. |
| **Pairwise** | Relating to PAIRS. Pairwise distances = distance between every possible pair. |
| **Symmetric** | dist[i,j] = dist[j,i]. Distance from A to B equals distance from B to A. |

---

# PART 5: WHAT TO DO RIGHT NOW (Action Checklist)

---

□ 1. Clone the RibonanzaNet repo:
      git clone https://github.com/Shujun-He/RibonanzaNet.git
      Put it somewhere accessible. Note the path.

□ 2. Download pretrained weights from Kaggle:
      https://www.kaggle.com/datasets/shujun717/ribonanzanet-weights
      (You need a Kaggle account. Click "Download" on that page.)
      Note where you save the .pt file.

□ 3. Download the training data pickle from Kaggle:
      https://www.kaggle.com/datasets/shujun717/stanford3d-dataprocessing-pickle
      Note where you save the pickle file.

□ 4. Update config.yaml with your actual paths:
      backbone.repo_path → path to cloned RibonanzaNet folder
      backbone.weights_path → path to downloaded .pt weights file
      data.train_pickle_path → path to downloaded pickle file

□ 5. Install Python dependencies:
      pip install -r requirements.txt

□ 6. Run a quick smoke test (optional but recommended):
      - Open a Python console
      - Try: import torch; print(torch.cuda.is_available())
      - If True → your GPU is ready

□ 7. Train:
      python train.py --config config.yaml

□ 8. Predict:
      python predict.py --config config.yaml --checkpoint checkpoints/best_model.pt --test_csv path/to/test_sequences.csv

□ 9. Submit submission.csv to Kaggle
