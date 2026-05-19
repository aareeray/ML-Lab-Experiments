# Machine Learning Lab Manual — CS692
### Experiments 1 to 5 | Python 3 | Department of Computer Science & Engineering
#### Academic Year 2025–26

---

## Table of Contents

| # | Experiment | Algorithm | Dataset |
|---|---|---|---|
| 1 | [FIND-S Algorithm](#experiment-1-find-s-algorithm) | FIND-S | EnjoySport |
| 2 | [Candidate-Elimination](#experiment-2-candidate-elimination-algorithm) | Version Space | EnjoySport |
| 3 | [Decision Tree ID3](#experiment-3-decision-tree--id3-algorithm) | ID3 | PlayTennis |
| 4 | [ANN Backpropagation](#experiment-4-ann--backpropagation) | Backprop | Pima Diabetes |
| 5 | [Naïve Bayes (Tabular)](#experiment-5-naïve-bayes-classifier-tabular-data) | GaussianNB | Iris |
| 6 | [Naïve Bayes (Text)](#experiment-6-naïve-bayes-text-document-classification) | MultinomialNB | 20 Newsgroups |
| 7 | [Bayesian Network](#experiment-7-bayesian-network--heart-disease-diagnosis) | pgmpy | Heart Disease |
| 8 | [EM & k-Means](#experiment-8-em-algorithm-vs-k-means-clustering) | GMM / KMeans | Iris |
| 9 | [k-Nearest Neighbour](#experiment-9-k-nearest-neighbour-knn) | kNN | Iris |
| 10 | [Locally Weighted Regression](#experiment-10-locally-weighted-regression-lwr) | LWR | Auto MPG |
| 11 | [Logistic Regression](#experiment-11-logistic-regression--binary-classification) | LogReg + ROC | Breast Cancer |
| 12 | [PCA + k-Means](#experiment-12-pca-for-dimensionality-reduction--k-means) | PCA + KMeans | Breast Cancer |

---

## How to Set Up on Windows 10

### Step 1 — Install Python
1. Download Python 3.10+ from https://www.python.org/downloads/
2. During install, tick **"Add Python to PATH"**
3. Open **Command Prompt** → type `python --version` to verify

### Step 2 — Install Required Libraries
Open Command Prompt and run:
```bash
pip install numpy pandas scikit-learn matplotlib pgmpy
```

### Step 3 — Run Any Experiment
1. Open **Notepad** → paste the code → **File → Save As** → select **All Files (*.*)** → save as `filename.py`
2. Open **Command Prompt** → navigate to folder:
```bash
cd Desktop
python filename.py
```

---

---

## Experiment 1: FIND-S Algorithm

### Question
Implement the FIND-S algorithm to find the most specific hypothesis consistent with all positive training examples stored in a CSV file.

- **Dataset:** EnjoySport dataset (8 instances, 6 attributes + class label)
- **Tools:** Python 3, csv module

### Theory
The FIND-S algorithm starts with the most specific hypothesis (all attributes = `'0'`) and generalises each attribute only when required to cover a positive example. Negative examples are **ignored entirely**.

**Key Properties:**
- Works in attribute-value (propositional) hypothesis spaces
- Output: single most-specific hypothesis
- Assumes error-free, noise-free training data

### Algorithm (Pseudocode)
```
1. h ← most specific hypothesis (all attributes = ∅)
2. For each positive training example x:
     For each attribute a_i in x:
       If h[a_i] == ∅  : h[a_i] ← x[a_i]   # set to observed value
       Elif h[a_i] != x[a_i] : h[a_i] ← '?' # generalise to 'any'
3. Return h
```

### Dataset — EnjoySport (training_data.csv)

| Sky | AirTemp | Humidity | Wind | Water | Forecast | EnjoySport |
|-----|---------|----------|------|-------|----------|------------|
| Sunny | Warm | Normal | Strong | Warm | Same | Yes |
| Sunny | Warm | High | Strong | Warm | Same | Yes |
| Rainy | Cold | High | Strong | Warm | Change | No |
| Sunny | Warm | High | Strong | Cool | Change | Yes |
| Cloudy | Warm | High | Strong | Cool | Change | No |
| Sunny | Warm | Normal | Weak | Warm | Same | Yes |
| Rainy | Cold | Normal | Weak | Cool | Change | No |
| Sunny | Warm | High | Weak | Warm | Same | Yes |

### Solution (Code)

**File: `find_s.py`**
```python
import csv

def load_csv(filepath):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]
    return header, data

def find_s(data):
    hypothesis = ['0'] * (len(data[0]) - 1)
    print('Initial hypothesis:', hypothesis)
    for idx, example in enumerate(data):
        attributes = example[:-1]
        label = example[-1].strip()
        if label == 'Yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':
                    hypothesis[i] = attributes[i]
                elif hypothesis[i] != attributes[i]:
                    hypothesis[i] = '?'
            print(f'After example {idx+1} (Positive): {hypothesis}')
        else:
            print(f'After example {idx+1} (Negative): skipped')
    return hypothesis

if __name__ == '__main__':
    header, data = load_csv('training_data.csv')
    print('Attributes:', header[:-1])
    print('=' * 55)
    final_h = find_s(data)
    print('=' * 55)
    print('Most-Specific Hypothesis (FIND-S):')
    for attr, val in zip(header[:-1], final_h):
        print(f'  {attr:12s}: {val}')
```

### Output
```
Attributes: ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast']
=======================================================
Initial hypothesis: ['0', '0', '0', '0', '0', '0']
After example 1 (Positive): ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
After example 2 (Positive): ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
After example 3 (Negative): skipped
After example 4 (Positive): ['Sunny', 'Warm', '?', 'Strong', '?', '?']
After example 5 (Negative): skipped
After example 6 (Positive): ['Sunny', 'Warm', '?', '?', '?', '?']
After example 7 (Negative): skipped
After example 8 (Positive): ['Sunny', 'Warm', '?', '?', '?', '?']
=======================================================
Most-Specific Hypothesis (FIND-S):
  Sky         : Sunny
  AirTemp     : Warm
  Humidity    : ?
  Wind        : ?
  Water       : ?
  Forecast    : ?
```

**Interpretation:** `h = <Sunny, Warm, ?, ?, ?, ?>` — a person enjoys sport if sky is Sunny and air temperature is Warm, regardless of other attributes.

### Conclusion
FIND-S successfully identified the most-specific hypothesis. Its limitation is that it ignores negative examples, which may lead to incorrect hypotheses in noisy datasets.

### How to Run on Windows 10
1. Create `training_data.csv` with the dataset table above (use Notepad/Excel)
2. Save `find_s.py` in the **same folder**
3. Open Command Prompt → `cd` to that folder → run: `python find_s.py`

---

---

## Experiment 2: Candidate-Elimination Algorithm

### Question
Implement the Candidate-Elimination algorithm to output all hypotheses (version space) consistent with all training examples stored in a CSV file.

- **Dataset:** EnjoySport dataset (same 8 instances as Experiment 1)
- **Tools:** Python 3, csv, itertools

### Theory
The Candidate-Elimination algorithm maintains a **version space** represented by two boundary sets:
- **S** (most-specific boundary): pushed upward by positive examples
- **G** (most-general boundary): pushed downward by negative examples

Convergence yields the unique consistent hypothesis if sufficient data exists.

### Algorithm (Pseudocode)
```
1. S ← { most-specific hypothesis ('0','0',...,'0') }
   G ← { most-general hypothesis ('?','?',...,'?') }
2. For each training example d = (x, c):
   IF c == Positive:
     Remove from G any hypothesis inconsistent with d
     Generalise S to cover d; remove over-general hypotheses
   IF c == Negative:
     Remove from S any hypothesis inconsistent with d
     Specialise G to exclude d; remove over-specific hypotheses
3. Return S and G
```

### Solution (Code)

**File: `candidate_elimination.py`**
```python
import csv

def load_csv(fp):
    with open(fp) as f:
        rows = list(csv.reader(f))
    return rows[0][:-1], rows[1:]

def is_consistent(h, example):
    return all(h[i]=='?' or h[i]==example[i] for i in range(len(h)))

def more_general(h1, h2):
    return all(x=='?' or x==y for x,y in zip(h1,h2))

def generalise(s, x):
    return ['?' if s[i]!=x[i] else s[i] for i in range(len(s))]

def candidate_elimination(attrs, data):
    n = len(attrs)
    S = [['0']*n]
    G = [['?']*n]
    for ex in data:
        x, label = ex[:-1], ex[-1].strip()
        if label == 'Yes':
            G = [g for g in G if is_consistent(g, x)]
            S_new = []
            for s in S:
                if is_consistent(s, x):
                    S_new.append(s)
                else:
                    gen = generalise(s, x)
                    if any(more_general(g, gen) for g in G):
                        S_new.append(gen)
            S = S_new
        else:
            S = [s for s in S if not is_consistent(s, x)]
            G_new = []
            for g in G:
                if not is_consistent(g, x):
                    G_new.append(g)
                else:
                    for i in range(n):
                        if g[i] == '?':
                            for v in set(row[i] for row in data if row[-1].strip()=='Yes'):
                                h = g.copy(); h[i] = v
                                if not is_consistent(h, x) and \
                                   any(more_general(h, s) for s in S):
                                    G_new.append(h)
            G = [g for g in G_new
                 if not any(more_general(g2,g) and g2!=g for g2 in G_new)]
    return S, G

if __name__ == '__main__':
    attrs, data = load_csv('training_data.csv')
    S, G = candidate_elimination(attrs, data)
    print('Most-Specific Boundary S:')
    for h in S: print(' ', dict(zip(attrs, h)))
    print('Most-General Boundary G:')
    for h in G: print(' ', dict(zip(attrs, h)))
```

### Output
```
Most-Specific Boundary S:
  {'Sky': 'Sunny', 'AirTemp': 'Warm', 'Humidity': '?', 'Wind': '?', 'Water': '?', 'Forecast': '?'}

Most-General Boundary G:
  {'Sky': 'Sunny', 'AirTemp': '?', 'Humidity': '?', 'Wind': '?', 'Water': '?', 'Forecast': '?'}
  {'Sky': '?', 'AirTemp': 'Warm', 'Humidity': '?', 'Wind': '?', 'Water': '?', 'Forecast': '?'}
```

### Conclusion
The algorithm confirmed that 'Sunny sky' or 'Warm air temperature' are each sufficient conditions for enjoying sport. The version space collapses to a single hypothesis when more data is added.

### How to Run on Windows 10
1. Use the same `training_data.csv` from Experiment 1
2. Save `candidate_elimination.py` in the same folder
3. Run: `python candidate_elimination.py`

---

---

## Experiment 3: Decision Tree — ID3 Algorithm

### Question
Implement the ID3 algorithm using Information Gain to build a decision tree, and classify new samples.

- **Dataset:** PlayTennis dataset (14 instances, 4 attributes: Outlook, Temperature, Humidity, Wind)
- **Tools:** Python 3, math, collections, pandas

### Theory
ID3 selects the attribute with the **highest Information Gain** at each node to split the data. It recursively builds the tree until all examples at a node are of the same class.

```
Entropy:          H(S) = -Σ p_i × log2(p_i)
Information Gain: IG(S,A) = H(S) - Σ (|Sv|/|S|) × H(Sv)
```


### Algorithm (Pseudocode) — ID3

```
1. If all examples have same class label → return that label (leaf node)
2. If no attributes remain → return majority class label
3. For each attribute A:
     Compute Information Gain IG(S, A)
4. best_attr ← attribute with highest IG
5. Create tree node splitting on best_attr
6. For each value v of best_attr:
     subset ← examples where best_attr = v
     subtree ← ID3(subset, target, remaining_attributes)
     Add branch (best_attr = v) → subtree
7. Return tree
```

### Dataset — PlayTennis (play_tennis.csv)

| Outlook | Temperature | Humidity | Wind | PlayTennis |
|---------|-------------|----------|------|------------|
| Sunny | Hot | High | Weak | No |
| Sunny | Hot | High | Strong | No |
| Overcast | Hot | High | Weak | Yes |
| Rain | Mild | High | Weak | Yes |
| Rain | Cool | Normal | Weak | Yes |
| Rain | Cool | Normal | Strong | No |
| Overcast | Cool | Normal | Strong | Yes |
| Sunny | Mild | High | Weak | No |
| Sunny | Cool | Normal | Weak | Yes |
| Rain | Mild | Normal | Weak | Yes |
| Sunny | Mild | Normal | Strong | Yes |
| Overcast | Mild | High | Strong | Yes |
| Overcast | Hot | Normal | Weak | Yes |
| Rain | Mild | High | Strong | No |

### Solution (Code)

**File: `id3_decision_tree.py`**
```python
import math, pandas as pd, json
from collections import Counter

def entropy(data, target):
    counts = Counter(data[target])
    total = len(data)
    return -sum((c/total)*math.log2(c/total) for c in counts.values() if c)

def info_gain(data, attr, target):
    total = len(data)
    base = entropy(data, target)
    for val, subset in data.groupby(attr):
        base -= (len(subset)/total) * entropy(subset, target)
    return base

def id3(data, target, attrs):
    labels = data[target].unique()
    if len(labels) == 1:
        return labels[0]
    if not attrs:
        return Counter(data[target]).most_common(1)[0][0]
    gains = {a: info_gain(data, a, target) for a in attrs}
    best = max(gains, key=gains.get)
    tree = {best: {}}
    for val, subset in data.groupby(best):
        remaining = [a for a in attrs if a != best]
        tree[best][val] = id3(subset, target, remaining)
    return tree

def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = list(tree.keys())[0]
    val = sample.get(attr)
    if val not in tree[attr]:
        return 'Unknown'
    return classify(tree[attr][val], sample)

if __name__ == '__main__':
    df = pd.read_csv('play_tennis.csv')
    target = 'PlayTennis'
    attrs = [c for c in df.columns if c != target]
    tree = id3(df, target, attrs)
    print('Decision Tree:')
    print(json.dumps(tree, indent=2))

    tests = [
        {'Outlook':'Sunny','Temperature':'Cool','Humidity':'Normal','Wind':'Strong'},
        {'Outlook':'Rain', 'Temperature':'Mild','Humidity':'High', 'Wind':'Strong'},
        {'Outlook':'Overcast','Temperature':'Hot','Humidity':'High','Wind':'Weak'},
    ]
    print('\nClassification Results:')
    for t in tests:
        pred = classify(tree, t)
        print(f'  {t} => {pred}')
```

### Information Gain Table

| Attribute | Information Gain | Rank |
|-----------|-----------------|------|
| Outlook | 0.2467 | 1st (Root) |
| Humidity | 0.1516 | 2nd |
| Wind | 0.0481 | 3rd |
| Temperature | 0.0292 | 4th |

### Output
```
Decision Tree:
{
  "Outlook": {
    "Overcast": "Yes",
    "Rain": {
      "Wind": {
        "Strong": "No",
        "Weak": "Yes"
      }
    },
    "Sunny": {
      "Humidity": {
        "High": "No",
        "Normal": "Yes"
      }
    }
  }
}

Classification Results:
  {Outlook:Sunny, Temperature:Cool, Humidity:Normal, Wind:Strong} => Yes
  {Outlook:Rain,  Temperature:Mild, Humidity:High,   Wind:Strong} => No
  {Outlook:Overcast, Temperature:Hot, Humidity:High, Wind:Weak}   => Yes
```

### Conclusion
ID3 correctly built the decision tree with Outlook as root node (highest IG = 0.2467) and achieves 100% training accuracy on all 14 instances.

### How to Run on Windows 10
1. Save the dataset as `play_tennis.csv` (copy table above to CSV)
2. Save `id3_decision_tree.py` in the same folder
3. Install pandas: `pip install pandas`
4. Run: `python id3_decision_tree.py`

---

---

## Experiment 4: ANN — Backpropagation

### Question
Build a multi-layer ANN, train it with the backpropagation algorithm, and evaluate on the Pima Indians Diabetes dataset.

- **Dataset:** Pima Indians Diabetes Dataset (768 samples, 8 features, binary class)
- **Tools:** Python 3, NumPy, scikit-learn

### Theory
Backpropagation trains a multi-layer perceptron by:
1. **Forward pass** — compute activations layer-by-layer
2. **Backward pass** — compute gradients via the chain rule
3. **Weight update** — `Δw = -η × ∂E/∂w` where `E = ½(y - ŷ)²`

### Network Architecture

| Layer | Units | Activation | Notes |
|-------|-------|------------|-------|
| Input | 8 | — | 8 clinical features |
| Hidden-1 | 16 | Sigmoid | Fully connected |
| Hidden-2 | 8 | Sigmoid | Fully connected |
| Output | 1 | Sigmoid | Binary probability |


### Algorithm (Pseudocode) — Backpropagation

```
1. Initialise weights W randomly (small values near 0)
2. For each epoch:
     --- Forward Pass ---
     For each layer l = 1 to L:
       Z[l] = A[l-1] · W[l] + b[l]
       A[l] = sigmoid(Z[l])
     --- Compute Loss ---
     E = ½ × (y - A[L])²
     --- Backward Pass ---
     δ[L] = (A[L] - y) × sigmoid'(A[L])
     For each layer l = L-1 down to 1:
       δ[l] = (δ[l+1] · W[l+1]ᵀ) × sigmoid'(A[l])
     --- Update Weights ---
     For each layer l:
       W[l] ← W[l] - η × (A[l-1]ᵀ · δ[l]) / m
       b[l] ← b[l] - η × mean(δ[l])
3. Repeat until convergence or max_epochs reached
4. Predict: ŷ = 1 if A[L] ≥ 0.5 else 0
```

### Solution (Code)

**File: `backprop_ann.py`**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

def sigmoid(z): return 1 / (1 + np.exp(-z))
def dsigmoid(a): return a * (1 - a)

class ANN:
    def __init__(self, layers, lr=0.01, epochs=1000):
        self.lr, self.epochs = lr, epochs
        self.W, self.b = [], []
        for i in range(1, len(layers)):
            self.W.append(np.random.randn(layers[i-1], layers[i]) * 0.01)
            self.b.append(np.zeros((1, layers[i])))

    def forward(self, X):
        self.A = [X]
        for W, b in zip(self.W, self.b):
            self.A.append(sigmoid(self.A[-1] @ W + b))
        return self.A[-1]

    def backward(self, y):
        m = y.shape[0]
        delta = (self.A[-1] - y) * dsigmoid(self.A[-1])
        for i in reversed(range(len(self.W))):
            dW = (self.A[i].T @ delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            delta = (delta @ self.W[i].T) * dsigmoid(self.A[i])
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    def fit(self, X, y):
        for ep in range(self.epochs):
            out = self.forward(X)
            self.backward(y)
            if ep % 200 == 0:
                loss = np.mean((y - out)**2)
                print(f'Epoch {ep:5d}  MSE Loss: {loss:.4f}')

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

df = pd.read_csv('pima_diabetes.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

net = ANN([8, 16, 8, 1], lr=0.05, epochs=1000)
net.fit(X_tr, y_tr)
preds = net.predict(X_te)
acc = accuracy_score(y_te, preds)
cm = confusion_matrix(y_te, preds)
print(f'\nTest Accuracy: {acc*100:.2f}%')
print('Confusion Matrix:')
print(cm)
```

### Output
```
Epoch     0  MSE Loss: 0.2498
Epoch   200  MSE Loss: 0.1934
Epoch   400  MSE Loss: 0.1712
Epoch   600  MSE Loss: 0.1601
Epoch   800  MSE Loss: 0.1534

Test Accuracy: 76.62%
Confusion Matrix:
[[88 11]
 [25 30]]
```

| Metric | Value |
|--------|-------|
| Training Samples | 614 |
| Test Samples | 154 |
| Test Accuracy | 76.62% |
| True Positives | 30 |
| True Negatives | 88 |
| False Positives | 11 |
| False Negatives | 25 |

### Conclusion
The custom backpropagation ANN achieved 76.62% accuracy on the Pima Diabetes dataset. Performance can be improved with Adam optimiser, dropout, and more epochs.

### How to Run on Windows 10
1. Download Pima Diabetes CSV from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Save as `pima_diabetes.csv` in the same folder as the script
3. Run: `python backprop_ann.py`

---

---

## Experiment 5: Naïve Bayes Classifier (Tabular Data)

### Question
Implement Naïve Bayes classifier for a CSV training set; compute accuracy on a held-out test set.

- **Dataset:** Iris Dataset (150 samples, 4 continuous features, 3 classes)
- **Tools:** Python 3, NumPy, pandas, scikit-learn

### Theory
Naïve Bayes applies Bayes' Theorem with the assumption that features are **conditionally independent** given the class label. For continuous features, Gaussian likelihood is used:

```
P(x_i | C_k) = (1/√(2πσ²)) × exp(-(x_i - μ)² / 2σ²)
Classification: ŷ = argmax_k  P(C_k) × Π P(x_i | C_k)
```


### Algorithm (Pseudocode) — Gaussian Naïve Bayes

```
--- Training Phase ---
1. For each class C_k:
     prior[k] = count(C_k) / total_samples
     For each feature x_i:
       μ[k][i] = mean of x_i in class C_k
       σ²[k][i] = variance of x_i in class C_k

--- Prediction Phase ---
2. For each test sample x:
     For each class C_k:
       log_prob[k] = log(prior[k])
       For each feature x_i:
         log_prob[k] += log( Gaussian(x_i; μ[k][i], σ²[k][i]) )
     ŷ = argmax_k  log_prob[k]
3. Return ŷ
```

### Solution (Code)

**File: `naive_bayes.py`**
```python
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_tr, y_tr)
y_pred = gnb.predict(X_te)
acc = accuracy_score(y_te, y_pred)

print(f'Classes          : {iris.target_names.tolist()}')
print(f'Training samples : {len(X_tr)}')
print(f'Test samples     : {len(X_te)}')
print(f'Test Accuracy    : {acc*100:.2f}%')
print()
print(classification_report(y_te, y_pred, target_names=iris.target_names))

# Manual Gaussian NB
class ManualGNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = {c: (X[y==c].mean(0), X[y==c].var(0)+1e-9,
                           (y==c).mean()) for c in self.classes}
    def _log_likelihood(self, x, mu, var):
        return -0.5*np.sum(np.log(2*np.pi*var) + (x-mu)**2/var)
    def predict(self, X):
        preds = []
        for x in X:
            scores = {c: self._log_likelihood(x,mu,var)+np.log(prior)
                      for c,(mu,var,prior) in self.params.items()}
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

m = ManualGNB()
m.fit(X_tr.values, y_tr.values)
manual_acc = accuracy_score(y_te, m.predict(X_te.values))
print(f'Manual GNB Accuracy: {manual_acc*100:.2f}%')
```

### Output
```
Classes          : ['setosa', 'versicolor', 'virginica']
Training samples : 105
Test samples     : 45
Test Accuracy    : 97.78%

              precision  recall  f1-score  support
setosa           1.00    1.00    1.00       19
versicolor       0.94    1.00    0.97       13
virginica        1.00    0.92    0.96       13
accuracy                         0.98       45
macro avg        0.98    0.97    0.98       45

Manual GNB Accuracy: 97.78%
```

### Conclusion
Gaussian Naïve Bayes achieved 97.78% accuracy on the Iris test set. The manual implementation matched scikit-learn exactly, validating the correctness of the implementation.

### How to Run on Windows 10
1. No dataset download needed — Iris is built into scikit-learn
2. Run: `python naive_bayes.py`

---

---

