# comparative_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# --- Define your models and their metrics ---
# Replace these with your actual model results

models = ["Simple CNN", "ResNet50 Transfer", "Final Model (Ours)"]

# Example metrics (you can update with actual evaluation results)
accuracy = [0.32, 0.55, 0.76]
precision = [0.30, 0.53, 0.74]
recall = [0.28, 0.50, 0.72]
f1_score = [0.29, 0.51, 0.73]
comments = [
    "Poor on minor classes",
    "Much better performance",
    "Best overall, still struggles with some classes"
]

# --- Create a DataFrame for tabular display ---
df = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1_score,
    "Comments": comments
})

print("📊 Comparative Analysis Table:\n")
print(df.to_string(index=False))

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10,6))

x = np.arange(len(models))
width = 0.2

ax.bar(x - width, accuracy, width, label="Accuracy", color="#4CAF50")
ax.bar(x, precision, width, label="Precision", color="#2196F3")
ax.bar(x + width, recall, width, label="Recall", color="#FF9800")
ax.bar(x + 2*width, f1_score, width, label="F1-Score", color="#9C27B0")

ax.set_ylabel("Score")
ax.set_title("Comparative Analysis of Different Models")
ax.set_xticks(x + width/2)
ax.set_xticklabels(models)
ax.set_ylim(0,1)
ax.legend()

plt.tight_layout()
plt.show()
