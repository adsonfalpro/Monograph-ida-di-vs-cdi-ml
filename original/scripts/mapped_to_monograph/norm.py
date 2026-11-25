import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Matriz original (valores absolutos)
conf_matrix = np.array([[562, 2],
                        [190, 61]])

# Normalização linha a linha
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Labels
labels = ["Não < CDI", "< CDI"]

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix_normalized, annot=True, fmt=".3f", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.title("Matriz de Confusão Normalizada - LSTM (Threshold 0.27)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()
