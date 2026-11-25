import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Matriz normalizada da RF (valores já calculados)
conf_matrix_normalized = np.array([[0.57, 0.43],
                                   [0.23, 0.77]])

# Labels ajustados
labels = ["Não Abaixo do CDI", "Abaixo do CDI"]

# Plot da matriz normalizada em azul
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.title("Matriz de Confusão Normalizada - RF 10d (Threshold 0.24)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()
