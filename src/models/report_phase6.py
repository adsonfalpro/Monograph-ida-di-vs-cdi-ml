"""Gera docs/model_specification.md a partir de outputs/metrics/phase6_training_summary.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ROOT  # noqa: E402


def ms(vals):
    v = np.array(vals, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f} (mín {v.min():.3f}, máx {v.max():.3f})" if len(v) > 1 else f"{v[0]:.3f}"


def main() -> None:
    s = json.loads((ROOT / "outputs" / "metrics" / "phase6_training_summary.json").read_text(encoding="utf-8"))
    d, per, rf, ls = s["data"], s["perceptron"], s["rf"], s["lstm"]
    L = ["# Especificação dos modelos e treino (Fase 6)\n",
         "Gerado por `src/models/report_phase6.py` a partir de `outputs/metrics/phase6_training_summary.json`. "
         "Código: `src/models/`. Testes: `tests/test_models.py`.\n",
         "**Escopo desta fase:** treinar com o bloco `fit` e usar `val` apenas para desenvolvimento (parada antecipada do LSTM e "
         "diagnóstico). Nenhuma matriz de confusão, threshold ou métrica de teste foi calculada: o bloco `test` está bloqueado "
         "(`Dataset.release_test` + `TestSetGuard`). Calibração e threshold: Fase 7.\n",
         "## 1. Dados de treino\n",
         f"`fit`: {d['n_fit']} amostras, {d['pos_fit']} positivos; `val`: {d['n_val']} amostras, {d['pos_val']} positivos. "
         f"Scaler ajustado nas linhas {d['scaler_rows'][0]}–{d['scaler_rows'][1]} (até a última decisão do `fit`). "
         f"Estatísticas de janela indefinidas fixadas em 0: {d['undefined_agg_stats']}.\n",
         "| | Perceptron | Random Forest | LSTM |\n|---|---|---|---|",
         "| Entrada | 120 estatísticas agregadas da janela de 100 dias | idem | sequência 100 × 12 |",
         "| Alvo | `Y_t` da Fase 3 | idem | idem |\n",
         "## 2. Parâmetros\n",
         "### Perceptron (`perceptron.py`)\n",
         "Camada única, degrau (`≥ 0 → 1`), pesos e viés 0, `learning_rate = 0,01`, `n_epochs = 100`, passagem em ordem aleatória por época "
         "(gerador próprio com semente 42, sem alterar o estado global do NumPy), regra `w ← w + η (y − ŷ) x`, sem pesos de classe. "
         "É fiel ao Apêndice C. Acrescenta `decision_function` (score contínuo) para ROC/PR.\n",
         "### Random Forest (`random_forest.py`)\n",
         "| parâmetro | valor |\n|---|---|"]
    for k, v in rf["params"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "Idêntico ao Apêndice A / `LS.py`. A calibração isotônica do original (`CalibratedClassifierCV(cv='prefit')`) passa para a Fase 7.\n",
          "### LSTM (`lstm_model.py`)\n",
          "`Input(100,12)` → `LSTM(64, return_sequences=True)` → `Dropout(0,3)` → `LSTM(32)` → `BatchNormalization` → `Dense(32, relu)` → "
          "`Dropout(0,2)` → `Dense(1, sigmoid)`; 32 897 parâmetros. Adam (1e‑3), `binary_crossentropy`, até 50 épocas, batch 32, `class_weight` "
          "balanceado, `EarlyStopping(val_loss, paciência 5, restaura pesos)`, `ReduceLROnPlateau(0,5; paciência 3)`. Igual ao Apêndice B.\n",
          "Diferenças de implementação (não metodológicas): `Input` explícito (Keras 3 não aceita `input_shape` na camada); semente via "
          "`keras.utils.set_random_seed` e `enable_op_determinism()` (CPU); **5 sementes (42–46)** em vez de uma só. "
          "Versões: TensorFlow 2.20, Keras 3.11, scikit‑learn 1.7.2, NumPy 2.3, pandas 2.3, Python 3.11 (a monografia só cita Python 3.9).\n",
          "Observação: a parada antecipada usa `val` (como no original). Isso torna as pontuações do LSTM em `val` levemente otimistas; "
          "a Fase 7 considera isso ao calibrar.\n",
          "## 3. Resultado do treino (diagnóstico, sem teste)\n",
          "| modelo | AUC em `fit` | AUC em `val` | observações |\n|---|---|---|---|",
          f"| Perceptron | {per['auc_fit']:.3f} | {per['auc_val']:.3f} | erros/época: {per['errors_first_epoch']} → {per['errors_last_epoch']}; prevê positivo em {per['val_pred_positive_rate']:.1%} de `val` |",
          f"| Random Forest (5 sementes) | {ms([r['auc_fit'] for r in rf['runs']])} | {ms([r['auc_val'] for r in rf['runs']])} | |",
          f"| LSTM (5 sementes) | {ms([r['auc_fit'] for r in ls['runs']])} | {ms([r['auc_val'] for r in ls['runs']])} | épocas executadas {ms([r['epochs_run'] for r in ls['runs']])}; melhor época {ms([r['best_epoch'] for r in ls['runs']])} |",
          "",
          "LSTM por semente:\n",
          "| semente | épocas | melhor época | melhor val_loss | AUC fit | AUC val |\n|---|---|---|---|---|---|"]
    for r in ls["runs"]:
        L.append(f"| {r['seed']} | {r['epochs_run']} | {r['best_epoch']} | {r['best_val_loss']:.4f} | {r['auc_fit']:.3f} | {r['auc_val']:.3f} |")
    L += ["",
          "## 4. Leitura (sem interpretar além do que os dados permitem)\n",
          "- **Sobreajuste forte** em RF e LSTM: AUC em `fit` próxima de 1 e muito menor em `val`. É esperado com apenas "
          f"{d['pos_fit']} positivos de treino e 120 atributos (ou 1200 entradas por janela). O LSTM atinge o melhor `val_loss` nas primeiras épocas.\n"
          "- A dispersão entre sementes do LSTM mede a incerteza de treino; os resultados da Fase 8 serão reportados com ela, não com uma semente escolhida.\n"
          "- Os AUC em `val` **não** são a medida final (a validação foi usada para parada antecipada e para o diagnóstico). Nenhum hiperparâmetro foi ajustado.\n"
          "- Não há razão para esperar que o ranking RF > Perceptron da monografia se mantenha; a comparação válida será feita no teste, na Fase 8.\n"]
    (ROOT / "docs" / "model_specification.md").write_text("\n".join(L), encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
