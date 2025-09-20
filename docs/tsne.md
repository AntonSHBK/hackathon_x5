## 🔹 Какие эмбеддинги доступны в NER-модели (BERT/DeBERTa + CRF)

1. **Token embeddings (после encoder)**

   * Это выход `last_hidden_state` → матрица `[batch, seq_len, hidden_size]`.
   * Каждому токену соответствует вектор размерности `hidden_size` (например, 768).
   * Их можно взять для анализа:

     * сравнивать, как токены разных сущностей "разошлись" в пространстве,
     * смотреть, кластеризуются ли токены одной сущности (например, все бренды ближе друг к другу).

2. **Entity embeddings (усреднённые по сущности)**

   * Можно взять токены, которые принадлежат одной сущности (например, "простоквашино" → 3 сабворда),
   * усреднить их эмбеддинги → получить вектор сущности.
   * Эти векторы можно визуализировать → и посмотреть, как расположены `BRAND`, `TYPE`, `PERCENT` и т.д.

3. **\[CLS] embedding (или pooled output)**

   * Обычно используется в задачах классификации, но для NER не так информативен.

4. **Logits (до CRF)**

   * Можно взять `emissions` из `classifier` (размер `[batch, seq_len, num_labels]`).
   * Это по сути "как модель видит вероятность каждой метки".
   * Можно снизить размерность и посмотреть, как распределяются токены разных классов.

---

## 🔹 Что можно визуализировать

1. **Сходство классов**

   * Например, собрать векторы токенов с меткой `B-BRAND`, `I-BRAND`, `O` и т.д.
   * Снизить размерность (UMAP / t-SNE / PCA) и построить scatter plot.
   * Мы увидим, есть ли "кластеризация" по меткам.

2. **Сходство сущностей внутри класса**

   * Например, бренды разных товаров → можно увидеть, формируют ли они под-кластеры.

3. **Ошибки модели**

   * Можно отобразить эмбеддинги токенов, которые модель классифицировала неправильно → и понять, к каким классам они ближе.

---

## 🔹 Как визуализировать (пример с Plotly)

```python
import torch
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

def visualize_embeddings(dataset, model, tokenizer, idx2label, max_samples=1000):
    all_embeddings = []
    all_labels = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model.bert(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            hidden_states = outputs.last_hidden_state.cpu().numpy()  # [batch, seq_len, hidden_size]
            labels = batch["labels"].cpu().numpy()

            for hs, lbls in zip(hidden_states, labels):
                for vec, lbl in zip(hs, lbls):
                    if lbl == -100:
                        continue
                    all_embeddings.append(vec)
                    all_labels.append(idx2label[int(lbl)])
            if len(all_embeddings) > max_samples:
                break

    X = np.array(all_embeddings[:max_samples])
    y = np.array(all_labels[:max_samples])

    # PCA → 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig = px.scatter(
        x=X_2d[:, 0], y=X_2d[:, 1],
        color=y,
        title="Token embeddings visualization",
        labels={"color": "Entity"}
    )
    fig.show()
```
