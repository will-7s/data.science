# Changelog — EDA Dashboard

## v6.1.0 — Bug-fix & optimisation release

### 🔴 Bugs critiques

| # | Fichier | Problème | Correction |
|---|---------|----------|------------|
| 1 | `pca.py` · `run_pca()` | `n_optimal` biaisé : branche `else: elbow = 1` introduisait une valeur aberrante dans le vote médian quand `n_comp < 3`, affichant "1 PC recommandé" à tort | Guard `if n_comp >= 3` ; fallback `min(2, n_comp)` |
| 2 | `clustering.py` · `_get_X()` | Ne retournait pas le `valid_mask` des lignes NaN supprimées → `labels[n_clean]` sans lien avec l'index original, annotations individuelles incorrectes | Retourne `(X, valid_mask)` ; `valid_mask` exposé dans le résultat |

### 🟠 Bugs majeurs

| # | Fichier | Problème | Correction |
|---|---------|----------|------------|
| 3 | `clustering.py` | Elbow + silhouette : 2 boucles KMeans séparées = 14 fits | Fusionné en `_run_diagnostics()` — 7 fits, 2× plus rapide |
| 4 | `callbacks.py` · `on_upload` | PCA calculé de façon synchrone : status message bloqué 2–5 s sur fichiers larges | Split en 2 callbacks : `on_upload_parse` (fast) → `on_upload_pca` (déclenché en cascade) |
| 5 | `app.py` / `callbacks.py` | Dark mode : graphiques Plotly non re-rendus lors du toggle → textes noirs sur fond sombre | `clientside_callback` qui appelle `Plotly.relayout()` sur tous les graphs actifs |
| 6 | `ui_pca.py` · `circle_interpretation_panel()` | Seuil de qualité `cos² ≥ 0.4` trop bas pour inférer des corrélations ; pas de disclaimer si plan < 50% variance | Paramètre `cos2_quality_min=0.60` ; disclaimer dynamique |

### 🔵 Défauts mineurs

| # | Fichier | Problème | Correction |
|---|---------|----------|------------|
| 7 | `pca.py` | `row_labels = ["Ind. N"]` inutilisables pour l'annotation | Détection de colonne ID/string en priorité ; fallback "Ind. N" |
| 8 | `store.py` | `datetime_cols` non exposé → `pca.py` ne pouvait pas exclure les timestamps des candidats row_label | `datetime_cols: list[str]` ajouté à l'état public |
| 9 | `clustering.py` · `run_tsne()` | Pas de cap → 120 s+ pour n > 100k, sans avertissement | Cap à `T_SNE_MAX_OBS = 15 000` avec subsampling ; flag `tsne_subsampled` ; banner dans `clustering_summary_panel` |

### 🟣 Dettes architecturales

| # | Avant | Après |
|---|-------|-------|
| A | `_discrimination_power()` dans `ui_clustering.py` (UI module) | Déplacée dans `clustering.py` sous le nom `discrimination_power()` ; `ui_clustering` l'importe |
| B | `ui_pca.py` et `ui_clustering.py` — ~120 lignes de helpers dupliqués | Extraction dans `ui_common.py` (helpers génériques de référence) |
| C | `store.py` sans `datetime_cols` | `datetime_cols` ajouté, peuplé dans `reset()`, exclu des candidats ID |

### 📁 Nouveaux fichiers

- `ui_common.py` — helpers Dash partagés (`_table`, `_tip_box`, `_reading_grid`, etc.)
- `CHANGELOG.md`

### ✅ Non modifiés

`parsers.py` · `loader.py` · `utils.py` · `charts_pca.py` · `charts_clustering.py` · `assets/style.css`

---

### Lancement

```bash
pip install -r requirements.txt
python app.py
# ou
gunicorn app:server --workers 1 --bind 0.0.0.0:8050
```

> **Note multi-utilisateurs** : le state global dans `store.py` est un singleton
> single-user. Pour un déploiement multi-workers, migrer vers `dcc.Store(storage_type="session")`.
