# Changelog — A/B Testing Pipeline

## v2.0.0 — Bug-fix & optimisation release

### 🔴 Critical fixes

| # | Fichier | Problème | Correction |
|---|---------|----------|------------|
| 1 | `src/logistic_regression.py` · `_safe_glm()` | Conversion `.fillna(0).values` → numpy array : `model.params[col]` levait `IndexError` absorbée silencieusement → OR = 1.0, p = 1.0 systématiquement | Passer le DataFrame directement à `sm.GLM` (sans `.values`) pour préserver les noms de colonnes |
| 2 | `src/hypothesis_testing.py` · `z_test_proportions()` | `proportions_ztest([ctrl, trt], alternative='larger')` testait H₁ : ctrl > trt — opposé de l'étiquette "T > C" affichée | Swapper l'ordre : `proportions_ztest([trt, ctrl], alternative='larger')` |

### 🟠 Bugs majeurs

| # | Fichier | Problème | Correction |
|---|---------|----------|------------|
| 3 | `src/data_loader.py` · `prepare_data()` | Aucun nettoyage — 3 893 mismatches groupe/page et doublons inclus dans tous les calculs | Nouvelle fonction `clean_data()` appelée dans `prepare_data(page_col=...)` ; rapport de nettoyage exposé dans le dashboard |
| 4 | `src/robustness.py` · `_bootstrap_sample_size()` | Plafond à 5 000 → seulement 3.4 % des données utilisées par itération ; IC artificiellement larges | Plafond relevé à `BOOTSTRAP_MAX_SAMPLE = 50 000` (configurable via `config.py`) |
| 5 | `src/segmentation.py` | `multipletests` importé mais jamais appelé → p-valeurs non corrigées, faux positifs ×4–5 | Correction BH FDR appliquée sur l'ensemble des segments via `_apply_fdr()` ; clés `p_value_raw` et `p_value` distinctes |
| 6 | `src/bayesian_analysis.py` | ROPE ±0.001 trop étroit (±0.1 pp) — `practical_equivalence` quasi-inatteignable ; constantes de `config.py` ignorées | ROPE relevé à ±0.002 par défaut ; tous les paramètres lus depuis `config.py` (`BAYESIAN_SIMULATIONS`, `ROPE_LOWER/UPPER`, `RANDOM_SEED`) |
| 7 | `app/dashboard.py` · onglet "Statistical Tests" | `st.markdown("""...""")` sans préfixe `f` → expressions Python affichées telles quelles (code brut visible) | Ajout du préfixe `f` ; interpolation correcte de toutes les valeurs |

### 🔵 Défauts mineurs

| # | Fichier | Problème | Correction |
|---|---------|----------|------------|
| 8 | `src/effect_size.py` · `nnt()` | Retournait toujours "NNT" même quand diff < 0 → terminologie trompeuse | `nnt()` retourne un dict `{value, label, direction}` ; dashboard affiche "NNT" ou "NNH" selon le signe |
| 9 | `src/store.py` · `_is_id_column()` | `timestamp` classé comme candidat ID (haute cardinalité) → selectbox "ID column" incohérent | Exclusion explicite des colonnes datetime de `id_candidates` |
| 10 | `src/temporal_analysis.py` | Pearsonr non pondéré sur moyennes hétéroscédastiques | Pearsonr pondéré par `√n` par jour ; p-value via t-distribution |

### 🟣 Dettes architecturales

| # | Avant | Après |
|---|-------|-------|
| A | Analyse de puissance dupliquée dans `run.py` et `dashboard.py` | Centralisée dans `src/power_analysis.py` ; les deux l'importent |
| B | `config.py` déclarait des constantes ignorées par tous les modules | Tous les modules `src/` lisent leurs constantes depuis `config.py` |
| C | `app/dashboard.py` ne proposait aucun export | Onglet **Export** avec `st.download_button` pour le rapport `.txt` et les métriques `.csv` |
| D | Sélection du groupe contrôle par ordre alphabétique (fragile) | Champ `control_value` explicite dans `ABTestSchema` + selectbox dédié dans la sidebar |
| E | `COLORS` défini dans `tabs[0]`, utilisé dans `tabs[3]` (scope fragile) | Défini une fois en tête de module, accessible partout |

### 📁 Nouveaux fichiers

- `src/power_analysis.py` — module centralisé : `compute_power(cohens_h, n_control, n_treatment, alpha)`
- `requirements.txt`
- `CHANGELOG.md`

### ✅ Non modifiés (inchangés)

`src/parsers.py` · `src/descriptive_stats.py` · `src/data_quality.py` (hors signature) · `src/visualizations.py` · `src/schema.py` (étendu : champ `control_value`)

---

### Lancement

```bash
# CLI
python run.py --data ab_data.csv --page-col landing_page --control control

# Dashboard Streamlit
streamlit run app/dashboard.py
```
