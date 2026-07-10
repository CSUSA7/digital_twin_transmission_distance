# `07_beamline_emulator_final.ipynb`

Gemelo digital final: un solo objeto con dos cabezas de GP sobre el mismo espacio de entrada
(cubo unitario de los 8 voltajes):

- **Cabeza de transmisión** — GP sobre `log1p(hits)`. Modela la cantidad objetivo, pero es
  ciega en el ~90% del espacio donde `hits = 0`.
- **Cabeza de distancia** — GP sobre `log1p(distancia media al centro del detector)`. Continua
  en todo el espacio, así que ordena la meseta donde la cabeza de transmisión no tiene señal.

**Acople:** la cabeza de distancia se convierte en `p_close` (probabilidad de caer cerca del
detector, con umbral adaptativo = mejor cuartil observado, piso `CLOSE_MM`). La adquisición y
las dos tareas de control (dirección y consigna inversa) quedan multiplicadas/filtradas por
`p_close`, así el gemelo deja de perseguir puntos donde la cabeza de hits alucina señal pero el
haz nunca llega al detector.

**Datos:** un único CSV, `simion_opt_v/beamline_distance_results.csv` — cada fila trae `hits` y
`distance_mm` de la misma corrida de SIMION, así que las dos cabezas se entrenan siempre con
las mismas filas. El diseño inicial reusa el estudio de Optuna de `optimizer.py` (TPE, no
Sobol) porque la señal es demasiado escasa (~2-5%) para que un barrido espacial la encuentre.

## Dependencias: `gp.py` / `acquisition.py` (sin tocar) → `_final.py` (nuevo)

Restricción de diseño del notebook: `gp.py` y `acquisition.py` (los ejercicios base de GP y
adquisición) no se modifican. Todo lo nuevo vive en dos módulos que construyen sobre ellos:

- **[`gp_final.py`](gp_final.py)** — `BeamlineEmulator`: combina `GaussianProcess` (de
  [`gp.py`](gp.py), cabeza de transmisión) con `DistanceGP` (de
  [`gp_distance.py`](gp_distance.py), cabeza de distancia) y añade `p_close` y
  `coupled_transmission` como capa de acople.
- **[`acquisition_final.py`](acquisition_final.py)** — EI acoplado (`coupled_expected_improvement`,
  `propose_next_point`) y las dos tareas de control (`best_predicted_point`,
  `match_predicted_point`). Reutiliza los generadores de candidatos de
  [`acquisition_distance.py`](acquisition_distance.py) (`sobol_candidates`, `local_candidates`,
  `expected_improvement_min`) en vez de duplicarlos; no depende de `acquisition.py` directamente.

```
gp.py  ──────────────┐
                      ├─▶ gp_final.py (BeamlineEmulator) ─┐
gp_distance.py ───────┘                                   │
                                                            ├─▶ notebook 07
acquisition_distance.py ─▶ acquisition_final.py ──────────┘
                            (acquisition.py NO se usa aquí)
```

Cada corrida real de SIMION alimenta las dos cabezas a la vez y se persiste de inmediato en el
CSV antes de reajustar los hiperparámetros, para no perder presupuesto pagado ante una
interrupción.
