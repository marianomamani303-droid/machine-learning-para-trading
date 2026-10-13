# Investigación de gestión de riesgo

Run: `20261013T221816_a8b7d443`.

**MIXTO** — comparación principal prefijada: v20261012T130540_d2bb6ac0 / combined_25. Neto -45.32% → -1.51%; drawdown -46.89% → -2.37%.

Las probabilidades del modelo son idénticas en todas las políticas. La pérdida menor puede provenir de exposición reducida, distinto sizing y menor rotación; no acredita mejor capacidad predictiva. La política principal quedó fijada en el TOML antes de ejecutar el estudio y no se seleccionó usando el historial final. No hay confirmación futura independiente.

## Auditoría de pérdidas

Se evalúan tres modelos registrados: original, ampliado y SMC. El original invierte todo el capital por señal y usa salida a plazo fijo. Entradas consecutivas pueden cerrar/reabrir en la misma apertura, pagando comisión y slippage por ambos lados incluso si la dirección prevista no cambió. Un retorno bruto negativo revela pérdidas que no se explican sólo por costos; la diferencia bruto/neto mide su arrastre compuesto en las mismas decisiones.

Las métricas de clasificación (AUC, log-loss, Brier, accuracy) se guardan en prediction_audits. El objetivo predictivo close/close del modelo difiere del retorno ejecutable open/open; una clasificación correcta puede perder después de costos. PF bruto/neto, win rate bruto/neto y retorno unitario por trade permiten distinguir señales deficientes de costos y de pesos distintos.

Momentum y Mean Reversion existentes quedan intactos. Sus informes previos ya documentan pérdidas con baja rotación y pocas operaciones: reducir costos no basta para asumir que el problema de esas estrategias desapareció. Se registran hashes de esos estudios en el plan, sin volver a entrenarlos.

## Políticas simples y causales

Controles fijos: exposición máxima 10%, 25%, 50% frente al 100% original. Sizing por volatilidad: min(cap, 0,01/σ20), donde σ20 es desviación de retornos simples disponible al cierre. Sizing ATR: min(cap, 0,01/(3ATR14/P)). Sizing combinado toma el menor de ambos. Sin apalancamiento. Los objetivos son por vela; con datos diarios 0,01 corresponde a 1% diario, no a un objetivo anual. Parámetros completos por política en el protocolo copiado.

Stop fijo de referencia: apertura de entrada−3ATR conocido al cierre de señal. Se dispara si un cierre posterior cae bajo ese nivel, y se vende en la próxima apertura con fricción. No se usa high/low posterior para afirmar un fill intravela. Gaps pueden exceder el stop. Con holding de una vela, el stop no puede anticipar la salida ya prevista: una comparación nula es un resultado válido y se conserva.

Drawdown causal: marca de cartera al cierre conocido, comparada con el máximo alcanzado hasta entonces. Al caer 5% se reduce la exposición máxima a la mitad; al caer 10% se liquida en la próxima apertura y se permanece en efectivo hasta el final del fold. El límite no garantiza drawdown ≤10% por gaps. El estado se reinicia sólo al comenzar otra ventana, como declara el protocolo. No se resetea para esconder una pérdida dentro del fold.

La política de persistencia exige dos cierres consecutivos con p≥0,55, mantiene mientras p≥0,50 hasta cinco velas y aplica dos velas de cooldown tras salir. La política combinada agrega sizing, stop y drawdown con cap 25%. Las confirmaciones anteriores al inicio del fold no se usan como predicciones fuera de muestra. La reducción de rotación es una decisión de ejecución distinta, no una recalibración de XGBoost.

Los caps se cumplen tras cada apertura ejecutada; si una posición crece por encima del límite, se reduce con costos, sin agregar exposición entre entradas. No se garantiza un cap continuo ante gaps intravela. El sizing ATR es un presupuesto aproximado basado en información de cierre, no un límite contractual de pérdida neta.

## Validación temporal y controles

Siete folds cronológicos expansivos de desarrollo, con labels de train purgadas por target_time. Se reajustan referencias usando features y cantidades fijas de árboles históricas; la referencia SMC recalibra con una cola temporal disjunta de train. Esos parámetros heredados tienen sesgo de selección histórica, por lo que desarrollo no es evidencia nueva ciega. En el período posterior se cargan los modelos existentes sin alterarlos.

Las ventanas comienzan en efectivo y liquidan al último open disponible. La agregación compone retornos de ventanas reiniciadas; no representa una cartera continua sin reinicios. Los costos en dinero y PF agregados suman presupuestos independientes de 10.000 por fold. En el período histórico de una ventana, los montos son directamente comparables.

Control matched_original: cap original fijado exclusivamente con el ratio de exposición media combinada/original en desarrollo, antes de evaluar historial final. Es un control aproximado de exposición, no igualdad exacta de riesgo ni una normalización realizada mirando el test. Los controles cap_10/25/50 mantienen las decisiones originales, para aislar el efecto mecánico de invertir menos.

Bruto: replay de las mismas entradas, salidas, reducciones y fracciones del libro neto, eliminando sólo comisión/spread/slippage. No se vuelven a decidir stops o drawdowns con una cartera bruta diferente. Intereses y financiación declarados se mantienen en ambos libros. Costos base y stress prefijados; no cambian parámetros ni probabilidades.

## development

| Sistema | Bruto | Neto | Drawdown neto | PF neto | Trades | Capital medio | Tiempo invertido | Costos ejecución |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_hold | 121.60% | 117.00% | -66.73% | 3.057 | 7 | 99.52% | 99.52% | 232.72 |
| cash | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T130540_d2bb6ac0__original | 8.44% | -69.42% | -77.06% | 0.776 | 422 | 33.76% | 33.76% | 11176.79 |
| v20261012T130540_d2bb6ac0__cap_10 | 2.65% | -9.55% | -12.18% | 0.793 | 422 | 3.37% | 33.76% | 1249.33 |
| v20261012T130540_d2bb6ac0__cap_25 | 5.97% | -22.76% | -28.24% | 0.790 | 422 | 8.43% | 33.76% | 3066.75 |
| v20261012T130540_d2bb6ac0__cap_50 | 9.52% | -41.82% | -49.72% | 0.785 | 422 | 16.87% | 33.76% | 5947.63 |
| v20261012T130540_d2bb6ac0__vol_50 | -2.05% | -39.73% | -45.23% | 0.736 | 422 | 12.95% | 33.76% | 4600.47 |
| v20261012T130540_d2bb6ac0__atr_50 | -0.54% | -11.42% | -13.03% | 0.715 | 422 | 3.09% | 33.76% | 1142.71 |
| v20261012T130540_d2bb6ac0__atr_stop_50 | -0.54% | -11.42% | -13.03% | 0.715 | 422 | 3.09% | 33.76% | 1142.71 |
| v20261012T130540_d2bb6ac0__dd_50 | -7.46% | -30.43% | -36.67% | 0.689 | 253 | 7.61% | 20.24% | 2826.45 |
| v20261012T130540_d2bb6ac0__persistence_25 | -12.04% | -16.70% | -19.65% | 0.536 | 73 | 4.21% | 17.04% | 536.50 |
| v20261012T130540_d2bb6ac0__combined_25 | -4.01% | -5.91% | -6.90% | 0.553 | 74 | 1.59% | 16.88% | 198.96 |
| v20261012T131650_5106daf4__original | -49.17% | -70.20% | -73.28% | 0.575 | 178 | 14.24% | 14.24% | 4201.41 |
| v20261012T131650_5106daf4__cap_10 | -5.47% | -10.36% | -11.36% | 0.604 | 178 | 1.42% | 14.24% | 520.03 |
| v20261012T131650_5106daf4__cap_25 | -13.52% | -24.29% | -26.37% | 0.599 | 178 | 3.56% | 14.24% | 1254.91 |
| v20261012T131650_5106daf4__cap_50 | -26.38% | -43.59% | -46.63% | 0.591 | 178 | 7.11% | 14.24% | 2365.45 |
| v20261012T131650_5106daf4__vol_50 | -27.38% | -39.55% | -42.27% | 0.498 | 178 | 4.91% | 14.24% | 1669.33 |
| v20261012T131650_5106daf4__atr_50 | -5.96% | -9.58% | -10.31% | 0.506 | 178 | 1.05% | 14.24% | 386.56 |
| v20261012T131650_5106daf4__atr_stop_50 | -5.96% | -9.58% | -10.31% | 0.506 | 178 | 1.05% | 14.24% | 386.56 |
| v20261012T131650_5106daf4__dd_50 | -14.22% | -21.74% | -25.96% | 0.511 | 87 | 2.46% | 6.96% | 897.81 |
| v20261012T131650_5106daf4__persistence_25 | -0.84% | -3.25% | -15.11% | 0.864 | 33 | 2.10% | 8.56% | 246.43 |
| v20261012T131650_5106daf4__combined_25 | -0.68% | -1.43% | -4.56% | 0.789 | 33 | 0.63% | 8.16% | 76.48 |
| v20261012T134917_f88d6f93__original | 44.23% | -16.20% | -40.01% | 0.924 | 181 | 14.48% | 14.48% | 5775.25 |
| v20261012T134917_f88d6f93__cap_10 | 4.42% | -1.10% | -4.61% | 0.943 | 181 | 1.45% | 14.48% | 547.85 |
| v20261012T134917_f88d6f93__cap_25 | 11.11% | -3.00% | -11.27% | 0.939 | 181 | 3.62% | 14.48% | 1386.89 |
| v20261012T134917_f88d6f93__cap_50 | 22.32% | -6.77% | -21.70% | 0.934 | 181 | 7.23% | 14.48% | 2822.86 |
| v20261012T134917_f88d6f93__vol_50 | 19.45% | -2.68% | -17.87% | 0.964 | 181 | 5.46% | 14.48% | 2168.37 |
| v20261012T134917_f88d6f93__atr_50 | 4.50% | -0.26% | -4.38% | 0.983 | 181 | 1.24% | 14.48% | 473.43 |
| v20261012T134917_f88d6f93__atr_stop_50 | 4.50% | -0.26% | -4.38% | 0.983 | 181 | 1.24% | 14.48% | 473.43 |
| v20261012T134917_f88d6f93__dd_50 | 19.85% | 5.41% | -10.81% | 1.109 | 121 | 3.42% | 9.68% | 1359.74 |
| v20261012T134917_f88d6f93__persistence_25 | 4.03% | 2.24% | -4.91% | 1.159 | 23 | 2.25% | 9.12% | 175.15 |
| v20261012T134917_f88d6f93__combined_25 | 2.07% | 1.46% | -1.49% | 1.326 | 23 | 0.79% | 9.12% | 60.38 |

Estabilidad por fold, trimestre y régimen en summary.json. La exposición de capital es promedio ponderado por tiempo; tiempo invertido es fracción de intervalos con alguna posición. No son la misma métrica. Costos incluyen compras, ventas y trims; su monto no es exactamente la diferencia de capital bruto/neto debido a reinversión y capitalizaciones.

## development_stress

| Sistema | Bruto | Neto | Drawdown neto | PF neto | Trades | Capital medio | Tiempo invertido | Costos ejecución |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_hold | 121.60% | 112.49% | -66.83% | 3.012 | 7 | 99.52% | 99.52% | 464.73 |
| cash | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T130540_d2bb6ac0__original | 8.44% | -91.38% | -93.14% | 0.581 | 422 | 33.76% | 33.76% | 20350.84 |
| v20261012T130540_d2bb6ac0__cap_10 | 2.65% | -20.27% | -22.13% | 0.593 | 422 | 3.37% | 33.76% | 2469.75 |
| v20261012T130540_d2bb6ac0__cap_25 | 5.97% | -43.66% | -46.88% | 0.591 | 422 | 8.42% | 33.76% | 5973.30 |
| v20261012T130540_d2bb6ac0__cap_50 | 9.52% | -69.07% | -72.47% | 0.588 | 422 | 16.85% | 33.76% | 11312.96 |
| v20261012T130540_d2bb6ac0__vol_50 | -2.05% | -62.88% | -65.45% | 0.546 | 422 | 12.94% | 33.76% | 8842.45 |
| v20261012T130540_d2bb6ac0__atr_50 | -0.54% | -21.08% | -22.04% | 0.523 | 422 | 3.09% | 33.76% | 2258.70 |
| v20261012T130540_d2bb6ac0__atr_stop_50 | -0.54% | -21.08% | -22.04% | 0.523 | 422 | 3.09% | 33.76% | 2258.70 |
| v20261012T130540_d2bb6ac0__dd_50 | 0.75% | -39.62% | -43.52% | 0.564 | 236 | 6.83% | 18.88% | 5008.07 |
| v20261012T130540_d2bb6ac0__persistence_25 | -12.04% | -21.10% | -22.48% | 0.434 | 73 | 4.20% | 17.04% | 1066.95 |
| v20261012T130540_d2bb6ac0__combined_25 | -4.01% | -7.77% | -8.15% | 0.445 | 74 | 1.59% | 16.88% | 396.68 |
| v20261012T131650_5106daf4__original | -49.17% | -82.53% | -84.23% | 0.448 | 178 | 14.24% | 14.24% | 7741.38 |
| v20261012T131650_5106daf4__cap_10 | -5.47% | -14.99% | -15.88% | 0.473 | 178 | 1.42% | 14.24% | 1028.68 |
| v20261012T131650_5106daf4__cap_25 | -13.52% | -33.70% | -35.41% | 0.468 | 178 | 3.55% | 14.24% | 2448.65 |
| v20261012T131650_5106daf4__cap_50 | -26.38% | -56.76% | -58.95% | 0.461 | 178 | 7.11% | 14.24% | 4519.04 |
| v20261012T131650_5106daf4__vol_50 | -27.38% | -49.67% | -51.78% | 0.386 | 178 | 4.90% | 14.24% | 3238.54 |
| v20261012T131650_5106daf4__atr_50 | -5.96% | -13.06% | -13.71% | 0.388 | 178 | 1.05% | 14.24% | 766.94 |
| v20261012T131650_5106daf4__atr_stop_50 | -5.96% | -13.06% | -13.71% | 0.388 | 178 | 1.05% | 14.24% | 766.94 |
| v20261012T131650_5106daf4__dd_50 | -9.75% | -23.66% | -27.52% | 0.434 | 76 | 2.24% | 6.08% | 1628.24 |
| v20261012T131650_5106daf4__persistence_25 | -0.83% | -5.60% | -15.47% | 0.766 | 33 | 2.10% | 8.56% | 489.86 |
| v20261012T131650_5106daf4__combined_25 | -0.68% | -2.18% | -4.69% | 0.688 | 33 | 0.63% | 8.16% | 152.51 |
| v20261012T134917_f88d6f93__original | 44.23% | -51.31% | -56.48% | 0.726 | 181 | 14.48% | 14.48% | 8921.45 |
| v20261012T134917_f88d6f93__cap_10 | 4.42% | -6.32% | -7.61% | 0.711 | 181 | 1.44% | 14.48% | 1065.07 |
| v20261012T134917_f88d6f93__cap_25 | 11.11% | -15.29% | -18.08% | 0.714 | 181 | 3.61% | 14.48% | 2590.83 |
| v20261012T134917_f88d6f93__cap_50 | 22.32% | -28.91% | -33.27% | 0.718 | 181 | 7.23% | 14.48% | 4940.55 |
| v20261012T134917_f88d6f93__vol_50 | 19.45% | -20.68% | -26.73% | 0.735 | 181 | 5.45% | 14.48% | 3915.38 |
| v20261012T134917_f88d6f93__atr_50 | 4.50% | -4.79% | -6.90% | 0.734 | 181 | 1.24% | 14.48% | 923.85 |
| v20261012T134917_f88d6f93__atr_stop_50 | 4.50% | -4.79% | -6.90% | 0.734 | 181 | 1.24% | 14.48% | 923.85 |
| v20261012T134917_f88d6f93__dd_50 | 13.72% | -6.86% | -10.10% | 0.840 | 104 | 2.65% | 8.32% | 1956.87 |
| v20261012T134917_f88d6f93__persistence_25 | 4.03% | 0.50% | -5.12% | 1.033 | 23 | 2.25% | 9.12% | 346.87 |
| v20261012T134917_f88d6f93__combined_25 | 2.07% | 0.85% | -1.55% | 1.179 | 23 | 0.79% | 9.12% | 120.23 |

Estabilidad por fold, trimestre y régimen en summary.json. La exposición de capital es promedio ponderado por tiempo; tiempo invertido es fracción de intervalos con alguna posición. No son la misma métrica. Costos incluyen compras, ventas y trims; su monto no es exactamente la diferencia de capital bruto/neto debido a reinversión y capitalizaciones.

## observed_historical

| Sistema | Bruto | Neto | Drawdown neto | PF neto | Trades | Capital medio | Tiempo invertido | Costos ejecución |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_hold | -20.09% | -20.33% | -52.97% | 0.000 | 1 | 100.00% | 100.00% | 26.95 |
| cash | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T130540_d2bb6ac0__original | -22.56% | -45.32% | -46.89% | 0.546 | 116 | 24.07% | 24.07% | 2745.29 |
| v20261012T130540_d2bb6ac0__cap_10 | -2.17% | -5.51% | -5.78% | 0.554 | 116 | 2.40% | 24.07% | 339.04 |
| v20261012T130540_d2bb6ac0__cap_25 | -5.48% | -13.33% | -13.97% | 0.553 | 116 | 6.01% | 24.07% | 818.06 |
| v20261012T130540_d2bb6ac0__cap_50 | -11.10% | -25.27% | -26.36% | 0.550 | 116 | 12.02% | 24.07% | 1542.44 |
| v20261012T130540_d2bb6ac0__vol_50 | -9.80% | -22.75% | -24.05% | 0.535 | 116 | 10.73% | 24.07% | 1395.29 |
| v20261012T130540_d2bb6ac0__atr_50 | -2.36% | -6.14% | -6.47% | 0.510 | 116 | 2.73% | 24.07% | 384.26 |
| v20261012T130540_d2bb6ac0__atr_stop_50 | -2.36% | -6.14% | -6.47% | 0.510 | 116 | 2.73% | 24.07% | 384.26 |
| v20261012T130540_d2bb6ac0__dd_50 | -4.06% | -9.90% | -11.21% | 0.491 | 56 | 4.35% | 11.62% | 614.97 |
| v20261012T130540_d2bb6ac0__persistence_25 | 0.12% | -1.30% | -5.30% | 0.803 | 19 | 3.34% | 13.49% | 141.78 |
| v20261012T130540_d2bb6ac0__combined_25 | -0.92% | -1.51% | -2.37% | 0.555 | 19 | 1.55% | 13.49% | 59.68 |
| v20261012T130540_d2bb6ac0__matched_original | -1.02% | -2.62% | -2.76% | 0.555 | 116 | 1.13% | 24.07% | 161.89 |
| v20261012T131650_5106daf4__original | -9.16% | -13.68% | -21.92% | 0.559 | 17 | 3.53% | 3.53% | 485.35 |
| v20261012T131650_5106daf4__cap_10 | -0.77% | -1.27% | -2.33% | 0.604 | 17 | 0.35% | 3.53% | 50.60 |
| v20261012T131650_5106daf4__cap_25 | -1.99% | -3.23% | -5.77% | 0.596 | 17 | 0.88% | 3.53% | 125.71 |
| v20261012T131650_5106daf4__cap_50 | -4.19% | -6.59% | -11.35% | 0.583 | 17 | 1.76% | 3.53% | 248.63 |
| v20261012T131650_5106daf4__vol_50 | -4.43% | -6.51% | -9.41% | 0.508 | 17 | 1.52% | 3.53% | 215.57 |
| v20261012T131650_5106daf4__atr_50 | -0.53% | -0.96% | -1.61% | 0.591 | 17 | 0.30% | 3.53% | 43.59 |
| v20261012T131650_5106daf4__atr_stop_50 | -0.53% | -0.96% | -1.61% | 0.591 | 17 | 0.30% | 3.53% | 43.59 |
| v20261012T131650_5106daf4__dd_50 | -8.09% | -9.45% | -11.28% | 0.244 | 10 | 1.04% | 2.07% | 149.50 |
| v20261012T131650_5106daf4__persistence_25 | -1.38% | -1.52% | -3.54% | 0.000 | 2 | 0.45% | 1.87% | 14.69 |
| v20261012T131650_5106daf4__combined_25 | -0.41% | -0.45% | -0.91% | 0.000 | 2 | 0.15% | 1.87% | 4.69 |
| v20261012T131650_5106daf4__matched_original | -0.34% | -0.56% | -1.03% | 0.607 | 17 | 0.16% | 3.53% | 22.33 |
| v20261012T134917_f88d6f93__original | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__cap_10 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__cap_25 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__cap_50 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__vol_50 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__atr_50 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__atr_stop_50 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__dd_50 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__persistence_25 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__combined_25 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__matched_original | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |

Estabilidad por fold, trimestre y régimen en summary.json. La exposición de capital es promedio ponderado por tiempo; tiempo invertido es fracción de intervalos con alguna posición. No son la misma métrica. Costos incluyen compras, ventas y trims; su monto no es exactamente la diferencia de capital bruto/neto debido a reinversión y capitalizaciones.

## observed_historical_stress

| Sistema | Bruto | Neto | Drawdown neto | PF neto | Trades | Capital medio | Tiempo invertido | Costos ejecución |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_hold | -20.09% | -20.57% | -52.97% | 0.000 | 1 | 100.00% | 100.00% | 53.82 |
| cash | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T130540_d2bb6ac0__original | -22.56% | -61.39% | -62.19% | 0.380 | 116 | 24.07% | 24.07% | 4730.12 |
| v20261012T130540_d2bb6ac0__cap_10 | -2.17% | -8.72% | -8.91% | 0.384 | 116 | 2.40% | 24.07% | 665.71 |
| v20261012T130540_d2bb6ac0__cap_25 | -5.48% | -20.52% | -20.94% | 0.384 | 116 | 6.00% | 24.07% | 1567.42 |
| v20261012T130540_d2bb6ac0__cap_50 | -11.10% | -37.17% | -37.83% | 0.383 | 116 | 12.02% | 24.07% | 2843.93 |
| v20261012T130540_d2bb6ac0__vol_50 | -9.80% | -33.82% | -34.51% | 0.363 | 116 | 10.72% | 24.07% | 2593.65 |
| v20261012T130540_d2bb6ac0__atr_50 | -2.36% | -9.76% | -9.99% | 0.332 | 116 | 2.73% | 24.07% | 752.79 |
| v20261012T130540_d2bb6ac0__atr_stop_50 | -2.36% | -9.76% | -9.99% | 0.332 | 116 | 2.73% | 24.07% | 752.79 |
| v20261012T130540_d2bb6ac0__dd_50 | 1.51% | -9.27% | -10.23% | 0.407 | 54 | 3.88% | 11.20% | 1075.81 |
| v20261012T130540_d2bb6ac0__persistence_25 | 0.12% | -2.69% | -5.61% | 0.637 | 19 | 3.33% | 13.49% | 281.24 |
| v20261012T130540_d2bb6ac0__combined_25 | -0.92% | -2.10% | -2.61% | 0.448 | 19 | 1.55% | 13.49% | 118.84 |
| v20261012T130540_d2bb6ac0__matched_original | -1.02% | -4.20% | -4.30% | 0.384 | 116 | 1.13% | 24.07% | 320.70 |
| v20261012T131650_5106daf4__original | -9.16% | -17.97% | -23.54% | 0.454 | 17 | 3.53% | 3.53% | 947.30 |
| v20261012T131650_5106daf4__cap_10 | -0.77% | -1.77% | -2.53% | 0.493 | 17 | 0.35% | 3.53% | 100.81 |
| v20261012T131650_5106daf4__cap_25 | -1.99% | -4.45% | -6.24% | 0.486 | 17 | 0.88% | 3.53% | 249.56 |
| v20261012T131650_5106daf4__cap_50 | -4.19% | -8.93% | -12.24% | 0.475 | 17 | 1.76% | 3.53% | 490.74 |
| v20261012T131650_5106daf4__vol_50 | -4.43% | -8.54% | -10.09% | 0.402 | 17 | 1.52% | 3.53% | 426.18 |
| v20261012T131650_5106daf4__atr_50 | -0.53% | -1.39% | -1.79% | 0.459 | 17 | 0.30% | 3.53% | 86.86 |
| v20261012T131650_5106daf4__atr_stop_50 | -0.53% | -1.39% | -1.79% | 0.459 | 17 | 0.30% | 3.53% | 86.86 |
| v20261012T131650_5106daf4__dd_50 | -2.58% | -6.29% | -8.74% | 0.460 | 17 | 1.35% | 3.53% | 382.50 |
| v20261012T131650_5106daf4__persistence_25 | -1.38% | -1.67% | -3.57% | 0.000 | 2 | 0.45% | 1.87% | 29.33 |
| v20261012T131650_5106daf4__combined_25 | -0.41% | -0.50% | -0.92% | 0.000 | 2 | 0.15% | 1.87% | 9.37 |
| v20261012T131650_5106daf4__matched_original | -0.34% | -0.78% | -1.12% | 0.495 | 17 | 0.15% | 3.53% | 44.54 |
| v20261012T134917_f88d6f93__original | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__cap_10 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__cap_25 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__cap_50 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__vol_50 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__atr_50 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__atr_stop_50 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__dd_50 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__persistence_25 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__combined_25 | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |
| v20261012T134917_f88d6f93__matched_original | 0.00% | 0.00% | 0.00% | — | 0 | 0.00% | 0.00% | 0.00 |

Estabilidad por fold, trimestre y régimen en summary.json. La exposición de capital es promedio ponderado por tiempo; tiempo invertido es fracción de intervalos con alguna posición. No son la misma métrica. Costos incluyen compras, ventas y trims; su monto no es exactamente la diferencia de capital bruto/neto debido a reinversión y capitalizaciones.

## Diagnóstico y límites

Estado principal: MIXTO. Desarrollo y test histórico pueden dar signos diferentes; mejorar un historial negativo respecto a una exposición del 100% no implica superar efectivo ni poseer una ventaja predictiva.

Costos base: `{'commission_bps': 10.0, 'slippage_bps': 5.0, 'spread_bps': 0.0, 'annual_holding_bps': 0.0, 'annual_cash_rate': 0.0, 'initial_capital': 10000.0}`; stress: `{'commission_bps': 20.0, 'slippage_bps': 10.0, 'spread_bps': 0.0, 'annual_holding_bps': 0.0, 'annual_cash_rate': 0.0, 'initial_capital': 10000.0}`. Son supuestos de ejecución, no mediciones del libro, y no cubren impuestos, latencia, lotes o impacto de mercado dependiente del tamaño. No hay trading real.

Como referencia de investigación, [Moreira y Muir, Volatility Managed Portfolios](https://www.nber.org/papers/w22208) estudian gestión de volatilidad en carteras; esos resultados no prueban que este overlay diario sobre BTC mejore las predicciones del modelo ni validan nuestros umbrales.

Se conservaron 518 registros, incluidos nulos y negativos, con 21 refits temporales de XGBoost. Se verificaron por hash modelos, punteros, datos y planes/modelos congelados de las reservas existentes. No se modificó ni evaluó la reserva prospectiva. Toda selección adicional exigiría un protocolo nuevo y no convertiría este historial ya observado en un test independiente.

## Ejecutar e integrar

```powershell
.\.venv\Scripts\python.exe -m trading_ml risk-study --protocol config/risk_study.toml
# Documentar resultados existentes, sin repetir el test:
.\.venv\Scripts\python.exe scripts/record_risk_study.py artifacts/risk_studies/20261013T221816_a8b7d443
```

Cada ejecución crea un directorio nuevo y conserva resultados negativos/nulos.
El comando compara 10 políticas prefijadas sobre tres modelos registrados,
con costos base y duplicados. No realiza búsqueda, promoción ni trading real.

El módulo `trading_ml.risk_overlay` separa configuración, simulador, estudio y
reporte. `RiskPolicy` y `run_overlay` permiten experimentar con probabilidades
ya calculadas; no modifican un modelo ni un archivo de estrategia original.

```python
import pandas as pd
from trading_ml.momentum.config import ResearchCosts
from trading_ml.risk_overlay.config import RiskPolicy
from trading_ml.risk_overlay.simulator import risk_features, run_overlay

# frame: OHLCV, timestamp y reference_time UTC; p es un array alineado.
# El llamador del API asegura que p sea una predicción causal fuera de muestra.
result = run_overlay(
    risk_features(frame),
    p,
    pd.Timestamp("2025-06-05", tz="UTC"),
    pd.Timestamp("2026-10-01", tz="UTC"),
    ResearchCosts(spread_bps=0),
    RiskPolicy(name="cap_25", capital_cap=0.25),
    horizon=1,
)
```

El simulador comprueba cierre de señal anterior a ejecución, probabilidades
finitas y caps en las aperturas. El runner valida procedencia, período anterior
a las reservas y entrenamiento temporal con purga. El API de bajo nivel exige
que el llamador mantenga el contrato fuera de muestra.

`events.csv` registra compra, venta y cada reducción parcial con sus costos;
`trades.csv` agrupa esos eventos por operación y contiene PnL bruto/neto.
`equity.csv` conserva exposición, drawdown usado para decidir, circuito de
pausa y las dos curvas. `matched_caps.json` se fija con desarrollo únicamente.

El caso principal fue fijado como `combined_25` sobre
`v20261012T130540_d2bb6ac0` antes de evaluar el período final.
La reserva prospectiva y los modelos registrados se preservaron por hash;
esa reserva no se evaluó ni se incorporó al entrenamiento. Los resultados
históricos siguen siendo descriptivos porque ese historial ya se observó.
