# Investigación de ventaja predictiva y económica — protocolo

Fecha local: 17 de octubre de 2026. Nueva investigación aislada; no reemplaza los informes anteriores.

La auditoría anterior encontró sobreajuste, confianza excesiva y fuerte costo de rotación. El estudio financiero focalizado encontró indicios retrospectivos en BTC diario (logística con liquidez: +45,29% neto, +28,24% a costos dobles) y mejoras predictivas en ETH/Forex. Esos resultados reutilizan fechas y familias ya observadas: se consideran hipótesis, no evidencia independiente de rentabilidad.

Antes de entrenar se fija `config/edge_research.toml`: BTC/ETH, datos 2022–junio 2025, tres ventanas exteriores de seis meses, H1→1h, H1→4h, H4→4h y D1→24h. El contraste H1→4h frente a H4→4h separa frecuencia y horizonte físico; no constituye una réplica independiente. Las reservas desde julio de 2025 permanecen excluidas. Se verifican SHA-256 de todos los artefactos previos y contratos de reservas.

Hipótesis acotadas:

1. Features compactas de volumen, liquidez aproximada, volatilidad y ruptura aportan sobre precio y sobre ruido de igual dimensión.
2. Una logística regularizada generaliza mejor que árboles sobreconfiados. Un XGBoost pequeño, calibrado en 90 días disjuntos, permite contrastarlo sin búsqueda de hiperparámetros.
3. La expectativa de retorno estimada sólo en validación, con margen sobre costos y tamaño por volatilidad causal, reduce operaciones sin beneficio esperado.

Las etiquetas son open/open ejecutables tras la señal; purga por fecha de salida y embargo de 48 horas entre ajuste, calibración/estimación de payoff y evaluación. Los huecos invalidan las etiquetas que los cruzan y reinician el warmup de features, sin interpolación. Todas las recetas comparten filas, warmup y folds. Hiperparámetros y filtros quedan fijos; no se eligen umbrales con los tests. Presupuesto: 96 estimadores y 24 calibradores; una semilla fija, sin búsqueda masiva. Se conserva cada probabilidad, split, modelo, curva y trade.

Ejecución spot long/cash sin apalancamiento: apertura posterior al cierre, sin solapamiento, costos por lado 10 bps comisión + 5 slippage + 2 medio spread (34 bps aproximados ida y vuelta). Se evalúan bruto, base, doble costo y una barra adicional de retraso. Los costos son supuestos conservadores, no cotizaciones históricas verificadas de un broker. Se marca equity en cada apertura y se liquida dentro de cada fold. No se simula profundidad ni impacto; los proxies OHLCV no son observaciones del libro.

Políticas predeclaradas: probabilidad >=0,55, y `payoff_risk` con expectativa >1,5 veces fricción, precio sobre EMA de 30 días, volumen relativo >=0,25 y tamaño máximo 100%, objetivo de volatilidad anual 20%. Se comparan cash, buy/hold y tendencia con el mismo tamaño por riesgo y calendario.

Inferencia por bootstrap pareado de días (se agregan pérdidas intradía por día antes de remuestrear), bloques de 7 y 30 días, 2.000 réplicas, Holm en 32 contrastes predictivos y separadamente 16 económicos. Se usa el mayor p de ambas longitudes. Los intervalos son puntuales, no simultáneos. No se atribuye significancia confirmatoria a estos contrastes sobre historia reutilizada. No se estima un Deflated Sharpe con un número ficticio de intentos independientes.

Un candidato sólo merece seguimiento si mejora frente a precio, ruido y p=0,5 tras Holm, supera buy/hold en retornos diarios pareados, tiene >=30 operaciones, >=2/3 folds positivos, retorno positivo a costo doble y demora adicional. Cualquier resultado permanece experimental. El dashboard sólo recibirá candidatos que cumplan estos criterios; si ninguno cumple, se guardará el contrato de seguimiento sin activar modelos.

Fuentes primarias consultadas: [validación temporal y transformaciones](https://scikit-learn.org/stable/modules/cross_validation.html), [calibración con datos disjuntos](https://scikit-learn.org/stable/modules/calibration.html), [sesgo de selección y Deflated Sharpe](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), [archivos Binance y checksums](https://github.com/binance/binance-public-data). Las decisiones específicas anteriores son el diseño de este experimento, no recomendaciones textuales de esas fuentes.

Resultados y limitaciones se incorporarán después de cerrar la corrida; no se cambiará el protocolo según su resultado.
