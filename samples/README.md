# Datos de ejemplo

CSV para comprobar la carga de datos del dashboard (pestaña **Tus datos**). También se
pueden ver y analizar desde la propia página, en «Ver datos de ejemplo».

| Archivo | Filas | Qué comprueba |
| --- | --- | --- |
| `marcas_etiquetadas.csv` | 359 | Columnas en inglés (`text`, `sentiment`, `brand`) con etiquetas `positive`/`negative`: métricas, matriz de confusión y gráfico por marca. |
| `sentiment140_original.csv` | 498 | Formato original de Sentiment140: sin cabecera, 6 columnas, etiquetas `0`/`2`/`4`. Los 139 neutrales se clasifican pero se excluyen de las métricas. |
| `tweets_sin_etiqueta.csv` | 600 | Solo texto (`tweet`) y fecha (`date`), sin sentimiento: el dashboard muestra únicamente estimaciones y la fecha se conserva en la descarga. Tweets del conjunto de evaluación, no usados para entrenar. |
| `resenas_productos.csv` | 30 | Columnas en español (`comentario`, `sentimiento`, `producto`, `fecha`), separador `;` y valores `neutral`, que se excluyen de las métricas. Reseñas ficticias escritas para la prueba. |

Los tres primeros salen de Sentiment140 (Go, Bhayani y Huang, 2009).
