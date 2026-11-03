# Datos incluidos: río Cauca (Colombia)

Conjunto curado completo que usa el modo Colombia (`--region cauca`). Es pequeño, así que todo se puede correr sin internet.

gauges.json: 27 estaciones automáticas del IDEAM en la cuenca del Cauca, desde Popayán hasta San Jacinto del Cauca, con sus niveles de referencia estadísticos (vigilancia = percentil 90 y alto = percentil 98 del nivel máximo diario).
edges.json: 26 tramos de río entre cada estación y la siguiente aguas abajo, con la distancia por el río tomada de HydroRIVERS.
events.json: 12 crecientes, los días en que más estaciones pasaron su nivel alto entre noviembre de 2022 y noviembre de 2026.
stage_hourly.parquet: nivel horario de cada estación en la ventana de cada creciente.
quality.json: problemas de calidad detectados (cambios de nivel de referencia del sensor, estaciones con poco histórico).

Fuentes:
IDEAM, "Nivel Instantáneo del Río", datos.gov.co (bdmn-sqnh), licencia CC BY-SA 4.0. Descargado el 2026-11-03.
HydroRIVERS v1.0, Lehner y Grill (2013), HydroSHEDS. Solo se guardan las distancias derivadas entre estaciones.

Los niveles de referencia de este proyecto no son los niveles de alerta oficiales del IDEAM.
Los datos de esta carpeta conservan su licencia CC BY-SA 4.0. El código del repositorio es MIT.

Para volver a construirlo: python -m upstream_watch --region cauca ingest
