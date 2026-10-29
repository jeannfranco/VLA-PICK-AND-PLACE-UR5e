# Integración y evaluación de modelos de visión, lenguaje y acción en tareas de pick-and-place con un UR5e

Proyecto de Trabajo de Fin de Grado centrado en la integración y evaluación experimental de modelos de visión, lenguaje y acción (VLA) en tareas de manipulación robótica con un UR5e.

## Objetivo

Evaluar el comportamiento de OpenVLA, π0-DROID y π0.5-DROID en tareas de pick-and-place sin fine-tuning, en un entorno robótico real. El trabajo estudia tanto la ejecución física como aspectos de selección visual, grounding y robustez lingüística.

## Plataforma experimental

- Robot industrial UR5e.
- Pinza OnRobot 3FG15.
- Cámara 3D Photoneo MotionCam-3D en configuración eye-to-hand.
- Jetson AGX Orin y PC central para los componentes de procesamiento e inferencia.
- ROS 2 y servicios externos para la ejecución de los modelos.

La reproducción física requiere configurar correctamente el hardware, la comunicación entre componentes y los parámetros geométricos del sistema.

## Metodología

Se evaluaron tres modelos: OpenVLA, π0-DROID y π0.5-DROID, sin fine-tuning.

Las salidas de los modelos no controlan por sí solas toda la manipulación. La integración utiliza componentes externos para gestionar aspectos como la orientación, la pinza, la cinemática inversa, la planificación y las comprobaciones de seguridad.

El sistema de percepción utiliza escenas capturadas con Photoneo. El grounding con SigLIP se emplea en las configuraciones experimentales correspondientes para evaluar la selección de objetos y regiones relevantes.

## Resultados experimentales

El cierre experimental comprende **66 episodios físicos** entre las dos tareas. Los criterios de éxito se mantienen separados porque representan métricas diferentes.

| Métrica | Tarea 1: objeto y zona | Tarea 2: clasificación por forma |
|---|---:|---:|
| Episodios físicos | 30 | 36 |
| Éxito operativo / colocación correcta | 21/30 (70,0 %) | 14/36 (38,9 %) |
| Éxito estricto | 12/30 (40,0 %) | 13/36 (36,1 %) |
| Grounding correcto | Evaluado por separado | 19/36 (52,8 %) |

Las campañas de robustez lingüística y las pruebas de grounding se analizan por separado de las métricas de ejecución física. Las tablas completas, los resúmenes y las gráficas se encuentran en `resultados_finales/`.

Los resultados deben interpretarse dentro de las condiciones y métricas definidas para cada experimento. Las diferencias entre modelos no demuestran por sí solas una causa concreta.

## Estructura del repositorio

- `codigo/`: paquete de código complementario documentado en el Apéndice C de la memoria.
- `resultados_finales/`: tablas CSV, resúmenes, gráficas SVG y sumas de comprobación del cierre experimental.
- `instrucciones/`: instrucciones utilizadas en las tareas experimentales.
- `audiovisual/`: vídeos de ejecuciones físicas representativas.
- `TFG_JeanfrancoAbchiSoto.pdf`: memoria completa del TFG.

## Código y reproducibilidad

El paquete de `codigo/` corresponde al material complementario documentado en la memoria. **No es una instalación autónoma de todo el sistema experimental.**

Parte de los módulos auxiliares y de la configuración de despliegue permaneció en la Jetson AGX Orin y en el entorno utilizado durante los experimentos. La reproducción exacta puede requerir, entre otros elementos:

- ROS 2, controladores y configuración de comunicación del UR5e y la pinza OnRobot 3FG15.
- Cámara Photoneo y sus componentes de captura.
- Calibración extrínseca cámara-robot y parámetros geométricos de la celda.
- Módulos auxiliares de control, adaptación de acciones y grounding que no forman parte de este paquete público.
- Checkpoints, dependencias y servicios de inferencia externos de los modelos.
- Configuración de red y rutas de ejecución adaptadas a la instalación concreta. El destino SSH de la Jetson se configura mediante la variable de entorno `VLA_ORIN_SSH`; no se publica una IP ni un usuario predeterminados.

Por tanto, la publicación del código permite examinar la organización y parte de la implementación del sistema, pero no garantiza la reproducción directa de los experimentos sin reconstruir el entorno de despliegue.

Antes de ejecutar cualquier componente sobre hardware real, deben verificarse la configuración de la celda, los límites de movimiento y las condiciones de seguridad.

## Material audiovisual

Los vídeos incluidos en `audiovisual/` documentan ejecuciones físicas de las tareas. Se proporcionan como material complementario y no sustituyen a las tablas de resultados agregados.

## Memoria

La memoria completa está disponible en [`TFG_JeanfrancoAbchiSoto.pdf`](TFG_JeanfrancoAbchiSoto.pdf).
