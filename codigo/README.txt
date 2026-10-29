CÓDIGO PRINCIPAL DE LA IMPLEMENTACIÓN

Esta carpeta contiene los scripts principales utilizados en la integración
experimental de los modelos VLA.

El conjunto se proporciona con fines de documentación y trazabilidad. La
ejecución completa depende también de los entornos de OpenVLA y OpenPI, los
checkpoints, la configuración del UR5e, la Photoneo y otros servicios de la
célula que no forman parte de esta carpeta.

ARCHIVOS INCLUIDOS

principal.py
  Integración principal del sistema VLA.

flujo.py
  Preparación de la escena, normalización de instrucciones y coordinación
  de las distintas etapas del flujo experimental.

supervisor.py
  Supervisión y planificación asociada a pi0-DROID y pi0.5-DROID.

ejecucion.py
  Ejecución física protegida del UR5e y del efector terminal.

vision_escena.py
  Construcción de la escena multiobjeto a partir de la captura 3D.

vision_zonas.py
  Procesamiento geométrico de la tarea objeto-zona.

vision_forma.py
  Procesamiento específico de la tarea de clasificación por forma.

photoneo/servidor.py
  Servicio de adquisición Photoneo.

photoneo/captura.py
  Solicitud y validación de una nueva captura Photoneo.

siglip/zonas_cliente.py
  Cliente del grounding externo de la Tarea 1.

siglip/zonas_servidor.py
  Servidor del grounding externo de la Tarea 1.

siglip/forma_cliente.py
  Cliente del grounding específico de la Tarea 2.

siglip/forma_servidor.py
  Servidor del grounding específico de la Tarea 2.

Los conjuntos de instrucciones y los resultados utilizados en la memoria se
conservan de forma separada en instrucciones/ y resultados/.
