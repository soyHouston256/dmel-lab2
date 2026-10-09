# Verificación para GitHub

- [Cierre del 08/10/2026](cierre4h-local-20261008/README.md): ablación recalculada, ambos tests de 27 983 muestras y manifiesto de la nueva descarga.
- [Cumplimiento actualizado](laboratorio-2-cierre-verificado-20261008.json): reemplaza como estado la auditoría previa a ejecutar ablación/test; incluye el video final y distingue preparación local de publicación/entrega.
- [Video final](video-final-20261008.json): 17:49.84, hashes del MP4/DRP y comprobaciones de imagen, voz y cámara.
- [Preparación para el siguiente push](repository-readiness-20261008.json): inventario vigente y validación en copia sin datos ni medios.
- [Pruebas del 08/10/2026](tests-repository-20261008.xml): 89 aprobadas y 1 omitida por requerir Linux; ejecutadas sobre la selección de archivos para el siguiente push.
- [Respaldo Khipu](khipu-backup-20261003.json): firmas, corridas incluidas y cotejo de métricas.
- [Preparación local](repository-readiness.json): inventario publicable, enlaces, pruebas y límites de verificación.
- [Suite de pruebas](tests-repository.xml): resultado en una copia temporal que contiene únicamente los archivos admitidos por Git.

Las pruebas se ejecutan sin los HDF5 reales ni checkpoints. La verificación SHA-256 del respaldo se realiza por separado sobre los archivos ya descargados, sin SSH. Los informes locales no acreditan el estado de GitHub Actions; consultar la ejecución remota después del siguiente push.
