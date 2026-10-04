# Verificación para GitHub

- [Respaldo Khipu](khipu-backup-20261003.json): firmas, corridas incluidas y cotejo de métricas.
- [Preparación local](repository-readiness.json): inventario publicable, enlaces, pruebas y límites de verificación.
- [Suite de pruebas](tests-repository.xml): resultado en una copia temporal que contiene únicamente los archivos admitidos por Git.

Las pruebas se ejecutan sin los HDF5 reales ni checkpoints. La verificación SHA-256 del respaldo se realiza por separado sobre los archivos ya descargados, sin SSH. La ejecución remota de GitHub Actions queda pendiente del primer push del propietario.
