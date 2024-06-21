# Generación procedural de dungeons con puzzles para videojuegos

## Dependencias
- pip install virtualenv
- virtualenv -p python3 env
- .\env\Scripts\activate
- pip install -r dependencies.txt
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## Metodo de busqueda (Search Base)
- Revisar archivo main-SB.py para modificar variables si se desea
- para ejecutar el algoritmo se realiza con `py main-SB.py`

## Autosimilitud (Similarity)
- Configurar la ruta de la carpeta que contenga los mapas csv en el codigo `similarity.py`
- Ejecutar codigo `similarity.py`

## Dcgan experimental
- La implementacion de esta red fue pensada para ser entrenada en una tarjeta de video dedicada.
- Para modificar la configuracion revisar el archivo Dcgan/main.py
- Para ejecutar la wgan, se debe ocupar el comando `py Dcgan/main.py`.