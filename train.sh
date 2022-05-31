echo "Transformando notebook a script de python..."
jupyter nbconvert --to python TFG.ipynb
echo "Descargando dataset..."
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
echo "Descomprimiendo dataset..."
unzip maestro-v3.0.0.zip && trash maestro-v3.0.0.zip
echo "Ejecutando script de entrenamiento..."
python3 TFG.py
