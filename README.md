# GAN con TensorFlow y Keras para MNIST
Construiremos una Red Generativa Antagónica (GAN) utilizando TensorFlow y Keras para generar imágenes similares a las del dataset MNIST.

## Autores
- Luis Pedro González Aldana
- Rebecca Smith

## Componentes
1. Generador (G(x))
Genera imágenes a partir de un vector de ruido, utilizando capas densas y de BatchNormalization.

2. Discriminador (D(x))
Clasifica si una imagen es real (del dataset MNIST) o falsa (generada), utilizando capas densas.

## Construcción de Modelos

Generador

```python
def build_generator():
    model = models.Sequential([
        layers.Dense(256, input_dim=100, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(784, activation='tanh')
    ])
    return model
```

Discriminador

```python
def build_discriminator():
    model = models.Sequential([
        layers.Dense(1024, input_dim=784, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```

GAN

```python
def build_gan(generator, discriminator):
    model = models.Sequential([
        generator,
        discriminator
    ])
    return model
```

## Entrenamiento
1. Preprocesamiento:

- Cargar y normalizar el dataset MNIST.
- Configurar los parámetros de entrenamiento como el número de epochs y el tamaño del batch.

2. Compilación:

- Compilar el modelo del Discriminador.
- Compilar el modelo de la GAN (con el Discriminador congelado).
3. Ejecución:

Entrenar el Discriminador y el Generador en un bucle, alternando entre ellos.
python
Copy code
epochs = 20000
batch_size = 64
half_batch = batch_size // 2

for epoch in range(epochs):
    # Entrenar el Discriminador y el Generador aquí...
Visualización del Progreso
En cada época, o después de un número específico de épocas, imprime las pérdidas y la precisión del Discriminador y del Generador.

python
Copy code
if epoch % 1000 == 0:
    print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
Próximos Pasos
Añadir funciones para visualizar o guardar las imágenes generadas.
Realizar ajustes y optimizaciones en los modelos y parámetros de entrenamiento para mejorar los resultados.
Experimentar con diferentes arquitecturas de red, funciones de pérdida, y optimizadores.
Este Markdown ofrece un resumen y la estructura básica de cómo construir y entrenar una GAN con TensorFlow y Keras para el dataset MNIST, sirviendo como punto de partida para exploraciones y mejoras adicionales.






Regenerate
