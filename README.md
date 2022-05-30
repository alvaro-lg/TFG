# Trabajo de Fin de Grado: Interpretación de música de piano usando técnicas de Deep Learning.

Repositorio creado para el desarrollo del código del Trabajo de Fin de Grado para acceder al Grado en Ingeniería 
Informática (Facultad de Ciencias - Universidad de Cantabria).

## Resumen

La correcta interpretación de la música es una tarea extremadamente compleja. Los intérpretes profesionales dedican toda
una vida a perfeccionarse en esta labor. La preparación requerida junto con los costes asociados a la infraestructura 
hace que la producción de una determinada obra sea altamente costosa. Y pese a esto, la ciencia aún no nos ha brindado 
una solución que no requiera de tantos recursos.

El objetivo de este Trabajo Fin de Grado es el desarrollo de un agente inteligente basado en técnicas de Aprendizaje 
Profundo (en inglés, Deep Learning) que sea capaz de interpretar música de piano de la forma más humana posible. Hasta 
ahora, las técnicas existentes para la generación de interpretaciones se basaban en grandes bancos de sonido y las 
interpretaciones que ofrecían como resultado eran evidentemente artificiales. Es por esto por lo que se desea crear un 
agente que no precise de tales cantidades de memoria y que realice interpretaciones indistinguibles de las realizadas 
por humanos.

Para el desarrollo de dicho agente se emplearán Redes Neuronales Autorregresivas (ARNNs), dados los buenos resultados 
que estas han mostrado en tareas de Text-To-Speech (TTS), que a su vez presentan una gran similitud con el tema abordado 
(el artículo de referencia para este trabajo es [Oord2016](https://arxiv.org/abs/1609.03499)). El entrenamiento de dicho 
modelo se realizará sobre el [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) (MIDI and Audio Edited 
for Synchronous TRacks and Organization) y su implementación se realizará en Python, usando la librería TensorFlow.

## Abstract

The correct interpretation of music is an extremely complex task. Professional interpreters dedicate a lifetime to 
perfecting themselves in this work. The preparation required together with the costs associated with the infrastructure 
makes the production of a given work highly expensive. And despite this, science has not yet provided us with a solution 
that does not require so many resources.

The objective of this Final Degree Project is the development of an intelligent agent based on Deep Learning techniques 
that is capable of interpreting piano music in the most human way possible. Until now, existing techniques for 
generating performances were based on large banks of sounds, and the resulting performances were patently artificial. 
This is why it is desired to create an agent that does not require such amounts of memory and that performs 
interpretations indistinguishable from those made by humans.

For the development of said agent, Autoregressive Neural Networks (ARNNs) will be used, given the good results that 
these have shown in Text-To-Speech (TTS) tasks, which in turn present a great similarity with the topic addressed (the 
article reference for this work is [Oord2016](https://arxiv.org/abs/1609.03499)). The training of said model will be 
carried out on the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) (MIDI and Audio Edited for 
Synchronous TRAcks and Organization) and its implementation will be carried out in Python, using the TensorFlow library.

## Contacto

Para más información contactar con:
* **Alumno:** Álvaro López García ([alvaro.lopezgar@alumnos.unican.es](mailto:alvaro.lopezgar@alumnos.unican.es)).
* **Director:** Cristina Tîrnăucă ([cristina.tirnauca@unican.es](mailto:cristina.tirnauca@unican.es)).