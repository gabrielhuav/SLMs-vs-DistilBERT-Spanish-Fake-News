# 📰 Evaluador Comparativo de Modelos de Lenguaje (SLMs vs DistilBERT)

Este repositorio contiene un framework de evaluación automatizada diseñado para comparar el rendimiento de un modelo codificador ligero con ajuste fino (**DistilBERT**) frente a Modelos de Lenguaje Locales (**SLMs**) en la tarea de clasificación de noticias falsas, verdaderas y sátira periodística en el contexto mexicano.

## 🎯 Objetivo de la Investigación
El objetivo de este proyecto es demostrar empíricamente el impacto del sesgo cultural y la necesidad de anclajes semánticos locales (*Few-Shot Prompting*) en modelos genéricos frente a modelos especializados de 3 clases entrenados con corpus regionales (ej. "El Deforma").

## 🛠️ Arquitectura e Infraestructura Local
El entorno está completamente dockerizado y optimizado para ejecutarse localmente sin conexión a internet, garantizando la privacidad de los datos y aprovechando la aceleración por hardware (NVIDIA RTX 4060 - 8GB VRAM).

* **Backend de Inferencia:** `Ollama` (Contenedor con acceso a GPU).
* **Frontend y Orquestación:** `Streamlit` + `Python 3.11`.
* **Modelos Evaluados (Fase Actual):** LLaMA 3.2 (11B/3B).
* **Métricas Extraídas:** Matriz de Confusión y Classification Report (F1-Score Macro/Micro) vía `scikit-learn`.

## 📁 Estructura del Proyecto
* `app.py`: Interfaz web y lógica de inferencia (Zero-Shot / Few-Shot).
* `docker-compose.yml`: Orquestador de servicios (Ollama + Streamlit).
* `Dockerfile`: Receta de construcción del entorno de evaluación.
* `requirements.txt`: Dependencias estandarizadas del proyecto.

## 🚀 Instalación y Uso
1. Clonar el repositorio.
2. Iniciar la infraestructura: `docker compose up -d --build`
3. Descargar el modelo base en Ollama: `docker exec -it ollama_server ollama run llama3.2`
4. Acceder a la interfaz web en: `http://localhost:8501`

---
*Investigación desarrollada para publicación académica - gabrielhuav.*