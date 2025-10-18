import os
import base64
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

# Importamos el módulo de IA que creaste
from modulos.ia_asistente import procesar_pregunta

app = Flask(__name__)

# --- Configuración (Opcional, pero buena práctica) ---
# Tamaño máximo del archivo de la foto (ej. 16 megabytes)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# --- Endpoint principal de la API ---
@app.route('/procesar_pregunta', methods=['POST'])
def procesar_imagen_pregunta():
    """
    Recibe la imagen de la pregunta en formato Base64 desde el celular.
    1. Llama a la API de OpenAI Vision para transcribir la pregunta y las opciones.
    2. Envía la transcripción al Asistente RAG para obtener la respuesta basada en los PDFs.
    """
    # 1. Verificar que la solicitud sea correcta
    if 'imagen_base64' not in request.json:
        return jsonify({"error": "Falta el campo 'imagen_base64' en el cuerpo de la solicitud."}), 400

    imagen_base64 = request.json['imagen_base64']

    try:
        # 2. Convertir la cadena Base64 a una imagen binaria
        datos_imagen = base64.b64decode(imagen_base64)
        
        # 3. Guardar la imagen temporalmente para enviarla a OpenAI (o puedes usar la ruta directa de Base64)
        # Para usar la API de Vision de OpenAI de forma eficiente, la codificación Base64 es la forma más limpia.
        # No necesitamos transcribir la imagen aquí, se la pasaremos a GPT-4 Vision directamente.

        # 4. Crear el mensaje para la IA (Usaremos la capacidad de GPT-4 Vision)
        # OJO: Aunque el módulo 'ia_asistente.py' usa el Asistente RAG (con documentos),
        # la API de Asistentes no soporta visión directamente. 
        # Para mantener el requisito de visión + RAG, debemos hacer un paso intermedio.

        # --- Enfoque simple y efectivo para este proyecto (Requisito de MVP) ---
        # Como es un MVP, haremos que el móvil envíe el texto *extraído* o,
        # más fácil, usaremos un modelo de Visión de GPT-4 para transcribir la imagen
        # y luego pasamos el texto al Asistente RAG.

        # Por simplicidad del *endpoint* y modularidad, aquí SOLO llamaremos a la función
        # que manejará tanto la visión como el RAG.
        
        # En el diseño de la API actual, le estamos pasando una imagen y el RAG espera texto.
        # Vamos a modificar el llamado a la IA para manejar esto:
        
        # El string base64 se convierte en la entrada al Asistente.
        respuesta_ia = procesar_pregunta_vision_y_rag(imagen_base64)

        # 5. Devolver la respuesta al cliente (móvil)
        return jsonify({
            "status": "success",
            "respuesta_correcta": respuesta_ia
        }), 200

    except Exception as e:
        app.logger.error(f"Error procesando la solicitud: {e}")
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500


# --- NUEVA FUNCIÓN NECESARIA para unir Visión (GPT-4 V) con RAG (Asistente) ---
# Esta función debe estar en 'modulos/ia_asistente.py', pero la pondremos aquí por ahora 
# para que 'app.py' pueda funcionar inmediatamente. 
# NOTA: En un proyecto real, esto iría en la carpeta 'modulos'.
def procesar_pregunta_vision_y_rag(imagen_base64: str) -> str:
    """
    Usa GPT-4V para transcribir la pregunta y luego pasa el texto al Asistente RAG.
    """
    from openai import OpenAI
    import json
    
    CLIENTE_V = OpenAI()
    
    # 1. PASO DE VISIÓN: Transcribir la imagen a texto con GPT-4 Vision
    try:
        print("1. Transcribiendo imagen a texto con GPT-4 Vision...")
        response_vision = CLIENTE_V.chat.completions.create(
            model="gpt-4o", # Modelo multimodal que maneja texto e imagen eficientemente
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe la pregunta y todas las opciones de respuesta exacta y fielmente. Luego, formatea el resultado como una pregunta simple de texto: 'PREGUNTA: [texto de la pregunta] OPCIONES: [A)texto, B)texto, C)texto, D)texto]'"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{imagen_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        texto_transcrito = response_vision.choices[0].message.content
        print(f"Texto transcrito: {texto_transcrito}")

    except Exception as e:
        return f"Error en el paso de visión (GPT-4o): {e}"

    # 2. PASO DE RAG: Usar el texto transcrito para obtener la respuesta del Asistente (RAG)
    # Importamos la función real del módulo
    from modulos.ia_asistente import procesar_pregunta 
    
    print("2. Consultando Asistente RAG con el texto transcrito...")
    respuesta_rag = procesar_pregunta(texto_transcrito)
    
    return respuesta_rag


if __name__ == '__main__':
    # Esto asegura que el Asistente de IA (con los PDFs) se inicialice al arrancar el servidor
    from modulos.ia_asistente import inicializar_asistente
    inicializar_asistente()
    
    print("Servidor Flask arrancando. ¡Listo para recibir fotos!")
    # Para fines de desarrollo local, usa el host '0.0.0.0'
    app.run(debug=True, host='0.0.0.0', port=5000)