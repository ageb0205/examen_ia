import os
import base64
from flask import Flask, request, jsonify
# Importamos las funciones necesarias desde el módulo modularizado
from modulos.ia_asistente import inicializar_asistente, analizar_pregunta_desde_imagen 

app = Flask(__name__)

# Configuración: Límite de tamaño de archivo para la foto
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# --- RUTAS DE LA API ---

@app.route('/', methods=['GET'])
def status_check():
    """Ruta raíz para verificar que el servidor de Render está activo."""
    return "Backend del Asistente IA funcionando correctamente.", 200

@app.route('/procesar_pregunta', methods=['POST'])
def procesar_imagen_pregunta():
    """Recibe la imagen, la analiza con Visión (GPT-4o) y consulta al Asistente RAG."""
    if not request.json or 'imagen_base64' not in request.json:
        return jsonify({"error": "Falta el campo 'imagen_base64' en el cuerpo de la solicitud."}), 400

    imagen_base64 = request.json['imagen_base64']

    try:
        # Toda la lógica de Visión y RAG ahora debe estar en analizar_pregunta_desde_imagen
        respuesta_ia = analizar_pregunta_desde_imagen(imagen_base64)

        return jsonify({
            "status": "success",
            "respuesta_correcta": respuesta_ia
        }), 200

    except Exception as e:
        app.logger.error(f"Error procesando la solicitud: {e}")
        return jsonify({"status": "error", "error": f"Error interno del servidor: {e}"}), 500


if __name__ == '__main__':
    # Esto asegura que el Asistente de IA (con los PDFs) se inicialice al arrancar el servidor
    inicializar_asistente()
    
    print("Servidor Flask arrancando. ¡Listo para recibir fotos!")
    app.run(debug=True, host='0.0.0.0', port=5000)