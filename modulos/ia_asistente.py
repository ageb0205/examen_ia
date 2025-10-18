import os
from openai import OpenAI
from dotenv import load_dotenv
import time


# Archivo: modulos/ia_asistente.py

# --- 1. PROMPT DE INSTRUCCIONES PARA EL ASISTENTE (RAG) ---
PROMPT_INSTRUCCIONES = (
    "Usted es un Asistente RAG experto en documentos. Su ÚNICA tarea es responder la pregunta de opción múltiple con el siguiente formato estricto de DOS LÍNEAS. NO DEBE incluir NINGÚN texto adicional, introducción o explicación.\n"
    "FUENTES: [Indicar la fuente de la respuesta]\n"
    "RESPUESTA_FINAL: [OPCIÓN] - [JUSTIFICACIÓN DETALLADA]\n"
    
    "\nInstrucciones de Búsqueda ESTRICTAS:\n"
    "1. Jerarquía: Priorice siempre la respuesta basada en el contenido de los documentos provistos.\n"
    "2. Si la respuesta se encuentra en los documentos:\n"
    "   - En el campo FUENTES, coloque ESTRICTAMENTE 'DOCUMENTOS'.\n"
    "   - La JUSTIFICACIÓN debe citar el documento y la página o sección.\n"
    "3. Si la respuesta NO se encuentra en los documentos:\n"
    "   - En el campo FUENTES, coloque ESTRICTAMENTE 'CONOCIMIENTO GENERAL'.\n"
    "   - En la JUSTIFICACIÓN, **EXPLIQUE EL MOTIVO DEL FALLO (Ej: No se encontró contenido relevante que coincida con la pregunta)** y luego proporcione la mejor respuesta posible basada en el conocimiento general, detallando por qué esa opción es la correcta en el contexto de gestión de proyectos."
)

# ... (El resto de tu código sigue igual)

# Carga la clave del archivo .env
load_dotenv()
CLIENTE = OpenAI()
# **SOLUCIÓN AL FALLO 404:** Usamos el ID del asistente que Render ya creó
ASISTENTE_ID = "asst_V4lApoD3ZjBqn9I6nKVdwMbi" 

def inicializar_asistente():
    """
    Simplemente verifica la existencia del ASISTENTE_ID y no intenta crearlo de nuevo
    para evitar el timeout en Render. (El ID es forzado arriba).
    """
    global ASISTENTE_ID
    
    if ASISTENTE_ID != "asst_V4lApoD3ZjBqn9I6nKVdwMbi": # Solo creamos si el ID de reserva está activo
        print(f"✅ Usando Asistente pre-creado: {ASISTENTE_ID}")
        return

    # Si tuvieras que crearlo, el código de creación iría aquí, pero lo hemos saltado 
    # para garantizar que Render se inicie rápido y use el ID conocido.

# --- FUNCIÓN 1: SÓLO RAG (Consulta de PDFs) ---

def consultar_asistente_rag(texto_pregunta: str) -> str:
    """
    Envía el texto transcrito al Asistente RAG para obtener la respuesta de los PDFs,
    creando un nuevo hilo para garantizar la estabilidad.
    """
    if not ASISTENTE_ID:
        return "Error: El asistente de IA no está inicializado correctamente."
        
    try:
        # **CORRECCIÓN FINAL:** Crear un hilo nuevo para esta solicitud
        hilo = CLIENTE.beta.threads.create()
        thread_id_actual = hilo.id 
        
        # 1. Agregar el mensaje del usuario al nuevo hilo
        CLIENTE.beta.threads.messages.create(
            thread_id=thread_id_actual,
            role="user",
            content=texto_pregunta
        )

        # 2. Ejecutar el Asistente
        ejecucion = CLIENTE.beta.threads.runs.create(
            thread_id=thread_id_actual,
            assistant_id=ASISTENTE_ID
        )

        # 3. Esperar a que la ejecución termine
        while ejecucion.status not in ["completed", "failed"]:
            time.sleep(1)
            ejecucion = CLIENTE.beta.threads.runs.retrieve(
                thread_id=thread_id_actual,
                run_id=ejecucion.id
            )
        
        if ejecucion.status == "completed":
            mensajes = CLIENTE.beta.threads.messages.list(thread_id=thread_id_actual, order="desc", limit=1)
            return mensajes.data[0].content[0].text.value
        else:
            return f"Error en la ejecución del Asistente. Estado: {ejecucion.status}"

    except Exception as e:
        print(f"Error al procesar la pregunta con el asistente: {e}")
        return "Error interno al comunicarse con la IA."

# --- FUNCIÓN 2: VISIÓN + RAG ---

def analizar_pregunta_desde_imagen(imagen_base64: str) -> str:
    """
    Usa GPT-4o para transcribir la pregunta (Visión) y luego la pasa al RAG.
    """
    # ... (El código de la función analizar_pregunta_desde_imagen sigue igual)
    from openai import OpenAI
    
    CLIENTE_V = OpenAI()
    
    VISION_PROMPT = "Usted es un transcriptor óptico de documentos. Su única tarea es transcribir el texto visible, identificando el campo 'PREGUNTA:' y el campo 'OPCIONES:' para facilitar el ingreso de datos a un sistema interno. Por favor, devuelva el resultado en el formato estricto: 'PREGUNTA: [texto de la pregunta] OPCIONES: [A)texto, B)texto, C)texto, D)texto]'"
    
    try:
        # 1. PASO DE VISIÓN
        response_vision = CLIENTE_V.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": VISION_PROMPT 
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{imagen_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        texto_transcrito = response_vision.choices[0].message.content
        print(f"Texto transcrito: {texto_transcrito}")

    except Exception as e:
        return f"Error en el paso de visión (GPT-4o): {e}"

    # 2. PASO DE RAG
    print("2. Consultando Asistente RAG con el texto transcrito...")
    respuesta_rag = consultar_asistente_rag(texto_transcrito) 
    
    return respuesta_rag


# Llamada para inicializar al importar el módulo
inicializar_asistente()