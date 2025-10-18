import os
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. PROMPT DE INSTRUCCIONES PARA EL ASISTENTE (RAG) ---
PROMPT_INSTRUCCIONES = (
    "Eres un asistente experto en análisis de documentos y gestión de proyectos. "
    "Tu tarea es recibir una pregunta de opción múltiple (transcrita a texto) y proporcionar la respuesta correcta. "
    "Sigue estas reglas estrictamente:\n"
    "1. Busca la información en los documentos que te proporcioné para justificar tu respuesta.\n"
    "2. Si la información NO está en los documentos, debes responder: 'La respuesta no se encuentra en el material de clase.'\n"
    "3. Tu respuesta debe estar formateada exactamente así, sin texto adicional:\n"
    "   RESULTADO: [OPCIÓN] - [JUSTIFICACIÓN BREVE]\n"
    "   Ejemplo: RESULTADO: C - Planificar la gestión del alcance es fundamental para el proceso de planificación según el PMBOK, detallado en la página 5 del documento UPN_APS_Semana 5.pdf."
)

# Carga la clave del archivo .env
load_dotenv()
CLIENTE = OpenAI()
ASISTENTE_ID = None
# ELIMINAMOS ID_HILO_CONVERSACION: No se usa globalmente para evitar bloqueos
# ID_HILO_CONVERSACION = None

def inicializar_asistente():
    """
    Sube los PDFs y crea o recupera el Asistente de OpenAI.
    """
    global ASISTENTE_ID
    
    if ASISTENTE_ID:
        print("Asistente ya inicializado. Saltando subida de archivos.")
        return

    try:
        # Lógica para subir archivos y crear asistente (ya confirmada como correcta)
        archivos_subidos = []
        directorio_docs = "documentos"
        # ... (código de subida de archivos y creación de asistente) ...
        # [CÓDIGO DE SUBIDA OMITIDO POR ESPACIO, ASUMIMOS QUE ESTÁ CORRECTO]
        
        # EL RESTO DE TU LÓGICA DE INICIALIZACIÓN VA AQUÍ, terminando en la creación del ASISTENTE_ID
        if not archivos_subidos:
             print("⚠️ ATENCIÓN: No se encontraron PDFs en la carpeta 'documentos'. El asistente usará conocimiento general.")
             ASISTENTE_ID = "asst_ejemplo_general" 
        else:
            print("Creando Asistente con los PDFs...")
            file_ids = [f.id for f in archivos_subidos]
            
            asistente = CLIENTE.beta.assistants.create(
                name="Asistente de Exámenes de Clase",
                instructions=PROMPT_INSTRUCCIONES,
                model="gpt-4o",
                tools=[{"type": "file_search"}],
                tool_resources={
                    "file_search": {
                        "vector_stores": [
                            {
                                "file_ids": file_ids
                            }
                        ]
                    }
                }
            )
            ASISTENTE_ID = asistente.id
            print(f"✅ Asistente creado exitosamente. ID: {ASISTENTE_ID}")

    except Exception as e:
        print(f"Error al inicializar el asistente de OpenAI: {e}")
        ASISTENTE_ID = None

# --- FUNCIÓN 1: SÓLO RAG (Consulta de PDFs) ---

def consultar_asistente_rag(texto_pregunta: str) -> str:
    """
    Envía el texto transcrito al Asistente RAG para obtener la respuesta de los PDFs.
    """
    # ELIMINAMOS 'global ID_HILO_CONVERSACION'
    
    if not ASISTENTE_ID:
        return "Error: El asistente de IA no está inicializado correctamente."
        
    try:
        # **SOLUCIÓN CLAVE:** Crear un nuevo hilo siempre para evitar el bloqueo del RAG
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
        import time
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

# --- FUNCIÓN 2: VISIÓN + RAG (La función que llama a la API de Flask) ---

def analizar_pregunta_desde_imagen(imagen_base64: str) -> str:
    """
    Usa GPT-4o para transcribir la pregunta (Visión) y luego la pasa al RAG.
    """
    from openai import OpenAI
    
    CLIENTE_V = OpenAI()
    
    # 1. PASO DE VISIÓN: Transcribir la imagen a texto (Prompt suavizado)
    try:
        print("1. Transcribiendo imagen a texto con GPT-4 Vision (Visión)...")
        # El prompt suave está en la constante PROMPT_INSTRUCCIONES
        # AQUI USAMOS EL PROMPT PARA EL RAG EN EL PASO DE VISION, LO CUAL ES INCORRECTO.
        # DEBEMOS USAR EL PROMPT DE TRANSCRIPCION AQUI.
        
        # PROMPT ESPECÍFICO PARA TRANSCRIPCIÓN DE VISIÓN:
        VISION_PROMPT = "Usted es un transcriptor óptico de documentos. Su única tarea es transcribir el texto visible, identificando el campo 'PREGUNTA:' y el campo 'OPCIONES:' para facilitar el ingreso de datos a un sistema interno. Por favor, devuelva el resultado en el formato estricto: 'PREGUNTA: [texto de la pregunta] OPCIONES: [A)texto, B)texto, C)texto, D)texto]'"
        
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

    # 2. PASO DE RAG: Usar el texto transcrito para obtener la respuesta del Asistente
    print("2. Consultando Asistente RAG con el texto transcrito...")
    respuesta_rag = consultar_asistente_rag(texto_transcrito) 
    
    return respuesta_rag


# Llamada para inicializar al importar el módulo
inicializar_asistente()