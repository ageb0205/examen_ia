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
ID_HILO_CONVERSACION = None

def inicializar_asistente():
    """
    Sube los PDFs y crea o recupera el Asistente de OpenAI.
    """
    global ASISTENTE_ID
    
    if ASISTENTE_ID:
        print("Asistente ya inicializado. Saltando subida de archivos.")
        return

    try:
        # 2. Subir archivos
        archivos_subidos = []
        directorio_docs = "documentos"
        print(f"Buscando archivos en: {directorio_docs}")
        
        for nombre_archivo in os.listdir(directorio_docs):
            if nombre_archivo.endswith(".pdf"):
                ruta_completa = os.path.join(directorio_docs, nombre_archivo)
                print(f"Subiendo archivo: {nombre_archivo}...")
                
                with open(ruta_completa, "rb") as file:
                    archivo_openai = CLIENTE.files.create(file=file, purpose="assistants")
                    archivos_subidos.append(archivo_openai)
                    print(f"Archivo subido. ID: {archivo_openai.id}")

        if not archivos_subidos:
             print("⚠️ ATENCIÓN: No se encontraron PDFs en la carpeta 'documentos'. El asistente usará conocimiento general.")
             ASISTENTE_ID = "asst_ejemplo_general" 
        else:
            # 3. Crear el Asistente (RAG)
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
    global ID_HILO_CONVERSACION
    
    if not ASISTENTE_ID:
        return "Error: El asistente de IA no está inicializado correctamente."
        
    try:
        if not ID_HILO_CONVERSACION:
            hilo = CLIENTE.beta.threads.create()
            ID_HILO_CONVERSACION = hilo.id

        CLIENTE.beta.threads.messages.create(
            thread_id=ID_HILO_CONVERSACION,
            role="user",
            content=texto_pregunta
        )

        ejecucion = CLIENTE.beta.threads.runs.create(
            thread_id=ID_HILO_CONVERSACION,
            assistant_id=ASISTENTE_ID
        )

        import time
        while ejecucion.status not in ["completed", "failed"]:
            time.sleep(1)
            ejecucion = CLIENTE.beta.threads.runs.retrieve(
                thread_id=ID_HILO_CONVERSACION,
                run_id=ejecucion.id
            )
        
        if ejecucion.status == "completed":
            mensajes = CLIENTE.beta.threads.messages.list(thread_id=ID_HILO_CONVERSACION, order="desc", limit=1)
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
        response_vision = CLIENTE_V.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Usted es un transcriptor óptico de documentos. Su única tarea es transcribir el texto visible, identificando el campo 'PREGUNTA:' y el campo 'OPCIONES:' para facilitar el ingreso de datos a un sistema interno. Por favor, devuelva el resultado en el formato estricto: 'PREGUNTA: [texto de la pregunta] OPCIONES: [A)texto, B)texto, C)texto, D)texto]'"
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
            max_tokens=500 # Más tokens para evitar truncamiento
        )
        texto_transcrito = response_vision.choices[0].message.content
        print(f"Texto transcrito: {texto_transcrito}")

    except Exception as e:
        return f"Error en el paso de visión (GPT-4o): {e}"

    # 2. PASO DE RAG: Usar el texto transcrito para obtener la respuesta del Asistente
    print("2. Consultando Asistente RAG con el texto transcrito...")
    # Llamamos a la función RAG definida arriba
    respuesta_rag = consultar_asistente_rag(texto_transcrito) 
    
    return respuesta_rag


# Llamada para inicializar al importar el módulo
inicializar_asistente()