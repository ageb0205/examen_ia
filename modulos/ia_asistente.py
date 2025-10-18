import os
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. PROMPT DE INSTRUCCIONES PARA EL ASISTENTE (DEBE ESTAR FUERA DE LA FUNCIÓN) ---
PROMPT_INSTRUCCIONES = (
    "Eres un asistente experto en análisis de documentos ingeniería y gestión de proyectos. "
    "Tu tarea es recibir una pregunta de opción múltiple (transcrita desde una imagen) y proporcionar la respuesta correcta. "
    "Sigue estas reglas estrictamente:\n"
    "1. Busca la información en los documentos que te proporcioné para justificar tu respuesta.\n"
    "2. Tu respuesta debe estar formateada exactamente así, sin texto adicional, para que el móvil pueda leerla:\n"
    "   RESULTADO: [OPCIÓN] - [JUSTIFICACIÓN BREVE]\n"
    "   Ejemplo: RESULTADO: C - Planificar la gestión del alcance es fundamental para el proceso de planificación según los principios del PMBOK, detallado en la página 5 del documento UPN_APS_Semana 5.pdf."
)

# Carga la clave del archivo .env
load_dotenv()
CLIENTE = OpenAI()
# Guarda el ID del asistente una vez creado para no crearlo de nuevo
ASISTENTE_ID = None
ID_HILO_CONVERSACION = None

def inicializar_asistente():
    """
    Sube los PDFs y crea o recupera el Asistente de OpenAI.
    """
    global ASISTENTE_ID, ID_HILO_CONVERSACION
    
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
            # 3. Crear el Asistente con los archivos (USANDO tool_resources)
            print("Creando Asistente con los PDFs...")
            
            file_ids = [f.id for f in archivos_subidos] # Lista de IDs de archivos subidos
            
            asistente = CLIENTE.beta.assistants.create(
                name="Asistente de Exámenes de Clase",
                instructions=PROMPT_INSTRUCCIONES, # Usamos la constante definida arriba
                model="gpt-4o", # gpt-4o para visión y retrieval
                tools=[{"type": "file_search"}], # Habilitar búsqueda de archivos
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

def procesar_pregunta(texto_pregunta: str) -> str:
    """
    Envía el texto de la pregunta al Asistente y espera la respuesta.
    """
    global ID_HILO_CONVERSACION
    
    if not ASISTENTE_ID:
        return "Error: El asistente de IA no está inicializado correctamente."
        
    try:
        # Si no hay un hilo, crea uno nuevo para esta conversación
        if not ID_HILO_CONVERSACION:
            hilo = CLIENTE.beta.threads.create()
            ID_HILO_CONVERSACION = hilo.id

        # 1. Agregar el mensaje del usuario al hilo
        CLIENTE.beta.threads.messages.create(
            thread_id=ID_HILO_CONVERSACION,
            role="user",
            content=texto_pregunta
        )

        # 2. Ejecutar el Asistente para generar la respuesta
        ejecucion = CLIENTE.beta.threads.runs.create(
            thread_id=ID_HILO_CONVERSACION,
            assistant_id=ASISTENTE_ID
        )

        # 3. Esperar a que la ejecución termine (Simplificado)
        import time
        while ejecucion.status not in ["completed", "failed"]:
            time.sleep(1) # Espera 1 segundo antes de chequear el estado
            ejecucion = CLIENTE.beta.threads.runs.retrieve(
                thread_id=ID_HILO_CONVERSACION,
                run_id=ejecucion.id
            )
        
        if ejecucion.status == "completed":
            # 4. Obtener el último mensaje de la respuesta del Asistente
            mensajes = CLIENTE.beta.threads.messages.list(thread_id=ID_HILO_CONVERSACION, order="desc", limit=1)
            # Retorna el contenido del mensaje
            return mensajes.data[0].content[0].text.value
        else:
            return f"Error en la ejecución del Asistente. Estado: {ejecucion.status}"

    except Exception as e:
        print(f"Error al procesar la pregunta con el asistente: {e}")
        return "Error interno al comunicarse con la IA."

# Llamada para inicializar al importar el módulo
inicializar_asistente()