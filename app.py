import streamlit as st
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple
import io
from collections import Counter
import uuid

# Configuración de la página
st.set_page_config(
    page_title="JSON Consolidator AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FUNCIONES DE PROCESAMIENTO ====================

def generate_checksum(data: str) -> str:
    """Genera checksum MD5 de los datos"""
    return hashlib.md5(data.encode()).hexdigest()

def detect_json_structure(data: Any) -> str:
    """Detecta el tipo de estructura JSON"""
    if isinstance(data, list):
        return "array"
    elif isinstance(data, dict):
        if any(key in data for key in ['items', 'data', 'content', 'records']):
            return "object_with_array"
        return "object"
    return "unknown"

def extract_items(data: Any, structure: str) -> List[Dict]:
    """Extrae items del JSON según su estructura"""
    if structure == "array":
        return data if isinstance(data, list) else [data]
    elif structure == "object_with_array":
        # Busca arrays en el objeto
        for key in ['items', 'data', 'content', 'records', 'results']:
            if key in data and isinstance(data[key], list):
                return data[key]
        # Si no encuentra, devuelve el objeto como único item
        return [data]
    else:
        return [data]

def normalize_item(item: Any, file_name: str, index: int, 
                   generate_ids: bool, include_metadata: bool) -> Dict:
    """Normaliza un item con metadatos opcionales"""
    normalized = {
        "id": str(uuid.uuid4()) if generate_ids else f"{file_name}_{index}",
        "data": item if isinstance(item, dict) else {"value": item}
    }
    
    if include_metadata:
        normalized["metadata"] = {
            "source_file": file_name,
            "added_timestamp": datetime.utcnow().isoformat() + "Z",
            "original_index": index
        }
    
    return normalized

def detect_content_type(item: Dict) -> str:
    """Detecta el tipo de contenido basado en las keys"""
    keys = set(item.keys()) if isinstance(item, dict) else set()
    
    # Patrones comunes
    if {'question', 'answer'} & keys or {'q', 'a'} & keys:
        return "qa_pair"
    elif {'instruction', 'input', 'output'} & keys:
        return "instruction"
    elif {'prompt', 'completion'} & keys or {'prompt', 'response'} & keys:
        return "conversation"
    elif {'text', 'label'} & keys or {'content', 'category'} & keys:
        return "classification"
    elif 'description' in keys or 'content' in keys:
        return "document"
    else:
        return "generic"

def process_files(uploaded_files: List, config: Dict) -> Tuple[Dict, List[str]]:
    """Procesa todos los archivos y genera el JSON consolidado"""
    errors = []
    all_items = []
    source_files_info = []
    content_types = []
    
    for uploaded_file in uploaded_files:
        try:
            # Leer archivo
            content = uploaded_file.read()
            file_size = len(content)
            
            # Parsear JSON
            try:
                data = json.loads(content.decode('utf-8'))
            except json.JSONDecodeError as e:
                errors.append(f"❌ {uploaded_file.name}: Error de formato JSON - {str(e)}")
                continue
            
            # Detectar estructura
            structure = detect_json_structure(data)
            items = extract_items(data, structure)
            
            # Normalizar items
            normalized_items = []
            for idx, item in enumerate(items):
                normalized = normalize_item(
                    item, 
                    uploaded_file.name, 
                    idx,
                    config['generate_ids'],
                    config['include_metadata']
                )
                
                # Detectar tipo de contenido
                content_type = detect_content_type(item if isinstance(item, dict) else {})
                normalized['content_type'] = content_type
                content_types.append(content_type)
                
                normalized_items.append(normalized)
            
            all_items.extend(normalized_items)
            
            # Información del archivo fuente
            source_files_info.append({
                "filename": uploaded_file.name,
                "file_size_bytes": file_size,
                "items_contributed": len(items),
                "detected_structure": structure
            })
            
        except Exception as e:
            errors.append(f"❌ {uploaded_file.name}: Error inesperado - {str(e)}")
    
    # Eliminar duplicados si está habilitado
    if config['remove_duplicates']:
        original_count = len(all_items)
        # Usar hash del data para detectar duplicados
        seen = set()
        unique_items = []
        for item in all_items:
            item_hash = hashlib.md5(json.dumps(item['data'], sort_keys=True).encode()).hexdigest()
            if item_hash not in seen:
                seen.add(item_hash)
                unique_items.append(item)
        all_items = unique_items
        if original_count > len(all_items):
            errors.append(f"ℹ️ Se eliminaron {original_count - len(all_items)} duplicados")
    
    # Construir dataset consolidado
    content_distribution = dict(Counter(content_types))
    
    consolidated = {
        "dataset_metadata": {
            "name": config.get('dataset_name', 'Dataset Consolidado'),
            "version": config.get('version', '1.0'),
            "created_date": datetime.utcnow().isoformat() + "Z",
            "total_source_files": len(source_files_info),
            "total_items": len(all_items),
            "consolidation_strategy": config['consolidation_mode']
        },
        "source_files": source_files_info,
        "content": all_items,
        "statistics": {
            "content_types_distribution": content_distribution,
            "items_per_file_avg": round(len(all_items) / max(len(source_files_info), 1), 2),
            "total_size_kb": sum(f['file_size_bytes'] for f in source_files_info) / 1024
        }
    }
    
    # Agregar checksum
    json_str = json.dumps(consolidated['content'], sort_keys=True)
    consolidated['dataset_metadata']['checksum'] = generate_checksum(json_str)
    
    return consolidated, errors

# ==================== INTERFAZ DE USUARIO ====================

# Header
st.title("🤖 JSON Consolidator para Entrenamiento de IA")
st.markdown("""
Esta aplicación consolida múltiples archivos JSON en un dataset optimizado para entrenar agentes de IA.
Soporta diferentes estructuras JSON y ofrece opciones avanzadas de procesamiento.
""")

# Sidebar - Configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("Dataset")
    dataset_name = st.text_input("Nombre del Dataset", "Dataset Consolidado")
    version = st.text_input("Versión", "1.0")
    
    st.subheader("Opciones de Consolidación")
    consolidation_mode = st.selectbox(
        "Modo de Consolidación",
        ["array_simple", "hierarchical"],
        help="Array simple: lista plana de items. Jerárquica: agrupa por fuente"
    )
    
    include_metadata = st.checkbox(
        "Incluir Metadatos de Origen",
        value=True,
        help="Agrega información sobre el archivo fuente y timestamp"
    )
    
    generate_ids = st.checkbox(
        "Generar IDs Únicos (UUID)",
        value=True,
        help="Crea identificadores únicos para cada item"
    )
    
    remove_duplicates = st.checkbox(
        "Eliminar Duplicados",
        value=True,
        help="Detecta y elimina items duplicados basándose en su contenido"
    )
    
    st.subheader("Formato de Salida")
    output_format = st.radio(
        "Formato JSON",
        ["Legible (indentado)", "Minificado (compacto)"],
        help="Legible para revisión, minificado para producción"
    )
    
    include_schema = st.checkbox(
        "Incluir Schema de Validación",
        value=False,
        help="Agrega un JSON Schema para validar el dataset"
    )

# Área principal - Carga de archivos
st.header("📁 Cargar Archivos JSON")

uploaded_files = st.file_uploader(
    "Arrastra múltiples archivos JSON o haz clic para seleccionar",
    type=['json'],
    accept_multiple_files=True,
    help="Puedes cargar hasta 50 archivos JSON simultáneamente"
)

if uploaded_files:
    # Mostrar resumen de archivos
    st.subheader(f"📊 Archivos Cargados: {len(uploaded_files)}")
    
    col1, col2, col3 = st.columns(3)
    
    total_size = sum(f.size for f in uploaded_files)
    with col1:
        st.metric("Total Archivos", len(uploaded_files))
    with col2:
        st.metric("Tamaño Total", f"{total_size / 1024:.2f} KB")
    with col3:
        st.metric("Tamaño Promedio", f"{total_size / len(uploaded_files) / 1024:.2f} KB")
    
    # Lista de archivos
    with st.expander("Ver lista de archivos"):
        for f in uploaded_files:
            st.text(f"• {f.name} ({f.size / 1024:.2f} KB)")
    
    # Botón de procesamiento
    if st.button("🚀 Consolidar Archivos", type="primary", use_container_width=True):
        with st.spinner("Procesando archivos..."):
            # Configuración
            config = {
                'dataset_name': dataset_name,
                'version': version,
                'consolidation_mode': consolidation_mode,
                'include_metadata': include_metadata,
                'generate_ids': generate_ids,
                'remove_duplicates': remove_duplicates
            }
            
            # Procesar
            consolidated, errors = process_files(uploaded_files, config)
            
            # Mostrar errores si existen
            if errors:
                st.warning("⚠️ Advertencias durante el procesamiento:")
                for error in errors:
                    st.text(error)
            
            # Guardar en session state
            st.session_state['consolidated'] = consolidated
            st.session_state['config'] = config
            
            st.success(f"✅ Consolidación completada: {consolidated['dataset_metadata']['total_items']} items procesados")

# Mostrar resultados
if 'consolidated' in st.session_state:
    st.header("📈 Resultados")
    
    consolidated = st.session_state['consolidated']
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Items Totales", consolidated['dataset_metadata']['total_items'])
    with col2:
        st.metric("Archivos Fuente", consolidated['dataset_metadata']['total_source_files'])
    with col3:
        st.metric("Promedio Items/Archivo", consolidated['statistics']['items_per_file_avg'])
    with col4:
        st.metric("Tamaño Total", f"{consolidated['statistics']['total_size_kb']:.2f} KB")
    
    # Distribución de tipos
    st.subheader("📊 Distribución de Tipos de Contenido")
    content_dist = consolidated['statistics']['content_types_distribution']
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(content_dist)
    with col2:
        for ctype, count in content_dist.items():
            st.metric(ctype, count)
    
    # Vista previa
    st.subheader("👁️ Vista Previa del Dataset")
    
    tab1, tab2, tab3 = st.tabs(["Primeros Items", "Metadatos", "Estadísticas"])
    
    with tab1:
        num_preview = st.slider("Número de items a visualizar", 1, 10, 3)
        st.json(consolidated['content'][:num_preview])
    
    with tab2:
        st.json(consolidated['dataset_metadata'])
    
    with tab3:
        st.json(consolidated['statistics'])
    
    # Descarga
    st.header("💾 Descargar Dataset Consolidado")
    
    # Preparar JSON según formato
    if output_format == "Legible (indentado)":
        json_output = json.dumps(consolidated, indent=2, ensure_ascii=False)
    else:
        json_output = json.dumps(consolidated, ensure_ascii=False)
    
    # Agregar schema si está habilitado
    if include_schema:
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["dataset_metadata", "content"],
            "properties": {
                "dataset_metadata": {"type": "object"},
                "source_files": {"type": "array"},
                "content": {"type": "array"},
                "statistics": {"type": "object"}
            }
        }
        full_output = {
            "schema": schema,
            "data": consolidated
        }
        json_output = json.dumps(full_output, indent=2 if output_format == "Legible (indentado)" else None, ensure_ascii=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Descargar JSON Consolidado",
            data=json_output,
            file_name=f"{dataset_name.replace(' ', '_')}_{version}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Descargar configuración
        config_output = json.dumps(st.session_state['config'], indent=2)
        st.download_button(
            label="⚙️ Descargar Configuración",
            data=config_output,
            file_name="consolidation_config.json",
            mime="application/json",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>JSON Consolidator AI v1.0 | Optimizado para entrenamiento de agentes IA</p>
    <p>Desarrollado con Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)
