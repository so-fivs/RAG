rom langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from pathlib import Path
import json
import pandas as pd
import sys
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
from config import ProjectConfig


def cargar_y_chunkar_md(markdown_path: Path, metadata_base: dict) -> list[Document]:
    """Divide el Markdown de Docling manteniendo la jerarquía de secciones."""
    markdown_text = markdown_path.read_text(encoding='utf-8')
    
    # 1. Separación por encabezados de la FDS
    headers_to_split_on = [
        ("#", "seccion_titulo"),
        ("##", "subseccion"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs_por_seccion = md_splitter.split_text(markdown_text)
    
    # 2. Refinamiento por tamaño
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks_finales = []
    for doc in docs_por_seccion:
        sub_chunks = char_splitter.split_documents([doc])
        for chunk in sub_chunks:
            chunk.metadata.update(metadata_base)
            chunk.metadata["tipo_contenido"] = "texto"
            chunks_finales.append(chunk)
    # Guardar en silver/chunked/
    silver_path = config.get_folder('data/silver') / f"{nombre_base}_chunks.json"
    silver_data = [{"content": c.page_content, "metadata": c.metadata} for c in chunks_finales]
    silver_path.write_text(json.dumps(silver_data, ensure_ascii=False, indent=2), encoding='utf-8')
    return chunks_finales

def procesar_tablas_como_docs(config, nombre_base: str, metadata_base: dict) -> list[Document]:
    """Convierte las tablas extraídas por Docling (CSVs) en Documentos de LangChain."""
    carpeta_tablas = config.get_folder('tables')
    archivos_tabla = list(carpeta_tablas.glob(f"{nombre_base}_tabla_*.csv"))
    
    docs_tablas = []
    for path_csv in archivos_tabla:
        df = pd.read_csv(path_csv)
        # Convertimos la tabla a texto para el embedding
        tabla_texto = df.to_markdown(index=False) 
        
        meta = metadata_base.copy()
        meta.update({
            "tipo_contenido": "tabla",
            "archivo_csv": path_csv.name
        })
        
        docs_tablas.append(Document(page_content=tabla_texto, metadata=meta))
    return docs_tablas

def ingestar_por_tipo(chunks: list[Document], config):
    """Ingesta en las colecciones clásicas fds_textos y fds_tablas."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db_path = str(config.get_folder('vector_db'))
    
    # Separar chunks por tipo
    chunks_texto = [c for c in chunks if c.metadata["tipo_contenido"] == "texto"]
    chunks_tabla = [c for c in chunks if c.metadata["tipo_contenido"] == "tabla"]

    # Ingestar Textos
    if chunks_texto:
        print(f"-> Guardando {len(chunks_texto)} chunks en fds_textos")
        db_txt = Chroma.from_documents(
            documents=chunks_texto,
            embedding=embeddings,
            collection_name="fds_textos",
            persist_directory=db_path
        )

    # Ingestar Tablas
    if chunks_tabla:
        print(f"-> Guardando {len(chunks_tabla)} tablas en fds_tablas")
        db_tab = Chroma.from_documents(
            documents=chunks_tabla,
            embedding=embeddings,
            collection_name="fds_tablas",
            persist_directory=db_path
        )

def pipeline_reingesta_total(config):
    """Pipeline que une Docling con la reingesta clasificada."""
    from extrac_pdf_docling import pipeline_fds_docling
    
    pdfs = list(config.get_folder('data/bronze').glob("*.pdf"))
    todos_los_chunks = []

    for i, pdf_path in enumerate(pdfs):
        print(f"\n--- [{i+1}/{len(pdfs)}] {pdf_path.name} ---")
        
        # 1. Ejecutar tu nuevo extractor Docling
        res_docling = pipeline_fds_docling(str(pdf_path), config, primera_ejecucion=(i==0))
        
        # 2. Cargar Metadata recién generada
        meta_path = config.get_folder('metadata') / f"{pdf_path.stem}.json"
        with open(meta_path, 'r') as f:
            meta_fds = json.load(f)
            
        metadata_base = {
            "producto": meta_fds["nombre_producto"],
            "archivo": pdf_path.name,
            "h_codes": str([h["codigo"] for h in meta_fds["h_codes"]]) # Guardamos como string para Chroma
        }

        # 3. Chunking de Texto
        md_path = config.get_folder('texts') / f"{pdf_path.stem}.md"
        chunks_txt = cargar_y_chunkar_md(md_path, metadata_base)
        
        # 4. Procesamiento de Tablas
        chunks_tab = procesar_tablas_como_docs(config, pdf_path.stem, metadata_base)
        
        todos_los_chunks.extend(chunks_txt)
        todos_los_chunks.extend(chunks_tab)

    # 5. Ingesta final organizada
    ingestar_por_tipo(todos_los_chunks, config)
    print("\n Proceso terminado. Colecciones fds_textos y fds_tablas actualizadas.")

if __name__ == "__main__":
    from config import ProjectConfig
    pipeline_reingesta_total(ProjectConfig())
