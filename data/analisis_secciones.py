"""
analyze_db.py
Analiza qué secciones tiene cada producto en ChromaDB
"""

import chromadb
from pathlib import Path
from collections import defaultdict

db_path = Path('/Users/sofiavelandiasierra/Documents/rag-fichas-seguridad/data/vector_db')
client = chromadb.PersistentClient(path=str(db_path))

# Obtener colecciones
collections = {
    'texto': client.get_collection('fds_textos'),
    'tabla': client.get_collection('fds_tablas'),
    'imagen': client.get_collection('fds_imagens')
}

print("\n" + "="*80)
print("ANÁLISIS DE SECCIONES POR PRODUCTO EN CHROMADB")
print("="*80)

for coll_name, collection in collections.items():
    print(f"\n{'='*80}")
    print(f"COLECCIÓN: {coll_name.upper()}")
    print(f"{'='*80}")
    
    # Obtener todos los documentos
    all_data = collection.get(include=['metadatas'])
    
    # Agrupar por producto
    producto_secciones = defaultdict(set)
    
    for metadata in all_data['metadatas']:
        producto = metadata.get('producto', 'N/A')
        seccion = metadata.get('seccion', 'N/A')
        producto_secciones[producto].add(seccion)
    
    # Mostrar resultados
    for producto in sorted(producto_secciones.keys()):
        secciones = sorted(producto_secciones[producto])
        print(f"\n📦 {producto}")
        print(f"   Total secciones: {len(secciones)}")
        print(f"   Secciones:")
        for sec in secciones:
            print(f"     • {sec}")

print("\n" + "="*80)
print("FIN DEL ANÁLISIS")
print("="*80 + "\n")