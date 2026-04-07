"""
analyze_db.py
Analiza secciones y estadísticas de ChromaDB
"""

import chromadb
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
from collections import defaultdict
from config import ProjectConfig

def main():
    config = ProjectConfig()
    db_path = config.get_folder('vector_db')
    
    try:
        client = chromadb.PersistentClient(path=str(db_path))
    except Exception as e:
        print(f"❌ Error conectando a ChromaDB: {e}")
        return
    
    # Obtener colecciones
    try:
        collections = {
            'texto': client.get_collection('fds_textos'),
            'tabla': client.get_collection('fds_tablas'),
            'imagen': client.get_collection('fds_imagens')
        }
    except Exception as e:
        print(f"❌ Error obteniendo colecciones: {e}")
        return
    
    print("\n" + "="*80)
    print("📊 ANÁLISIS DE CHROMADB")
    print("="*80)
    
    # Estadísticas globales
    total_chunks = sum(coll.count() for coll in collections.values())
    print(f"\n✅ Total chunks: {total_chunks}")
    
    for coll_name, collection in collections.items():
        count = collection.count()
        print(f"  • {coll_name}: {count} chunks")
    
    # Análisis por producto
    print("\n" + "="*80)
    print("📦 PRODUCTOS Y SECCIONES")
    print("="*80)
    
    for coll_name, collection in collections.items():
        print(f"\n{'─'*80}")
        print(f"Colección: {coll_name.upper()}")
        print(f"{'─'*80}")
        
        try:
            all_data = collection.get(include=['metadatas'])
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
        
        # Agrupar por producto
        producto_secciones = defaultdict(set)
        
        for metadata in all_data['metadatas']:
            producto = metadata.get('producto', 'N/A')
            seccion = metadata.get('seccion', 'N/A')
            producto_secciones[producto].add(seccion)
        
        # Mostrar resultados
        for producto in sorted(producto_secciones.keys()):
            secciones = sorted(producto_secciones[producto])
            print(f"\n  📄 {producto}")
            print(f"     Secciones ({len(secciones)}):")
            
            # Secciones esperadas
            secciones_esperadas = {
                'identificacion_producto',
                'identificacion_peligros',
                'composicion_componentes',
                'primeros_auxilios',
                'controles_exposicion_proteccion'
            }
            
            secciones_faltantes = secciones_esperadas - set(secciones)
            
            for sec in secciones:
                print(f"       ✓ {sec}")
            
            if secciones_faltantes:
                print(f"\n     ⚠️  Secciones faltantes:")
                for sec in secciones_faltantes:
                    print(f"       ✗ {sec}")
    
    print("\n" + "="*80)
    print("✅ Análisis completado")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()