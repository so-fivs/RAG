"""
chromadb_diagnostic.py

Script de diagnóstico para ver qué hay realmente en tu ChromaDB.
"""

import chromadb
from pathlib import Path
from collections import Counter
from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
from config import ProjectConfig
import json

def diagnostic_chromadb():
    """Diagnóstico completo de ChromaDB."""
    
    # Conectar a tu base de datos
    config = ProjectConfig()
    db_path = config.get_folder('vector_db')
    
    if not db_path.exists():
        print(f"❌ No se encontró la base de datos en: {db_path}")
        return
    
    print(f"✓ Conectando a: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))
    
    # Listar colecciones
    collections = client.list_collections()
    print(f"\n{'='*80}")
    print(f"COLECCIONES DISPONIBLES: {len(collections)}")
    print(f"{'='*80}")
    
    for coll in collections:
        print(f"  - {coll.name} ({coll.count()} chunks)")
    
    # Analizar cada colección
    for coll_name in ['fds_textos', 'fds_tablas', 'fds_imagens']:
        try:
            collection = client.get_collection(coll_name)
            
            print(f"\n{'='*80}")
            print(f"COLECCIÓN: {coll_name}")
            print(f"{'='*80}")
            print(f"Total de chunks: {collection.count()}")
            
            # Obtener muestra de chunks
            sample = collection.get(
                limit=50,
                include=['metadatas']
            )
            
            if not sample['metadatas']:
                print("  ⚠️  Sin metadatos disponibles")
                continue
            
            # Analizar productos
            productos = [m.get('producto', 'N/A') for m in sample['metadatas']]
            productos_unicos = Counter(productos)
            
            print(f"\n📦 PRODUCTOS (muestra de {len(productos)}):")
            for prod, count in productos_unicos.most_common(10):
                print(f"  {count:3d}x  {prod[:70]}")
            
            # Analizar secciones
            secciones = [m.get('seccion', 'N/A') for m in sample['metadatas']]
            secciones_unicas = Counter(secciones)
            
            print(f"\n📑 SECCIONES:")
            for sec, count in secciones_unicas.most_common():
                print(f"  {count:3d}x  {sec}")
            
            # Analizar metadatos estructurados
            chunks_con_codigos_h = sum(1 for m in sample['metadatas'] 
                                       if m.get('codigos_h') and len(m.get('codigos_h', [])) > 0)
            chunks_con_codigos_p = sum(1 for m in sample['metadatas'] 
                                       if m.get('codigos_p') and len(m.get('codigos_p', [])) > 0)
            chunks_con_cas = sum(1 for m in sample['metadatas'] 
                                 if m.get('componentes_cas') and len(m.get('componentes_cas', [])) > 0)
            
            print(f"\n🔬 METADATOS ESTRUCTURADOS:")
            print(f"  Chunks con códigos H: {chunks_con_codigos_h}/{len(sample['metadatas'])}")
            print(f"  Chunks con códigos P: {chunks_con_codigos_p}/{len(sample['metadatas'])}")
            print(f"  Chunks con CAS: {chunks_con_cas}/{len(sample['metadatas'])}")
            
            # Mostrar ejemplo de chunk completo
            if sample['metadatas']:
                print(f"\n📄 EJEMPLO DE CHUNK:")
                ejemplo = sample['metadatas'][0]
                print(json.dumps(ejemplo, indent=2, ensure_ascii=False)[:1000] + "...")
        
        except Exception as e:
            print(f"❌ Error al analizar {coll_name}: {e}")
    
    # Prueba de filtros
    print(f"\n{'='*80}")
    print("PRUEBA DE FILTROS")
    print(f"{'='*80}")
    
    try:
        collection = client.get_collection('fds_textos')
        
        # Test 1: Filtro por sección
        print("\n🧪 Test 1: Filtro por sección 'identificacion_producto'")
        results = collection.get(
            where={"seccion": {"$eq": "identificacion_producto"}},
            limit=5,
            include=['metadatas']
        )
        print(f"  Resultados: {len(results['ids'])} chunks")
        if results['metadatas']:
            print(f"  Ejemplo: {results['metadatas'][0].get('producto', 'N/A')[:60]}")
        
        # Test 2: Filtro por producto parcial
        print("\n🧪 Test 2: Filtro por producto con $contains 'Epóxico'")
        try:
            results = collection.get(
                where={"producto": {"$contains": "Epóxico"}},
                limit=5,
                include=['metadatas']
            )
            print(f"  Resultados: {len(results['ids'])} chunks")
            if results['metadatas']:
                for m in results['metadatas'][:3]:
                    print(f"  - {m.get('producto', 'N/A')[:60]}")
        except Exception as e:
            print(f"  ⚠️  $contains no funciona: {e}")
            print("  Intentando con búsqueda directa...")
            
            # Alternativa: buscar todos y filtrar manualmente
            all_results = collection.get(limit=100, include=['metadatas'])
            filtered = [m for m in all_results['metadatas'] 
                       if 'Epóxico' in m.get('producto', '')]
            print(f"  Resultados filtrados manualmente: {len(filtered)}")
            if filtered:
                for m in filtered[:3]:
                    print(f"  - {m.get('producto', 'N/A')[:60]}")
        
    except Exception as e:
        print(f"❌ Error en pruebas de filtros: {e}")


if __name__ == "__main__":
    diagnostic_chromadb()