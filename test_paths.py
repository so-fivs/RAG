"""
cleanup_chromadb.py

Script para eliminar ChromaDB completamente y empezar de cero.
Úsalo SOLO cuando necesites resetear todo.
"""

import shutil
from pathlib import Path
import sys

# Ajusta esta ruta a tu proyecto
DB_PATH = Path('/Users/sofiavelandiasierra/Documents/RAG/RAG/data/vector_db')

def cleanup_chromadb():
    """Elimina completamente la base de datos ChromaDB."""
    
    print("\n" + "="*60)
    print("LIMPIEZA DE CHROMADB")
    print("="*60)
    
    if not DB_PATH.exists():
        print(f"✅ No hay base de datos en {DB_PATH}")
        return
    
    print(f"\n⚠️  ADVERTENCIA: Esto eliminará TODA la base de datos en:")
    print(f"   {DB_PATH}")
    print(f"\n   Contenido actual:")
    
    # Mostrar contenido
    try:
        for item in DB_PATH.iterdir():
            if item.is_file():
                size = item.stat().st_size / 1024  # KB
                print(f"     📄 {item.name} ({size:.1f} KB)")
            elif item.is_dir():
                print(f"     📁 {item.name}/")
    except Exception as e:
        print(f"     Error listando contenido: {e}")
    
    response = input("\n¿Continuar? (escribe 'ELIMINAR' para confirmar): ")
    
    if response != "ELIMINAR":
        print("\n❌ Operación cancelada")
        return
    
    try:
        shutil.rmtree(DB_PATH)
        print(f"\n✅ Base de datos eliminada exitosamente")
        print(f"   Ahora puedes ejecutar: python chromadb_ingestor.py")
    except Exception as e:
        print(f"\n❌ Error al eliminar: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cleanup_chromadb()