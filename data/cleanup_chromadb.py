"""
cleanup_chromadb.py
Limpia ChromaDB y permite re-ingestar desde cero
"""

import shutil
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ProjectConfig

def main():
    config = ProjectConfig()
    db_path = config.get_folder('vector_db')
    
    print("🗑️  LIMPIEZA DE CHROMADB")
    print("="*60)
    print(f"Base de datos: {db_path}")
    print("")
    print("⚠️  ADVERTENCIA: Esto eliminará TODA la base de datos")
    print("   Los PDFs y chunks procesados NO se eliminarán")
    print("")
    
    respuesta = input("¿Estás seguro? Escribe 'SI' para confirmar: ")
    
    if respuesta.strip().upper() != 'SI':
        print("❌ Operación cancelada")
        return
    
    print("\n🗑️  Eliminando ChromaDB...")
    if db_path.exists():
        shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        print("✅ ChromaDB eliminada")
    else:
        print("⚠️  ChromaDB ya estaba vacía")
    
    print("\n📋 Siguiente paso:")
    print("  python chromadb_ingestor.py")
    print("\nEsto re-ingirá todos los chunks procesados")

if __name__ == "__main__":
    main()