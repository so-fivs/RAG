import os
from pathlib import Path

class ProjectConfig:
    """Configuración centralizada del proyecto"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).resolve().parent
        self.GEMINI_API_KEY = "AIzaSyAd_sMLJqVm4yKDxBVsZQDHQ0hMoTFKrt0" 

        
        # Estructura de carpetas 
        self.folders = {
            'raw_documents': self.base_path / 'raw_documents',  # Fuera de data
            'data': self.base_path / 'data',  # Carpeta principal de data
            'extracted_content': self.base_path / 'data' / 'extracted_content',
            'processed_chunks': self.base_path / 'data' / 'processed_chunks',
            'vector_db': self.base_path  / 'data' / 'vector_db',

        }
        
        # Subcarpetas para contenido extraído
        self.extracted_subfolders = {
            'texts': self.folders['extracted_content'] / 'texts',
            'tables': self.folders['extracted_content'] / 'tables',
            'images': self.folders['extracted_content'] / 'images',
            'metadata': self.folders['extracted_content'] / 'metadata'
        }
    
    def create_folders(self):
        """Crea todas las carpetas necesarias si no existen"""
        # Crear carpetas principales
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        
        # Crear subcarpetas de contenido extraído
        for subfolder in self.extracted_subfolders.values():
            subfolder.mkdir(parents=True, exist_ok=True)
        
        print("✓ Estructura de carpetas creada correctamente")
    
    def get_folder(self, folder_name: str) -> Path:
        """Obtiene la ruta de una carpeta por nombre"""
        # Buscar primero en subcarpetas
        if folder_name in self.extracted_subfolders:
            return self.extracted_subfolders[folder_name]
        # Luego en carpetas principales
        if folder_name in self.folders:
            return self.folders[folder_name]
        # Si no existe, retornar base_path
        return self.base_path
    
    def get_path(self, folder_type: str, subfolder: str = None) -> Path:
        """Obtiene la ruta de una carpeta específica"""
        if subfolder:
            return self.extracted_subfolders.get(subfolder, self.folders[folder_type])
        return self.folders.get(folder_type, self.base_path)
    
    def print_structure(self):
        """Muestra la estructura de carpetas configurada"""
        print("\n" + "="*60)
        print("ESTRUCTURA DEL PROYECTO")
        print("="*60)
        print(f"\n📁 Base: {self.base_path}\n")
        
        print("Carpetas principales:")
        for name, path in self.folders.items():
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {name}: {path.relative_to(self.base_path)}")
        
        print("\nSubcarpetas de contenido extraído:")
        for name, path in self.extracted_subfolders.items():
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {name}: {path.relative_to(self.base_path)}")
        print("="*60 + "\n")
    def verify_structure(self) -> bool:
        """Verifica que todas las carpetas críticas existan"""
        critical_folders = ['raw_documents', 'data', 'vector_db']
        
        missing = []
        for folder_name in critical_folders:
            if folder_name not in self.folders:
                missing.append(folder_name)
            elif not self.folders[folder_name].exists():
                missing.append(folder_name)
        
        if missing:
            print(f"❌ Carpetas faltantes: {missing}")
            return False
        
        print("✅ Estructura verificada correctamente")
        return True

    def get_stats(self) -> dict:
        """Retorna estadísticas del proyecto"""
        stats = {
            'pdfs': 0,
            'chunks': 0,
            'images': 0,
            'tables': 0
        }
        
        # Contar PDFs
        if self.folders['raw_documents'].exists():
            stats['pdfs'] = len(list(self.folders['raw_documents'].glob('*.pdf')))
        
        # Contar chunks
        if self.folders['processed_chunks'].exists():
            stats['chunks'] = len(list(self.folders['processed_chunks'].glob('*.json')))
        
        # Contar imágenes
        if self.extracted_subfolders['images'].exists():
            stats['images'] = len(list(self.extracted_subfolders['images'].glob('*')))
        
        # Contar tablas
        if self.extracted_subfolders['tables'].exists():
            stats['tables'] = len(list(self.extracted_subfolders['tables'].glob('*.csv')))
        
        return stats
if __name__ == "__main__":
    # Prueba la configuración
    config = ProjectConfig()
    config.print_structure()

    # Mostrar estadísticas
    print("\n📊ESTADÍSTICAS DEL PROYECTO:")
    stats = config.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Verificar estructura
    if not config.verify_structure():
        respuesta = input("\n¿Crear estructura de carpetas? (s/n): ")
        if respuesta.lower() == 's':
            config.create_folders()
            config.print_structure()    