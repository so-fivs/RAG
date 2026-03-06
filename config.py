import sys
from pathlib import Path

class ProjectConfig:
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).resolve().parent
        self.GEMINI_API_KEY = "AIzaSyAd_sMLJqVm4yKDxBVsZQDHQ0hMoTFKrt0"

        # Arquitectura medallón
        self.folders = {
            'bronze':            self.base_path / 'data' / 'bronze',
            'silver':            self.base_path / 'data' / 'silver',
            'gold':              self.base_path / 'data' / 'gold',
            'processed_chunks':  self.base_path / 'data' / 'silver' / 'chunked',
            'vector_db':         self.base_path / 'data' / 'vector_db',
            # Aliases para compatibilidad con scripts existentes
            'raw_documents':     self.base_path / 'data' / 'bronze',
            'data':              self.base_path / 'data',
            'extracted_content': self.base_path / 'data' / 'gold',
        }

        # Subcarpetas gold (antes extracted_content)
        self.extracted_subfolders = {
            'texts':    self.folders['gold'] / 'texts',
            'tables':   self.folders['gold'] / 'tables',
            'images':   self.folders['gold'] / 'images',
            'metadata': self.folders['gold'] / 'metadata',
        }

    def create_folders(self):
        """Crea todas las carpetas necesarias si no existen"""
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        for subfolder in self.extracted_subfolders.values():
            subfolder.mkdir(parents=True, exist_ok=True)
        print("✓ Estructura de carpetas creada correctamente")

    def get_folder(self, folder_name: str) -> Path:
        """Obtiene la ruta de una carpeta por nombre"""
        if folder_name in self.extracted_subfolders:
            return self.extracted_subfolders[folder_name]
        if folder_name in self.folders:
            return self.folders[folder_name]
        return self.base_path

    def get_path(self, folder_type: str, subfolder: str = None) -> Path:
        """Obtiene la ruta de una carpeta específica"""
        if subfolder:
            return self.extracted_subfolders.get(subfolder, self.folders[folder_type])
        return self.folders.get(folder_type, self.base_path)

    def print_structure(self):
        """Muestra la estructura de carpetas configurada"""
        print("\n" + "="*60)
        print("ESTRUCTURA DEL PROYECTO (Arquitectura Medallón)")
        print("="*60)
        print(f"\n📁 Base: {self.base_path}\n")

        print("Carpetas principales:")
        for name, path in self.folders.items():
            status = "✓" if path.exists() else "✗"
            try:
                rel = path.relative_to(self.base_path)
            except ValueError:
                rel = path
            print(f"  {status} {name}: {rel}")

        print("\nSubcarpetas gold (contenido extraído):")
        for name, path in self.extracted_subfolders.items():
            status = "✓" if path.exists() else "✗"
            try:
                rel = path.relative_to(self.base_path)
            except ValueError:
                rel = path
            print(f"  {status} {name}: {rel}")
        print("="*60 + "\n")

    def verify_structure(self) -> bool:
        """Verifica que todas las carpetas críticas existan"""
        critical_folders = ['bronze', 'silver', 'gold', 'vector_db']

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
            'pdfs':   0,
            'chunks': 0,
            'images': 0,
            'tables': 0
        }

        if self.folders['bronze'].exists():
            stats['pdfs'] = len(list(self.folders['bronze'].glob('*.pdf')))

        if self.folders['processed_chunks'].exists():
            stats['chunks'] = len(list(self.folders['processed_chunks'].glob('*.json')))

        if self.extracted_subfolders['images'].exists():
            stats['images'] = len(list(self.extracted_subfolders['images'].glob('*')))

        if self.extracted_subfolders['tables'].exists():
            stats['tables'] = len(list(self.extracted_subfolders['tables'].glob('*.csv')))

        return stats


if __name__ == "__main__":
    config = ProjectConfig()
    config.print_structure()

    print("\n📊 ESTADÍSTICAS DEL PROYECTO:")
    stats = config.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if not config.verify_structure():
        respuesta = input("\n¿Crear estructura de carpetas? (s/n): ")
        if respuesta.lower() == 's':
            config.create_folders()
            config.print_structure()