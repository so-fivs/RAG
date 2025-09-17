import os
from pathlib import Path

class ProjectConfig:
    """Configuración centralizada del proyecto"""

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()

        # Estructura de carpetas
        self.folders = {
            'raw_data': self.base_path / 'data' / 'raw_documents',
            'extracted_content': self.base_path / 'data' / 'extracted_content',
            'processed_chunks': self.base_path / 'data' / 'processed_chunks',
            'embeddings': self.base_path / 'data' / 'embeddings',
            'vector_db': self.base_path / 'data' / 'vector_db',
            'logs': self.base_path / 'logs',
            'outputs': self.base_path / 'outputs'
        }

        # Subcarpetas para contenido extraído
        self.extracted_subfolders = {
            'texts': self.folders['extracted_content'] / 'texts',
            'tables': self.folders['extracted_content'] / 'tables',
            'images': self.folders['extracted_content'] / 'images',
            'metadata': self.folders['extracted_content'] / 'metadata'
        }

    def get_path(self, folder_type: str, subfolder: str = None) -> Path:
        """Obtiene la ruta de una carpeta específica"""
        if subfolder:
            return self.extracted_subfolders.get(subfolder, self.folders[folder_type])
        return self.folders.get(folder_type, self.base_path)