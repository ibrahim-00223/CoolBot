"""
Fichier : data_cleaning.py
Rôle : Nettoyer et structurer les documents Carrier (PDF, DOCX) en texte brut.
"""

# 1. Imports
#   - PyPDF2, python-docx, os, re, etc.

# 2. Fonctions utilitaires
#   - convert_pdf_to_text(pdf_path: str) -> str
#   - convert_docx_to_text(docx_path: str) -> str
#   - clean_text(text: str) -> str  # Nettoyage (espaces, caractères spéciaux, etc.)

# 3. Fonction principale
#   - clean_and_structure_data(
#       input_dir: str,
#       output_dir: str,
#       categories: list = ["pompes_à_chaleur", "groupes_froids"]
#     ) -> None
#     - Parcourt les fichiers dans input_dir
#     - Convertit chaque fichier en texte brut
#     - Nettoie et classe les textes dans output_dir (par catégorie)

# 4. Bloc main (si exécuté directement)
#   - Exemple d'appel : clean_and_structure_data("data/raw", "data/processed")
