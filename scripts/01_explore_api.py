"""
explorar la API de MediaWiki de la wiki de Escape from Tarkov.

Esto NO hace scraping de HTML: le pega directo a la API oficial de
MediaWiki, que devuelve el contenido ya estructurado (wikitext).
"""

import requests
import json

API_URL = "https://escapefromtarkov.fandom.com/api.php"

HEADERS = {
    "User-Agent": "TarkovRAG-Portfolio-Project/1.0 (tu-email@ejemplo.com)"
}


def get_page_wikitext(page_title: str) -> str:
    """Trae el wikitext crudo de una página puntual."""
    params = {
        "action": "parse",
        "page": page_title,
        "format": "json",
        "prop": "wikitext",
    }
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data["parse"]["wikitext"]["*"]


def list_pages_in_category(category: str, limit: int = 50) -> list[str]:
    """
    Lista páginas dentro de una categoría, ej: 'Category:Quests'.
    Esto te sirve para después bajar TODAS las misiones automáticamente.
    """
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": limit,
        "format": "json",
    }
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    members = data["query"]["categorymembers"]
    return [m["title"] for m in members]


if __name__ == "__main__":
    print("=== 1) Wikitext crudo de una misión puntual ===")
    wikitext = get_page_wikitext("Debut")
    print(wikitext[:1000])
    print("\n...\n")

    print("=== 2) Primeras páginas dentro de la categoría de misiones ===")
    try:
        quests = list_pages_in_category("Category:Quests", limit=20)
        print(json.dumps(quests, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"No encontré esa categoría exacta ({e}).")
        print("Revisá el nombre real de la categoría en la wiki y ajustalo acá.")