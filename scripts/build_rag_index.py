"""Build the RAG vector index from advisory documents.

Usage:
    python -m scripts.build_rag_index [--recreate]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.rag.indexer import AdvisoryIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build RAG vector index")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection from scratch")
    args = parser.parse_args()

    settings = get_settings()

    # Load advisories
    advisories_path = settings.synthetic_dir / "advisories.json"
    if not advisories_path.exists():
        logger.error(f"Advisories not found at {advisories_path}. Run setup_data.py first.")
        sys.exit(1)

    with open(advisories_path) as f:
        advisories = json.load(f)

    logger.info(f"Loaded {len(advisories)} advisories")

    # Initialize indexer
    indexer = AdvisoryIndexer()

    # Create collection
    indexer.create_collection(recreate=args.recreate)

    # Index documents
    count = indexer.index_advisories(advisories)
    logger.info(f"Indexed {count} advisories")

    # Verify
    info = indexer.get_collection_info()
    logger.info(f"Collection info: {info}")

    # Test retrieval
    from src.rag.retriever import AdvisoryRetriever
    retriever = AdvisoryRetriever()
    results = retriever.retrieve("traffic disruption downtown", top_k=3)
    logger.info(f"Test query returned {len(results)} results")
    for r in results:
        logger.info(f"  - [{r['relevance_score']:.3f}] {r['title']}")


if __name__ == "__main__":
    main()
