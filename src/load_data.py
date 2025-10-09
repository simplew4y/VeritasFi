import os
import sys
import logging
logging.basicConfig(filename='load_data.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import yaml
import json
from tqdm import tqdm
import shutil
import hashlib
from langchain_community.document_loaders import JSONLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ragManager import RAGManager
from utils.bm25Retriever import load_from_chroma_and_save

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def import_collection_from_dir(rag_manager, collection_name: str, dir_path: str, batch_size: int, ignore_range: bool = False):
        """Load data from a directory into a chroma collection, processing metadata and content.
        
        Args:
            collection_name: Name of the collection to populate
            dir_path: Path to directory containing JSON files
            ignore_range: Whether to ignore page range restrictions when loading
        """
        chroma, ts_chroma = rag_manager._collections[collection_name]
        chroma.reset_collection()
        ts_chroma.reset_collection()

        content_dict = {}  # Maps content hash to a tuple of (content, metadata)
        gid = 0
        title_summaries = set()

        def hash_content(content: str) -> str:
            """Generate a SHA-256 hash of the content."""
            return hashlib.sha256(content.encode('utf-8')).hexdigest()

        filenames = os.listdir(dir_path)
        for filename in filenames:
            if filename.endswith(".json"):
                json_file = os.path.join(dir_path, filename)
                print(json_file)
                loader = JSONLoader(file_path=json_file, jq_schema=".[]", text_content=False)
                documents = loader.load()

                page_range = json.loads(documents[0].page_content)
                print(page_range)
                page_start = page_range['start']
                page_end = page_range['end']
                page_date_published = page_range['date_published']
                count = 0
                for doc in documents[1:]:
                    content_dict_data = json.loads(doc.page_content)
                    content = content_dict_data.get("content", "")
                    page_number = content_dict_data.get("page_number")
                    bundle_id = content_dict_data.get("bundle_id", None)
                    title_summary = content_dict_data.get("title_summary", None)

                    content_hash = hash_content(content)
                    if int(page_start) <= int(page_number) <= int(page_end) or ignore_range:
                        metadata = {
                            "filename": filename,
                            "page_number": page_number,
                            "date_published": page_date_published,
                            "doc_id": content_hash,
                            "global_id": gid,
                        }
                        gid += 1
                        if bundle_id:
                            metadata["bundle_id"] = bundle_id
                        if title_summary:
                            metadata["title_summary"] = title_summary
                            title_summaries.add(title_summary)

                        # Handle duplicates by comparing date_published
                        if content_hash in content_dict:
                            existing_content, existing_metadata = content_dict[content_hash]
                            if page_date_published > existing_metadata["date_published"]:
                                # Replace older content with newer one
                                content_dict[content_hash] = (content, metadata)
                                logger.debug(f"Replacing content file: {existing_metadata['filename']} page: {existing_metadata['page_number']} in {existing_metadata['date_published']} with new version file: {metadata['filename']} page: {metadata['page_number']} in {page_date_published}. Hash: {content_hash}")
                        else:
                            # First encounter of this content hash
                            content_dict[content_hash] = (content, metadata)

                        count += 1

                logger.info(f"{count} chunks processed in {json_file}.")
        logger.info(f"{len(content_dict)} unique chunks loaded in total.")

        # Store title summaries
        title_summaries = list(title_summaries)
        for i in tqdm(range(0, len(title_summaries), batch_size), desc="Storing title summaries"):
            ts_chroma.add_texts(texts=title_summaries[i:i + batch_size])
        logger.info(f"{len(title_summaries)} title summaries stored in ts_{collection_name} collection.")

        # Separate content and metadata for batch storage
        content_list = [item[0] for item in content_dict.values()]  # Extract content
        metadata_list = [item[1] for item in content_dict.values()]  # Extract metadata
        content_hashes_list = [metadata["doc_id"] for metadata in metadata_list]

        # Store the previous chunk id and next chunk id in the metadata
        for i in range(len(metadata_list)):
            # check if the previous chunk has the same filename
            if i > 0 and metadata_list[i]["filename"] == metadata_list[i - 1]["filename"]:
                metadata_list[i]["prev_chunk_id"] = content_hashes_list[i - 1]
            else:
                metadata_list[i]["prev_chunk_id"] = ""
            # check if the next chunk has the same filename
            if i < len(metadata_list) - 1 and metadata_list[i]["filename"] == metadata_list[i + 1]["filename"]:
                metadata_list[i]["next_chunk_id"] = content_hashes_list[i + 1]
            else:
                metadata_list[i]["next_chunk_id"] = ""

        for i in tqdm(range(0, len(content_list), batch_size), desc="Storing database"):
            batch_contents = content_list[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]
            batch_doc_ids = content_hashes_list[i:i + batch_size]
            chroma.add_texts(
                texts=batch_contents,
                metadatas=batch_metadata,
                ids=batch_doc_ids
            )

        logger.info(f"Database stored successfully in {collection_name} collection.")

if __name__ == '__main__':
    key_directory = '/root/autodl-tmp/irelia_pipeline'
    #config_path = "../config/production.yaml"
    config_path = f"{key_directory}/training/config/production.yaml"
    config = load_config(config_path)
    config['persist_directory'] = f"{key_directory}/{config['persist_directory']}"
    config['frequent_qa_directory'] = f"{key_directory}/{config['frequent_qa_directory']}"
    config['qa_table_directory'] = f"{key_directory}/{config['qa_table_directory']}"
    config['qa_table_persist_directory'] = f"{key_directory}/{config['qa_table_persist_directory']}"

    # clear the persist_directory
    if os.path.exists(config['persist_directory']):
        shutil.rmtree(config['persist_directory'])

    rag = RAGManager(config)
    
    # collection0_dir = "/root/autodl-tmp/RAG_Agent_production/tmp/2025_04_all/lotus"
    #collection0_dir = "/root/autodl-tmp/RAG_Agent_thomas/2025_0605_zeekr/zeekr"
    collection0_dir = f"{key_directory}/data/processed_pdf/2025_0805_zeekr/zeekr"
    collection_dirs = [collection0_dir]

    BATCH_SIZE = 100
    
    for collection_dir in collection_dirs:
        collection_name = collection_dir.split('/')[-1]
        logger.info(f"Importing collection {collection_name} from {collection_dir}.")

        rag.create_collection(collection_name)

        # rag.import_collection_from_dir(collection_name, collection_dir, ignore_range=False)
        import_collection_from_dir(rag, collection_name, collection_dir, BATCH_SIZE,  ignore_range=False)

        # Get all documents from the collection
        documents = rag.get_collection_documents(collection_name)
        
        # Create BM25 index directory
        bm25_save_dir = os.path.join(config['persist_directory'], "bm25_index", collection_name)
        
        # Save BM25 index
        load_from_chroma_and_save(documents, bm25_save_dir)
