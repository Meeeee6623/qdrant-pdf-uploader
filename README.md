# Rust Vectordb CMD

## Overview

`rust-vectordb-cmd` is a command-line tool written in Rust that allows you to extract text from a PDF file, split the text into chunks, embed the chunks using the ALLMiniLML6V2 embedding model, and then upload the embeddings to a Qdrant database. The tool also provides the ability to manage collections in the Qdrant database. \
\
Chunks in the Qdrant database are associated with a payload of the following format:
```json5
{
    "file_name": "example.pdf", // The name of the PDF file, extracted from the path
    "chunk_text": "This is an example chunk of text.", // The text of the chunk
    "chunk_number": 1 // The number of the chunk - this can be used to reconstruct the original text
}
```

This project uses several libraries:

- `pdf-extract`: To extract text from PDF files.
- `qdrant-client`: To interact with the Qdrant database.
- `text-splitter` and `tiktoken-rs`: To split the extracted text into chunks.
- `fastembed`: To embed the chunks of text.

## Running the Project

Before running the project, you need to start a Qdrant docker container. You can do this by running the following command:

```bash
docker run -p 6333:6333 -p 6334:6334 -e QDRANT__SERVICE__GRPC_PORT="6334" qdrant/qdrant
```

After starting the Qdrant docker container, you can run the project using the downloaded binary:

```bash
./rust-vectordb-cmd <path_to_pdf> [chunk_size] [--debug] [--collection <collection_name>]
```

## Usage Guide

The tool accepts the following command-line arguments:

- `<path_to_pdf>`: The path to the PDF file from which to extract text.
- `[chunk_size]`: The size of the chunks into which to split the text. This is optional and defaults to 200.
- `--debug`: A flag that enables debug mode. This is optional.
- `--collection <collection_name>`: The name of the collection in the Qdrant database. This is optional and defaults to "test".

## Code Walkthrough

The tool works in the following steps:

1. It parses the command-line arguments and sets the chunk size, debug flag, and collection name accordingly.

2. It reads the specified PDF file and extracts its text using the `pdf-extract` library.

3. It splits the extracted text into chunks of the specified size. For this, it uses the `text-splitter` library in combination with the `tiktoken-rs` library. Specifically, it uses the `cl100k_base` model from `tiktoken-rs` to tokenize the text, and then splits the tokenized text into chunks.

4. It connects to the Qdrant database using the `qdrant-client` library.

5. It checks if a collection with the specified name already exists in the Qdrant database. If the collection exists and the user wants to delete it, the tool deletes the collection. If the collection does not exist or has been deleted, the tool creates a new collection.

6. It creates a FastText embedding model using the `fastembed` library. Specifically, it uses the `AllMiniLML6V2` model from `fastembed` to embed the chunks of text.

7. It uploads the embeddings to the Qdrant database. Each embedding is associated with a payload that includes the file name, the chunk of text, and the chunk number.

8. Finally, it prints the number of embeddings uploaded to the Qdrant database.

## Future Improvements
Something I would like to include is the ability to choose the embedding model used to embed the chunks of text. I chose the `AllMiniLML6V2` model for this project as it was the best free model with an easily accessible Rust implementation. However, I would like to provide users with the option to choose from a variety of models based on their requirements, including paid models from service providers like OpenAI. 

In addition, I would like to create a command line tool to enable semantic search on the uploaded embeddings, as the embeddings are of no use if they cannot be queried. This tool would allow querying the Qdrant database with semantic search, as well as filters based on the payload data.