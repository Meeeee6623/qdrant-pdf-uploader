use std::env;

use pdf_extract::extract_text;
use text_splitter;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{VectorParams, VectorsConfig};
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use serde_json::json;
use uuid::{Uuid};


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    let mut chunk_size = 200; // default chunk size
    let mut debug = false; // debug flag
    let mut collection_name = String::from("test"); // default collection name

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--debug" => debug = true,
            "--collection" => {
                if i + 1 < args.len() {
                    collection_name = args[i + 1].clone();
                    i += 1;
                    if collection_name.is_empty() {
                        println!("Error: Collection name cannot be empty");
                        return Ok(());
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    if args.len() < 2 || args.len() > 6 {
        println!("Usage: {} <path_to_pdf> [chunk_size] [--debug] [--collection <collection_name>]", args[0]);
        return Ok(());
    } else if args.len() >= 3 && args[2] != "--debug" && args[2] != "--collection" {
        chunk_size = args[2].parse::<usize>().unwrap_or(500);
    }


    if debug {
        println!("Debug mode is on");
    }
    // Read the PDF file and extract its text
    let path = &args[1];
    let text = extract_text(path)?;

    println!("Extracted text from PDF file: {}", path);

    if debug {
        println!("Extracted text:");
        println!("{}", text);
    }

    // Split the text into sentences
    use text_splitter::TextSplitter;
    // Can also use anything else that implements the ChunkSizer
    // trait from the text_splitter crate.
    use tiktoken_rs::cl100k_base;

    let tokenizer = cl100k_base().unwrap();
    let splitter = TextSplitter::new(tokenizer)
        // Optionally can also have the splitter trim whitespace for you
        .with_trim_chunks(true);

    let chunks = splitter.chunks(text.as_str(), chunk_size).collect::<Vec<_>>();
    println!("Chunk size: {}", chunk_size);
    println!("Created {} Chunks!", chunks.len());
    if debug {
        println!("Chunks:");
        println!("{:?}", chunks);
    }

    use std::process::Command;

    // Attempt to connect to the Qdrant database
    let mut client = QdrantClient::from_url("http://localhost:6334").build();

    while client.is_err() || client.as_ref().unwrap().list_collections().await.is_err() {
        println!("Qdrant instance with Grpc not detected. Do you want to start a Qdrant instance? (Y/n)");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() != "n" {
            println!("Starting Qdrant instance...");
            let _output = Command::new("docker")
                .args(&["run", "-d", "-p", "6333:6333", "-p", "6334:6334", "-e", "QDRANT__SERVICE__GRPC_PORT=6334", "qdrant/qdrant"])
                .output()
                .expect("Failed to execute command. Make sure Docker is installed and running.");
            println!("Qdrant instance started! You can access the Qdrant dashboard at http://localhost:6333/dashboard/");
            client = QdrantClient::from_url("http://localhost:6334").build();
            // sleep for a second to give the Qdrant instance time to start
            std::thread::sleep(std::time::Duration::from_millis(1000));
        } else {
            return Err(anyhow::anyhow!("Cannot proceed without Qdrant instance"));
        }
    }

    let client = client.unwrap();

    println!("Connected to Qdrant database");
    println!("Collection name: {}", collection_name);

    // Check if collection exists
    let collections_list = client.list_collections().await?;


    // Check if collection already exists
    let exists = collections_list.collections.iter().any(|collection| collection.name == collection_name);

    let mut should_create = true;

    if exists {
        println!("Collection {} already exists", collection_name);
        // prompt user to delete collection
        println!("Do you want to clear the collection (y), or only add to it (n)? (Y/n)");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() == "n" {
            println!("Collection will not be cleared");
            should_create = false;
        } else {
            println!("Clearing collection...");
            client.delete_collection(&collection_name).await?;
            println!("Collection deleted");
        }
    }

    if should_create {
        // Create collection
        client
            .create_collection(&CreateCollection {
                collection_name: collection_name.clone(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: 384, // Size of AllMiniLML6V2 model's embeddings
                        distance: Distance::Cosine.into(),
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await?;
    }

    // Create embedding model
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        show_download_progress: true,
        ..Default::default()
    })?;

    // Embed chunks
    println!("Embedding chunks...");
    let embeddings = model.embed(chunks.clone(), None)?;
    println!("Embedded {} chunks", embeddings.len());
    if debug {
        println!("Embeddings:");
        println!("{:?}", embeddings);
    }

    // Upload embeddings to Qdrant with the payload structure: {file_name: <file_name>, text: <text>, chunk_id: <chunk_id>}
    // where chunk_id is the index of the chunk in the chunks vector
    let file_name = path.split('/').last().unwrap_or("unknown");
    let points = embeddings.iter().enumerate().map(|(i, embedding)| {
        let payload = json!({
            "file_name": file_name,
            "text": chunks[i],
            "chunk_number": i
        })
            .to_string();
        PointStruct::new(Uuid::new_v4().to_string(), embedding.to_vec(), serde_json::from_str(&payload).unwrap())
    }).collect::<Vec<_>>();
    println!("Uploading embeddings to Qdrant...");
    client
        .upsert_points_batch_blocking(collection_name, None, points.clone(), None, 6)
        .await?;
    println!("Uploaded {} embeddings to Qdrant", points.len());
    println!("Data uploaded successfully! See it at http://localhost:6333/dashboard/");

    Ok(())
}