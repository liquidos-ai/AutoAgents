use autoagents_mistral_rs::run_model;

#[tokio::main]
async fn main() {
    println!("Hello, world!");
    run_model().await.unwrap();
}
